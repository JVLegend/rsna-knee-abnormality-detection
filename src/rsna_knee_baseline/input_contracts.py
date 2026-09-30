"""#RSNA #Kaggle #Pesquisa — explicit channel ablation and partial DINO training.

Apply the channel mapping to NCHW triplets BEFORE encoder normalization.
Changing a deployed checkpoint's contract is forbidden, even if shapes match.
"""
import numpy as np

CONTRACTS = ('physical_adjacent', 'center_repeated')


def triplet_channels(images, contract):
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[1] != 3 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite NCHW triplets')
    if contract not in CONTRACTS:
        raise ValueError('Unknown channel contract')
    return x.copy() if contract == CONTRACTS[0] else np.repeat(x[:, 1:2], 3, axis=1)


def require_checkpoint_contract(trained, inference):
    required = {'channels', 'resolution', 'normalization', 'geometry_sha256'}
    if set(trained) != required or set(inference) != required or trained != inference:
        raise ValueError('Checkpoint/input contract mismatch')
    if trained['channels'] not in CONTRACTS:
        raise ValueError('Invalid checkpoint channel contract')
    if type(trained['resolution']) is not int or trained['resolution'] < 1:
        raise ValueError('Invalid resolution')
    sha = trained['geometry_sha256']
    if not isinstance(trained['normalization'], str) or not trained['normalization']:
        raise ValueError('Missing preprocessing identity')
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Invalid geometry digest')


def configure_dino_tail(model, last_blocks=2):
    """Hugging Face DINO encoder contract; never guess timm/ConvNeXt structure.

Call after model.train() at the beginning of EACH training epoch to retain
the frozen prefix in eval mode. Optimizer must contain returned parameters.
Supports last_blocks=0 as paired frozen-backbone control.
"""
    if not hasattr(model, 'encoder') or not hasattr(model.encoder, 'layer') or not hasattr(model, 'layernorm'):
        raise ValueError('Expected HF DINO encoder.layer and layernorm')
    layers = model.encoder.layer
    if type(last_blocks) is not int or not 0 <= last_blocks <= len(layers):
        raise ValueError('Invalid number of trainable blocks')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if last_blocks:
        for layer in list(layers)[-last_blocks:] + [model.layernorm]:
            layer.train()
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable, {'last_blocks': last_blocks,
                       'trainable_parameters': sum(p.numel() for p in trainable),
                       'total_parameters': sum(p.numel() for p in model.parameters())}
