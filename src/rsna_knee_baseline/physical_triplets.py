"""#RSNA #Kaggle #Pesquisa — strict geometry and paired triplet sampling.

Own implementation motivated by Kaggle Geometry to 6 Slots and discussion735154.
No inferred geometry, mixed-orientation sorting, or silent series exclusions.
"""
import numpy as np


def physical_plan(records, original_files):
    if not records or len(original_files) != 3:
        raise ValueError('Empty series or invalid original triplet')
    names = [str(name) for name, _ in records]
    if len(set(names)) != len(names) or not set(original_files) <= set(names):
        raise ValueError('Duplicate/missing files')
    orientations, positions = [], []
    for name, header in records:
        iop = np.asarray(getattr(header, 'ImageOrientationPatient', []), dtype=np.float64)
        ipp = np.asarray(getattr(header, 'ImagePositionPatient', []), dtype=np.float64)
        if iop.shape != (6,) or ipp.shape != (3,) or not np.isfinite(iop).all() or not np.isfinite(ipp).all():
            raise ValueError('Missing/nonfinite IOP/IPP')
        row, col = iop[:3], iop[3:]
        if (abs(np.linalg.norm(row)-1) > 1e-3 or abs(np.linalg.norm(col)-1) > 1e-3
                or abs(np.dot(row, col)) > 1e-3):
            raise ValueError('Invalid direction cosines')
        orientations.append(iop); positions.append(ipp)
    orientations = np.asarray(orientations)
    if np.max(np.abs(orientations-orientations[0])) > 1e-3:
        raise ValueError('Inconsistent in-plane orientation')
    normal = np.cross(orientations[0, :3], orientations[0, 3:])
    normal /= np.linalg.norm(normal)
    projections = np.asarray(positions) @ normal
    order = np.argsort(projections, kind='stable')
    ordered = [names[i] for i in order]
    distances = projections[order]
    if len(distances) > 1 and np.min(np.diff(distances)) <= 1e-4:
        raise ValueError('Duplicate/nonseparated physical locations')
    n = len(ordered)
    quartiles = [int(round(q*(n-1))) for q in [.25, .5, .75]]
    adjacent = [max(0, min(n-1, quartiles[1]+d)) for d in [-1, 0, 1]]
    plans = {'physical_quartiles': quartiles, 'physical_adjacent': adjacent}
    result = {'n_slices': n, 'normal': normal.tolist(), 'ordered_files': ordered,
              'positions_mm': distances.tolist(), 'center_index': quartiles[1], 'arms': {}}
    locations = dict(zip(ordered, distances.tolist()))
    result['original_positions_mm'] = [locations[name] for name in original_files]
    for arm, indices in plans.items():
        files = [ordered[i] for i in indices]
        result['arms'][arm] = {'indices': indices, 'files': files,
                               'positions_mm': distances[indices].tolist(),
                               'gaps_mm': np.diff(distances[indices]).tolist()}
    return result
