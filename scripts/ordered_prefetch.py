"""#RSNA #Kaggle #Tecnologia — um preparo futuro, ordem estável e falhas explícitas."""
from concurrent.futures import ThreadPoolExecutor


def ordered_one_ahead(items, prepare):
    """O consumidor deve fechar o gerador em saída antecipada (closing).

    Um único produtor; nunca chama prepare simultaneamente em dois itens.
    No máximo um resultado futuro além do item corrente. O shutdown espera
    o preparo ativo antes de permitir mudança de receita/global do leitor.
    """
    source = iter(enumerate(items))
    first = next(source, None)
    if first is None:
        return
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='e03-prepare')
    future = None
    try:
        index, item = first
        future = pool.submit(prepare, item)
        while True:
            value = future.result()  # Propaga erro; nunca preenche fallback.
            following = next(source, None)
            future = pool.submit(prepare, following[1]) if following is not None else None
            yield index, item, value
            del value
            if following is None:
                break
            index, item = following
    finally:
        if future is not None:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
