from multiprocessing import Process, Queue
from typing import Any, Callable, Iterable, List


def run(func: Callable, q_in, q_out) -> None:
    """Run function in parallel."""
    while True:
        i, x = q_in.get()
        if i < 0 or x is None:
            break
        q_out.put((i, func(*x[0])))


def pmap(func: Callable, inputs: Iterable[Any], nprocs: int) -> List[Any]:
    """Parallel map function."""
    q_in = Queue(1)
    q_out = Queue()

    proc = [Process(target=run, args=(func, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    count = 0
    for i, x in enumerate(inputs):
        q_in.put((i, (x,)))
        count += 1
    for _ in range(nprocs):
        q_in.put((-1, None))
    res = [q_out.get() for _ in range(count)]
    for p in proc:
        p.join()

    return [x for _, x in sorted(res)]
