from __future__ import annotations

import os


def resolve_worker_count(worker_count: int, *, allow_zero: bool = False) -> int:
    if worker_count == -1:
        return max(1, os.cpu_count() or 1)

    minimum = 0 if allow_zero else 1
    if worker_count < minimum or (not allow_zero and worker_count == 0):
        if allow_zero:
            raise ValueError("`worker_count` trebuie să fie 0, un număr pozitiv sau -1 pentru auto.")
        raise ValueError("`worker_count` trebuie să fie un număr pozitiv sau -1 pentru auto.")

    return worker_count