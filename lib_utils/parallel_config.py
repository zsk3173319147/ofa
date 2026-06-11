from __future__ import annotations

import os


_INNER_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _positive_int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"Ignoring invalid {name}={value!r}; using {default}")
        return default
    return max(parsed, 1)


def get_ofa_num_workers(default: int = 16) -> int:
    cpu_count = os.cpu_count() or 1
    requested = _positive_int_from_env("OFA_NUM_WORKERS", default)
    return max(1, min(requested, cpu_count))


def configure_cpu_parallelism(default_inner_threads: int = 1) -> None:
    inner_threads = _positive_int_from_env("OFA_INNER_NUM_THREADS", default_inner_threads)
    for name in _INNER_THREAD_ENV_VARS:
        os.environ.setdefault(name, str(inner_threads))

    if os.environ.get("OFA_SET_TORCH_THREADS", "1").lower() in {"0", "false", "no"}:
        return

    try:
        import torch
    except Exception:
        return

    torch_threads = _positive_int_from_env("OFA_TORCH_NUM_THREADS", inner_threads)
    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass

    interop_threads = _positive_int_from_env("OFA_TORCH_INTEROP_THREADS", inner_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass
