"""Reproduce every table and figure: python -m experiments.run_all"""
from __future__ import annotations

import time

from . import e1_optimality, e2_methods, e3_weights, e4_real, e5_fairness, e6_tradeoff, e7_delays


def main():
    for mod in (e1_optimality, e5_fairness, e6_tradeoff, e2_methods, e3_weights, e4_real, e7_delays):
        t = time.perf_counter()
        print(f"\n=== {mod.__name__} ===", flush=True)
        mod.main()
        print(f"--- {mod.__name__} done in {time.perf_counter() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
