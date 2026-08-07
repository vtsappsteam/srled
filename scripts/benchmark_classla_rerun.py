#!/usr/bin/env python3
"""CLASSLA benchmark - sveža sesija (par sa benchmark_optimized_v2p).
Preusmeren izlaz da stari benchmark_classla.json ostane netaknut."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_classla as b
assert hasattr(b, 'OUT_PATH'), 'OUT_PATH ne postoji!'
b.OUT_PATH = Path(__file__).parent.parent / 'results' / 'benchmark_classla_rerun.json'
if __name__ == '__main__':
    print(f'IZLAZ: {b.OUT_PATH}\n')
    b.main()
