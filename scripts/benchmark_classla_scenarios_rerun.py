#!/usr/bin/env python3
"""CLASSLA scenariji (batch + single-stream) - sveža sesija, preusmeren izlaz."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_classla_scenarios as b
assert hasattr(b, 'OUT'), 'OUT ne postoji!'
b.OUT = str(Path(__file__).parent.parent / 'results' / 'benchmark_classla_scenarios_rerun.json')
if __name__ == '__main__':
    print(f'IZLAZ: {b.OUT}\n')
    b.main()
