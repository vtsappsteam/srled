#!/usr/bin/env python3
"""Benchmark optimizovane implementacije sa NOVIM v2p tagerom.

Uvozi benchmark_optimized, preusmerava TAGGER_DIR i OUT_PATH (stari
benchmark_optimized.json OSTAJE - potreban za poređenje staro/novo).
Metodologija netaknuta: warm-up, REPS=5, mean±std, peak RSS, fork batch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_optimized as b

_new_tagger = b.BASE / 'NLTK Treniranje' / 'models_v2p_fast_pnpa'
_new_out = Path(__file__).parent.parent / 'results' / 'benchmark_optimized_v2p.json'
assert hasattr(b, 'TAGGER_DIR') and hasattr(b, 'OUT_PATH'), 'konstante ne postoje!'
assert _new_tagger.exists(), f'nema {_new_tagger}'
b.TAGGER_DIR = _new_tagger
b.OUT_PATH = _new_out

if __name__ == '__main__':
    print(f'TAGER: {b.TAGGER_DIR}')
    print(f'IZLAZ: {b.OUT_PATH}\n')
    b.main()
