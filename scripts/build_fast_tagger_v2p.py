#!/usr/bin/env python3
"""Izgradnja optimizovanog tagera iz NOVOG v2p combined modela.

Ne kopira logiku: uvozi build_fast_tagger i samo mu preusmeri ulaz/izlaz.
Stari models_v8_fast_pnpa ostaje netaknut (potreban za poređenje/rollback dok
novi brojevi ne uđu u rad). Ugrađeni identity check (UD test) je isti;
pun identity na svim evaluacionim skupovima radi verify_identity_all_v2p.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_fast_tagger as b  # noqa: E402

b.NLTK_MODEL = b.BASE / 'NLTK Treniranje' / 'models_v2p_pnpa' / 'perceptron-tagger-v2p-combined.pickle'
b.OUT_DIR = b.BASE / 'NLTK Treniranje' / 'models_v2p_fast_pnpa'

if __name__ == '__main__':
    print(f'ULAZ:  {b.NLTK_MODEL}')
    print(f'IZLAZ: {b.OUT_DIR}\n')
    b.main()
