#!/usr/bin/env python3
"""Pun identity check optimizovanog v2p tagera na SVIM evaluacionim skupovima
(UD test+dev, ReLDI test, SrpKor test = 61,778 tokena). Uvozi verify_identity_all
i preusmerava putanje na nove modele - logika netaknuta."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import verify_identity_all as v

v.NLTK_MODEL = v.BASE / 'NLTK Treniranje' / 'models_v2p_pnpa' / 'perceptron-tagger-v2p-combined.pickle'
v.FAST_DIR = v.BASE / 'NLTK Treniranje' / 'models_v2p_fast_pnpa'

if __name__ == '__main__':
    print(f'NLTK:  {v.NLTK_MODEL}')
    print(f'FAST:  {v.FAST_DIR}\n')
    v.main()
