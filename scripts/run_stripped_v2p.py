#!/usr/bin/env python3
"""Stripped-ReLDI test split rerun: (1) kontrola sa zlatnim MSD (tager ne
učestvuje - mora reprodukovati stare brojeve), (2) predviđeni MSD sa NOVIM
v2p tagerom. Stari rezultati ostaju netaknuti (novi izlazni fajlovi)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import evaluate_three_testsets as ev3
ev3.V7_MODEL = str(ev3.BASE / 'NLTK Treniranje' / 'models_v2p_pnpa'
                   / 'perceptron-tagger-v2p-combined.pickle')

import stripped_test_split_eval as g
g.OUT = str(Path(g.OUT).parent / 'stripped_reldi_test_split_v2p_control.json')
print('=== [1] KONTROLA (gold MSD, tager ne učestvuje) ===', flush=True)
print(f'IZLAZ: {g.OUT}', flush=True)
g.main()

import stripped_predicted_msd_eval as p
p_out = Path(__file__).parent.parent / 'results' / 'stripped_reldi_predicted_msd_v2p.json'
if hasattr(p, 'OUT'):
    p.OUT = str(p_out)
print('\n=== [2] PREDVIĐENI MSD (novi v2p tager) ===', flush=True)
print(f'TAGER: {ev3.V7_MODEL}', flush=True)
p.main()
