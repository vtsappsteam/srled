#!/usr/bin/env python3
"""Kompletan rerun svih accuracy brojeva rada sa NOVIM v2p tagerom.

Uvozi final_system_eval i evaluate_three_testsets, preusmerava SAMO putanju
tagera (V7_MODEL -> v2p combined). Sve ostalo (leksikoni, korekcije, CLASSLA
per-token vektori, statistika) ostaje zaključano - pa brojevi u tager-nezavisnim
ćelijama (gold-tag uslovi) MORAJU ispasti identični starim: to je ugrađena
kontrola da ništa drugo nije promenjeno.

Izlaz: results/final_system_eval_v2p.json (stari final_system_eval.json netaknut)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import evaluate_three_testsets as base  # noqa: E402
base.V7_MODEL = str(base.BASE / 'NLTK Treniranje' / 'models_v2p_pnpa' / 'perceptron-tagger-v2p-combined.pickle')

import final_system_eval as fse  # noqa: E402  (uvozi base koji je već preusmeren)
fse.OUT_JSON = fse.SCRIPT_DIR.parent / 'results' / 'final_system_eval_v2p.json'

if __name__ == '__main__':
    print(f'TAGER: {base.V7_MODEL}')
    print(f'IZLAZ: {fse.OUT_JSON}\n')
    fse.main()
