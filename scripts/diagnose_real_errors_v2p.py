#!/usr/bin/env python3
"""Analiza gresaka restauracije (Tabela 7) sa NOVIM v2p tagerom.

Uvozi diagnose_real_errors i preusmerava SAMO tager i izlaz. Stari dump
(real_errors_diacritics_v6.json) ostaje netaknut radi poredjenja.

Skripta PUCA ako atribut koji preusmerava ne postoji, da preusmeravanje ne bi
bilo tiho propusteno.

Izlaz: results/real_errors_diacritics_v2p.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose_real_errors as d

for attr in ('V5_MODEL', 'OUT_PATH'):
    if not hasattr(d, attr):
        raise SystemExit(f'diagnose_real_errors nema atribut {attr} - preusmeravanje '
                         f'bi bilo tiho propusteno; skripta se zaustavlja')

d.V5_MODEL = os.path.join(d.BASE_DIR, 'NLTK Treniranje', 'models_v2p_udonly',
                          'perceptron-tagger-v2p-udonly.pickle')
d.OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results',
                          'real_errors_diacritics_v2p.json')

if not os.path.exists(d.V5_MODEL):
    raise SystemExit(f'nema modela: {d.V5_MODEL}')

if __name__ == '__main__':
    print(f'TAGER: {d.V5_MODEL}')
    print(f'IZLAZ: {d.OUT_PATH}\n', flush=True)
    d.main()
