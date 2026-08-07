#!/usr/bin/env python3
"""Osetljivost na FREQ_GUARD_RATIO (Tabela/pasus 5.4) sa NOVIM v2p tagerom.

Uvozi guard_sensitivity i preusmerava SAMO tager i izlaz, po uzoru na
reeval_diacritics_v2p.py. Stari rezultat ostaje netaknut.

Skripta PUCA ako neki od atributa koje preusmerava ne postoji: tiho propusten
override je vec jednom pregazio postojece rezultate.

Izlaz: results/guard_sensitivity_v2p.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import guard_sensitivity as g

for attr in ('V5_MODEL', 'OUT'):
    if not hasattr(g, attr):
        raise SystemExit(f'guard_sensitivity nema atribut {attr} - preusmeravanje '
                         f'bi bilo tiho propusteno; skripta se zaustavlja')

g.V5_MODEL = os.path.join(g.BASE_DIR, 'NLTK Treniranje', 'models_v2p_udonly',
                          'perceptron-tagger-v2p-udonly.pickle')
g.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results',
                     'guard_sensitivity_v2p.json')

if not os.path.exists(g.V5_MODEL):
    raise SystemExit(f'nema modela: {g.V5_MODEL}')

if __name__ == '__main__':
    print(f'TAGER: {g.V5_MODEL}')
    print(f'IZLAZ: {g.OUT}\n', flush=True)
    g.main()
