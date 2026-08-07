#!/usr/bin/env python3
"""Restauracija dijakritika (Tabela 5) sa NOVIM v2p UD-only tagerom.

Uvozi reeval_diacritics_v6 i preusmerava SAMO tager: stari v5 (srWaC v3 + UD
fine-tune, čist P) -> novi v2p_udonly (v2p corrected + UD fine-tune, Pn/Pa).
Napomena o supstituciji: v2p linija nema čist-P varijantu (korpus je Pn/Pa);
restauracioni disambiguator koristi grubi POS (prvo slovo), a Pn*/Pa* počinju
sa P kao i pre, pa je evidencija ekvivalentna. Ovo dokumentovati u radu ako
brojevi uđu. Izlaz: results/evaluation_v2p_restoration.json (staro netaknuto).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import reeval_diacritics_v6 as r

r.V5_MODEL = os.path.join(r.BASE_DIR, 'NLTK Treniranje', 'models_v2p_udonly',
                          'perceptron-tagger-v2p-udonly.pickle')
r.OUT = os.path.join(os.path.dirname(__file__), '..', 'results',
                     'evaluation_v2p_restoration.json')

if __name__ == '__main__':
    print(f'TAGER: {r.V5_MODEL}')
    print(f'IZLAZ: {r.OUT}\n')
    r.main()
