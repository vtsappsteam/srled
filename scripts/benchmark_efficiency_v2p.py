#!/usr/bin/env python3
"""Naivna implementacija (Sekcija 5.10) merena sa NOVIM v2p tagerom.

Daje cetiri broja koja rad jos nosi iz starog merenja: propusnost naivne
implementacije, njeno vreme ucitavanja, njen peak RSS, i propusnost
lemma-only rezima sa hash-tabelama. Naivna implementacija ucitava perceptron
kroz NLTK, pa v2p pickle (veci model) menja i vreme ucitavanja i memoriju.

Preusmerava SAMO tager i izlaz; puca ako atribut ne postoji, da
preusmeravanje ne bi bilo tiho propusteno.

Merenje je vremensko, pa skripta odbija da pocne ako je Low Power Mode
ukljucen ili je masina opterecena: Low Power Mode se ume ukljuciti sam posle
restarta i tada obori propusnost na pola.

Pokretanje: python3 scripts/benchmark_efficiency_v2p.py
Izlaz: results/benchmark_efficiency_v2p.json
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Prag load average-a. Na masini sa otvorenim GUI-jem load raste i kad je CPU
# uglavnom neaktivan, pa se prag moze podici preko BENCH_MAX_LOAD, ali samo
# svesno i uz proveru da CPU zaista jeste slobodan (iostat idle).
MAX_LOAD = float(os.environ.get('BENCH_MAX_LOAD', '2.5'))


def machine_is_clean():
    lpm = subprocess.run(['pmset', '-g'], capture_output=True, text=True).stdout
    for line in lpm.splitlines():
        if 'lowpowermode' in line:
            if line.split()[-1] != '0':
                return False, f'Low Power Mode ukljucen: {line.strip()}'
    load1 = os.getloadavg()[0]
    if load1 > MAX_LOAD:
        return False, f'load average {load1:.2f} > {MAX_LOAD}'
    return True, f'LPM iskljucen, load {load1:.2f}'


import benchmark_efficiency as b  # noqa: E402

for attr in ('V7_MODEL', 'OUT_PATH'):
    if not hasattr(b, attr):
        raise SystemExit(f'benchmark_efficiency nema atribut {attr} - preusmeravanje '
                         f'bi bilo tiho propusteno; skripta se zaustavlja')

b.V7_MODEL = str(b.BASE / 'NLTK Treniranje' / 'models_v2p_pnpa'
                 / 'perceptron-tagger-v2p-combined.pickle')
b.OUT_PATH = b.OUT_PATH.parent / 'benchmark_efficiency_v2p.json'

if __name__ == '__main__':
    ok, why = machine_is_clean()
    print(f'Stanje masine: {why}')
    if not ok:
        raise SystemExit('merenje odbijeno - ocistiti masinu pa ponoviti')
    if not os.path.exists(b.V7_MODEL):
        raise SystemExit(f'nema modela: {b.V7_MODEL}')
    print(f'TAGER: {b.V7_MODEL}')
    print(f'IZLAZ: {b.OUT_PATH}\n', flush=True)
    b.main()
