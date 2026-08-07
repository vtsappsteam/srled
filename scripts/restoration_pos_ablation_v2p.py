#!/usr/bin/env python3
"""POS-ablacija restauracionog modula na SETimes.SR, sa v2p tagerom.

Rad tvrdi da tacnost restauracije pada "sa 99,57% na 99,56%" kad se POS
evidencija iskljuci. Puna vrednost je posle retreninga tagera 99,56, pa se
par mora ponovo izmeriti i ispisati sa dovoljno decimala da razlika uopste
bude vidljiva.

Dve konfiguracije nad istim ulazom:
  A) pun modul, tag daje v2p_udonly tager,
  B) pun modul, tag zamenjen neinformativnim ('X'): POS filtriranje, Np
     rutiranje i V-Q unakrsna provera ne mogu da se okinu, dok obrasci,
     frekvencija i OOV resursi ostaju.

IDENTITY: konfiguracija B ne koristi tager, pa mora reprodukovati stari
ablacioni rezultat (74.402 tacno, 99,56%, 98,12% na dijakritik-nosecim
tokenima) iz repo/results/restoration_no_pos_ablation.json. Bez tog PASS-a
skripta ne pise izlaz - to bi znacilo da ablacija nije ista kao ranije.

Izlaz: results/restoration_pos_ablation_v2p.json
"""
import json
import math
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from candidate_generator import CandidateGenerator          # noqa: E402
from pos_disambiguator import POSDisambiguator              # noqa: E402
from evaluator import DiacriticsEvaluator, load_setimes_sentences  # noqa: E402
from diacritics import strip_diacritics                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(HERE, '..', '..')
# repo/scripts layout is one level deeper than scripts/
if (not os.path.isdir(os.path.join(BASE_DIR, 'POS-Aware-Stemmer'))
        and os.path.isdir(os.path.join(HERE, '..', '..', '..', 'POS-Aware-Stemmer'))):
    BASE_DIR = os.path.join(HERE, '..', '..', '..')
TAGGER_PATH = os.path.join(BASE_DIR, 'NLTK Treniranje', 'models_v2p_udonly',
                           'perceptron-tagger-v2p-udonly.pickle')
SRLEX_PATH = os.path.join(BASE_DIR, 'POS-Aware-Stemmer', 'data', 'srLex_v1.3.gz')
SETIMES_PATH = os.path.join(BASE_DIR, 'data', 'SETimes.SR', 'set.sr.conll')
DATA = os.path.join(HERE, '..', 'data')
REF = os.path.join(HERE, '..', 'results', 'restoration_no_pos_ablation.json')
OUT = os.path.join(HERE, '..', 'results', 'restoration_pos_ablation_v2p.json')


class NullTagger:
    """Vraca neinformativan tag za svaku rec (nijedna POS grana se ne okida)."""

    def tag(self, words):
        return [(w, 'X') for w in words]


def per_word_correct(gold_sentences, system, ev):
    """Niz tacnosti po reci, po ISTOM kriterijumu koji koristi
    DiacriticsEvaluator.evaluate (njegov tokenizator, njegov filter
    interpunkcije i njegovo poravnanje po kracem nizu zbog dj->d)."""
    flags = []
    for gold in gold_sentences:
        restored = system.restore_text(strip_diacritics(gold), method='pos_aware')
        gw = ev._tokenize_words(gold)
        rw = ev._tokenize_words(restored)
        for i in range(min(len(gw), len(rw))):
            if not re.match(r'\w+$', gw[i], re.UNICODE):
                continue
            flags.append(gw[i] == rw[i])
    return flags


def mcnemar(a, b):
    """Uparen egzaktan/normalni test nad dva niza tacnosti."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {'only_a': 0, 'only_b': 0, 'p': 1.0}
    z = abs(only_a - only_b) / math.sqrt(n)
    p = math.erfc(z / math.sqrt(2.0))
    return {'only_a': only_a, 'only_b': only_b, 'z': round(z, 3),
            'p': float(f'{p:.6g}')}


def main():
    print('1. Ucitavam srLex + supplement v2 + Np tabelu...', flush=True)
    cg = CandidateGenerator(
        SRLEX_PATH,
        supplement_path=os.path.join(DATA, 'srwac_supplement_v2.json'),
        augment_np_path=os.path.join(DATA, 'srwac_augment_np.json'))

    print('2. Ucitavam v2p UD-only tager...', flush=True)
    with open(TAGGER_PATH, 'rb') as f:
        tagger = pickle.load(f)

    gold = load_setimes_sentences(SETIMES_PATH)
    print(f'3. SETimes.SR: {len(gold):,} recenica\n', flush=True)

    ev = DiacriticsEvaluator()
    full_sys = POSDisambiguator(cg, tagger=tagger)
    full = ev.evaluate(gold, lambda t: full_sys.restore_text(t, method='pos_aware'),
                       method_name='full (v2p tagger)', domain='SETimes (news)')

    ev2 = DiacriticsEvaluator()
    null_sys = POSDisambiguator(cg, tagger=NullTagger())
    abl = ev2.evaluate(gold, lambda t: null_sys.restore_text(t, method='pos_aware'),
                       method_name='no POS evidence', domain='SETimes (news)')

    # --- IDENTITY: ablacija ne koristi tager, mora dati stari rezultat ---
    ref = json.load(open(REF, encoding='utf-8'))
    fails = []
    for name, got, want in (('correct_words', abl['correct_words'], ref['correct_words']),
                            ('word_accuracy', abl['word_accuracy'], ref['word_accuracy']),
                            ('diac_word_accuracy', abl['diac_word_accuracy'],
                             ref['diac_word_accuracy'])):
        ok = got == want
        print(f'  {"PASS" if ok else "FAIL"}  ablacija.{name}: {got} (staro {want})')
        if not ok:
            fails.append(f'{name}: {got} != {want}')
    if fails:
        raise SystemExit('IDENTITY FAIL (ablacija se ne poklapa sa starim merenjem): '
                         + '; '.join(fails))

    print('\n4. Uparen test po recima...', flush=True)
    a = per_word_correct(gold, full_sys, ev)
    b = per_word_correct(gold, null_sys, ev2)
    if len(a) != full['total_words'] or sum(a) != full['correct_words']:
        raise SystemExit(f'preracun po recima se ne poklapa sa evaluatorom: '
                         f'{sum(a)}/{len(a)} vs {full["correct_words"]}/'
                         f'{full["total_words"]}')
    mc = mcnemar(a, b)

    n = full['total_words']
    res = {
        'corpus': 'SETimes.SR', 'total_words': n,
        'full': {'correct_words': full['correct_words'],
                 'word_accuracy_4dp': round(100 * full['correct_words'] / n, 4),
                 'diac_word_accuracy': full['diac_word_accuracy']},
        'no_pos': {'correct_words': abl['correct_words'],
                   'word_accuracy_4dp': round(100 * abl['correct_words'] / n, 4),
                   'diac_word_accuracy': abl['diac_word_accuracy']},
        'delta_pp_4dp': round(100 * (full['correct_words'] - abl['correct_words']) / n, 4),
        'mcnemar_full_vs_nopos': mc,
    }
    print(f'\n  pun modul   {res["full"]["word_accuracy_4dp"]:.4f}% '
          f'({full["correct_words"]:,})')
    print(f'  bez POS-a   {res["no_pos"]["word_accuracy_4dp"]:.4f}% '
          f'({abl["correct_words"]:,})')
    print(f'  razlika     {res["delta_pp_4dp"]:+.4f} pp; '
          f'samo pun tacan {mc["only_a"]}, samo bez-POS tacan {mc["only_b"]}, '
          f'p = {mc["p"]}')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\nSacuvano: {OUT}')


if __name__ == '__main__':
    main()
