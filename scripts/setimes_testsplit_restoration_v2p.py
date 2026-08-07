#!/usr/bin/env python3
"""Leakage provera (held-out SETimes test split) posle retreninga tagera.

Uskladisteni rezultat (results/setimes_testsplit_restoration.json) je dobijen
uparivanjem recenica preko ALFANUMERICKI NORMALIZOVANIH kljuceva (belezi to
polje "matching" u JSON-u): kljuc je mala slova + samo alfanumericki znaci.
Egzaktno poklapanje tokena daje 420 recenica, normalizovano 437 - kao u
uskladistenom rezultatu. Ovaj wrapper zato radi sopstveno uparivanje, pa
harness (kandidat generator + POS sistem + evaluator) poziva direktno.

Dva prolaza:
  1. KONTROLA: stari v5 tager - mora reprodukovati uskladisteni rezultat
     (8,164/8,188 = 99.71%). Bez tog PASS-a novi broj se ne pise.
  2. NOVO: v2p_udonly tager (isti koji vodi restauraciju u radu).

Izlaz: results/setimes_testsplit_restoration_v2p.json (stari fajl netaknut).
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import setimes_testsplit_restoration as base  # noqa: E402
from candidate_generator import CandidateGenerator  # noqa: E402
from pos_disambiguator import POSDisambiguator     # noqa: E402
from evaluator import DiacriticsEvaluator, load_setimes_sentences  # noqa: E402

RES = Path(__file__).resolve().parent.parent / 'results'
OLD = RES / 'setimes_testsplit_restoration.json'
NEW = RES / 'setimes_testsplit_restoration_v2p.json'
V2P_MODEL = str(Path(base.ROOT) / 'NLTK Treniranje' / 'models_v2p_udonly'
                / 'perceptron-tagger-v2p-udonly.pickle')


def alnum_key(text):
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def load_ud_test_keys_normalized(path):
    keys, cur = set(), []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if cur:
                    keys.add(alnum_key(''.join(cur)))
                cur = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if '-' in parts[0] or '.' in parts[0]:
                continue
            cur.append(parts[1])
    if cur:
        keys.add(alnum_key(''.join(cur)))
    return keys


def run(model_path, out_path, label):
    gold_all = load_setimes_sentences(base.SETIMES)
    ud_keys = load_ud_test_keys_normalized(base.UD_TEST)
    gold_test = [s for s in gold_all if alnum_key(s) in ud_keys]
    print(f'[{label}] SETimes: {len(gold_all):,}; UD kljucevi: {len(ud_keys)}; '
          f'upareno: {len(gold_test)}', flush=True)

    cg = CandidateGenerator(
        base.SRLEX,
        supplement_path=str(base.BASE / 'data' / 'srwac_supplement_v2.json'),
        augment_np_path=str(base.BASE / 'data' / 'srwac_augment_np.json'))
    with open(model_path, 'rb') as f:
        tagger = pickle.load(f)
    pos_system = POSDisambiguator(cg, tagger=tagger)
    r = DiacriticsEvaluator().evaluate(
        gold_test, lambda t: pos_system.restore_text(t, method='pos_aware'),
        method_name='POS-Aware v6', domain='SETimes held-out test split')

    res = {'matching': 'alphanumeric-normalized sentence keys vs UD Serbian-SET test',
           'tagger': Path(model_path).name,
           'sentences_matched': len(gold_test),
           'sentences_total_corpus': len(gold_all),
           'test_split': {k: v for k, v in r.items()
                          if not isinstance(v, (list, dict))}}
    if out_path is not None:
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=1, default=str)
        print(f'Sacuvano: {out_path}', flush=True)
    return res['test_split']


if __name__ == '__main__':
    old = json.loads(OLD.read_text())['test_split']

    ctrl = run(base.V5_MODEL, None, 'KONTROLA v5')
    ok = (ctrl['correct_words'] == old['correct_words']
          and ctrl['total_words'] == old['total_words'])
    print(f"KONTROLA: {ctrl['correct_words']}/{ctrl['total_words']} vs "
          f"uskladisteno {old['correct_words']}/{old['total_words']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        sys.exit('KONTROLA FAIL - novi broj se ne pise.')

    new = run(V2P_MODEL, NEW, 'NOVO v2p_udonly')
    print(f"\nSTARO (v5):  {old['word_accuracy']}%  "
          f"({old['correct_words']}/{old['total_words']})")
    print(f"NOVO (v2p): {new['word_accuracy']}%  "
          f"({new['correct_words']}/{new['total_words']})")
