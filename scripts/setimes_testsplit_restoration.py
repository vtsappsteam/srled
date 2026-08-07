#!/usr/bin/env python3
"""
Diacritics restoration accuracy on the HELD-OUT SETimes test split only
(reviewer-requested check: the published 99.57% is computed over the full
SETimes.SR corpus, which includes sentences seen during tagger fine-tuning;
the guiding v5 tagger was fine-tuned on the UD training split).

Protocol identical to reeval_diacritics_v6.py (v6 module, POS-aware),
restricted to SETimes sentences whose token sequence matches a sentence of
the UD Serbian-SET test split (= SETimes.SR 2.0 test split).

Output: results/setimes_testsplit_restoration.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
ROOT = BASE.parent.parent

from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator
from evaluator import DiacriticsEvaluator, load_setimes_sentences

V5_MODEL = str(ROOT / 'NLTK Treniranje' / 'models_v5' / 'perceptron-tagger-ud-clean.pickle')
SRLEX = str(ROOT / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
SETIMES = str(ROOT / 'data' / 'SETimes.SR' / 'set.sr.conll')
UD_TEST = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'
OUT = str(BASE / 'results' / 'setimes_testsplit_restoration.json')


def load_ud_test_keys(path):
    keys, cur = set(), []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if cur:
                    keys.add(' '.join(cur))
                cur = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if '-' in parts[0] or '.' in parts[0]:
                continue
            cur.append(parts[1])
    if cur:
        keys.add(' '.join(cur))
    return keys


def main():
    gold_all = load_setimes_sentences(SETIMES)
    ud_keys = load_ud_test_keys(UD_TEST)
    gold_test = [s for s in gold_all if s in ud_keys]
    print(f'SETimes sentences: {len(gold_all):,}; UD test keys: {len(ud_keys)}; '
          f'matched: {len(gold_test)}', flush=True)

    cg = CandidateGenerator(
        SRLEX,
        supplement_path=str(BASE / 'data' / 'srwac_supplement_v2.json'),
        augment_np_path=str(BASE / 'data' / 'srwac_augment_np.json'))
    with open(V5_MODEL, 'rb') as f:
        tagger = pickle.load(f)
    pos_system = POSDisambiguator(cg, tagger=tagger)
    evaluator = DiacriticsEvaluator()

    r_test = evaluator.evaluate(
        gold_test, lambda t: pos_system.restore_text(t, method='pos_aware'),
        method_name='POS-Aware v6', domain='SETimes test split (held out)')

    res = {'sentences_matched': len(gold_test),
           'sentences_total_corpus': len(gold_all),
           'test_split': {k: v for k, v in r_test.items()
                          if not isinstance(v, (list, dict))}}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:1500], flush=True)
    print(f'Saved: {OUT}', flush=True)


if __name__ == '__main__':
    main()
