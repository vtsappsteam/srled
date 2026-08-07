#!/usr/bin/env python3
"""
Re-evaluate diacritics restoration with the IMPROVED module (v6):
  - srWaC supplement v2 (24,114 OOV forms, full 100M-token chunk)
  - srWaC Np augmentation table (26,609 proper-noun forms)
  - supplement/augmentation consulted in the POS-aware path (bug fix)
  - frequency dominance guard over POS filtering (FREQ_GUARD_RATIO)
  - dj -> đ OOV rule with morpheme-boundary exceptions
  - hyphenated-compound OOV splitting

Same evaluator and datasets as reeval_diacritics_v5.py (paper Table 5).

"""
import os
import sys
import json
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator
from evaluator import DiacriticsEvaluator, load_setimes_sentences, load_reldi_sentences

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
V5_MODEL = os.path.join(BASE_DIR, 'NLTK Treniranje', 'models_v5',
                        'perceptron-tagger-ud-clean.pickle')
SRLEX_PATH = os.path.join(BASE_DIR, 'POS-Aware-Stemmer', 'data', 'srLex_v1.3.gz')
SETIMES_PATH = os.path.join(BASE_DIR, 'data', 'SETimes.SR', 'set.sr.conll')
RELDI_PATH = os.path.join(BASE_DIR, 'data', 'ReLDI-NormTagNER-sr', 'reldi-normtagner-sr.conllup')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'evaluation_v6_improved.json')


def main():
    print('1. Loading srLex + supplement v2 + Np augmentation...', flush=True)
    cg = CandidateGenerator(
        SRLEX_PATH,
        supplement_path=os.path.join(DATA, 'srwac_supplement_v2.json'),
        augment_np_path=os.path.join(DATA, 'srwac_augment_np.json'))

    print('2. Loading v5 clean tagger...', flush=True)
    with open(V5_MODEL, 'rb') as f:
        tagger = pickle.load(f)

    freq_system = POSDisambiguator(cg, tagger=None)
    pos_system = POSDisambiguator(cg, tagger=tagger)
    evaluator = DiacriticsEvaluator()

    for name, path, loader in [
            ('SETimes (news)', SETIMES_PATH, load_setimes_sentences),
            ('ReLDI (Twitter)', RELDI_PATH, load_reldi_sentences)]:
        gold = loader(path)
        print(f'\n=== {name}: {len(gold)} sentences', flush=True)
        evaluator.evaluate(gold, lambda t: freq_system.restore_text(t, method='frequency'),
                           method_name='Frequency', domain=name)
        freq_system.reset_stats()
        evaluator.evaluate(gold, lambda t: pos_system.restore_text(t, method='pos_aware'),
                           method_name='POS-Aware v6', domain=name)
        print('   stats:', dict(pos_system._stats), flush=True)
        pos_system.reset_stats()

    evaluator.compare_methods()
    evaluator.save_results(OUT)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
