#!/usr/bin/env python3
"""
Sensitivity analysis of FREQ_GUARD_RATIO (K) for the frequency dominance
guard: word-level accuracy on SETimes and ReLDI for K in {5, 10, 20, 50, inf}.
K=inf disables the guard. Loads resources once; tags once per corpus per K
(the tagger is deterministic, so only the guard decision changes).

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
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'guard_sensitivity.json')


def main():
    cg = CandidateGenerator(
        SRLEX_PATH,
        supplement_path=os.path.join(DATA, 'srwac_supplement_v2.json'),
        augment_np_path=os.path.join(DATA, 'srwac_augment_np.json'))
    with open(V5_MODEL, 'rb') as f:
        tagger = pickle.load(f)
    system = POSDisambiguator(cg, tagger=tagger)

    datasets = [('SETimes', load_setimes_sentences(SETIMES_PATH)),
                ('ReLDI', load_reldi_sentences(RELDI_PATH))]

    results = {}
    for K in [5, 10, 20, 50, float('inf')]:
        POSDisambiguator.FREQ_GUARD_RATIO = K
        evaluator = DiacriticsEvaluator()
        row = {}
        for name, gold in datasets:
            r = evaluator.evaluate(
                gold, lambda t: system.restore_text(t, method='pos_aware'),
                method_name=f'K={K}', domain=name)
            row[name] = r['word_accuracy']
            system.reset_stats()
        results[str(K)] = row
        print(f'K={K}: {row}', flush=True)

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
