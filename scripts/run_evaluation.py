#!/usr/bin/env python3
"""
Complete evaluation of the diacritics restoration system.

Runs all methods on all datasets and generates comparative results.

Methods:
    1. Frequency Baseline -- always picks the most frequent form
    2. POS-Aware -- uses POS tag for candidate filtering
    3. MSD-Aware -- uses full MSD tag
    4. POS + N-gram (hybrid)

Datasets:
    1. SETimes.SR -- news text (manually annotated)
    2. ReLDI -- Twitter text (manually annotated)

"""

import os
import sys
import pickle
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..', '..')

# Add src to path
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src'))

from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator
from ngram_model import NgramModel
from evaluator import (DiacriticsEvaluator, load_setimes_sentences,
                       load_reldi_sentences)

SRLEX_PATH = os.path.join(PROJECT_DIR, 'POS-Aware-Stemmer', 'data', 'srLex_v1.3.gz')
# Try models in order: v4 (fine-tuned) > v3 > base
for _mp in ['models_v4', 'models_v3', 'models']:
    TAGGER_PATH = os.path.join(PROJECT_DIR, 'NLTK Treniranje', _mp, 'perceptron-tagger.pickle')
    if os.path.exists(TAGGER_PATH):
        break
SETIMES_PATH = os.path.join(PROJECT_DIR, 'data', 'SETimes.SR', 'set.sr.conll')
RELDI_PATH = os.path.join(PROJECT_DIR, 'data', 'ReLDI-NormTagNER-sr', 'reldi-normtagner-sr.conllup')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("DIACRITICS RESTORATION SYSTEM EVALUATION")
    print("=" * 60)

    print("\n1. Loading srLex...")
    supplement_path = os.path.join(SCRIPT_DIR, '..', 'data', 'srwac_supplement.json')
    cg = CandidateGenerator(SRLEX_PATH,
                            supplement_path=supplement_path if os.path.exists(supplement_path) else None)

    tagger = None
    if os.path.exists(TAGGER_PATH):
        print("\n2. Loading POS tagger...")
        with open(TAGGER_PATH, 'rb') as f:
            tagger = pickle.load(f)
        print("   POS tagger loaded.")
    else:
        print(f"\n2. POS tagger not found: {TAGGER_PATH}")
        print("   Skipping POS-aware methods.")

    # N-gram model
    ngram = None
    ngram_path = os.path.join(SCRIPT_DIR, '..', 'data', 'ngram_model.json.gz')
    if os.path.exists(ngram_path):
        print("\n3. Loading n-gram model...")
        ngram = NgramModel()
        ngram.load(ngram_path)
    else:
        print(f"\n3. N-gram model not found: {ngram_path}")

    freq_system = POSDisambiguator(cg, tagger=None)
    pos_system = POSDisambiguator(cg, tagger=tagger, ngram_model=ngram) if tagger else None
    msd_system = POSDisambiguator(cg, tagger=tagger) if tagger else None
    hybrid_system = POSDisambiguator(cg, tagger=tagger, ngram_model=ngram) if tagger and ngram else None

    datasets = []

    if os.path.exists(SETIMES_PATH):
        print("\n3. Loading SETimes.SR...")
        setimes = load_setimes_sentences(SETIMES_PATH)
        print(f"   {len(setimes)} sentences")
        datasets.append(('SETimes (news)', setimes))
    else:
        print(f"   SETimes not found: {SETIMES_PATH}")

    if os.path.exists(RELDI_PATH):
        print("   Loading ReLDI...")
        reldi = load_reldi_sentences(RELDI_PATH)
        print(f"   {len(reldi)} sentences")
        datasets.append(('ReLDI (Twitter)', reldi))
    else:
        print(f"   ReLDI not found: {RELDI_PATH}")

    evaluator = DiacriticsEvaluator()

    for domain_name, gold_sents in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {domain_name}")
        print(f"{'='*60}")

        # Method 1: Frequency Baseline
        def freq_restore(text):
            return freq_system.restore_text(text, method='frequency')
        evaluator.evaluate(gold_sents, freq_restore,
                          method_name='1_Frequency', domain=domain_name)
        freq_system.reset_stats()

        # Method 2: POS-Aware
        if pos_system:
            def pos_restore(text):
                return pos_system.restore_text(text, method='pos_aware')
            evaluator.evaluate(gold_sents, pos_restore,
                              method_name='2_POS-Aware', domain=domain_name)
            pos_system.reset_stats()

        # Method 3: MSD-Aware
        if msd_system:
            def msd_restore(text):
                return msd_system.restore_text(text, method='msd_aware')
            evaluator.evaluate(gold_sents, msd_restore,
                              method_name='3_MSD-Aware', domain=domain_name)
            msd_system.reset_stats()

        # Method 4: POS + N-gram (hybrid)
        if hybrid_system:
            def hybrid_restore(text):
                return hybrid_system.restore_text(text, method='pos_ngram')
            evaluator.evaluate(gold_sents, hybrid_restore,
                              method_name='4_POS+Ngram', domain=domain_name)
            hybrid_system.reset_stats()

    evaluator.compare_methods()

    evaluator.save_results(os.path.join(RESULTS_DIR, 'evaluation_results.json'))

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
