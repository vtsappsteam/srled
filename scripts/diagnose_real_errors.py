#!/usr/bin/env python3
"""
Diagnose ACTUAL errors of the production diacritics restoration system.

Unlike analyze_diacritics_errors.py (which replicates only the dictionary
lookup), this runs the real POSDisambiguator (dictionary + supplement +
POS filtering + c/c patterns + OOV heuristics) on SETimes and ReLDI with
stripped diacritics, and dumps every wrongly restored word with context
and a diagnostic category. Output: results/real_errors_diacritics.json

"""
import os
import sys
import json
import pickle
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator
from evaluator import load_setimes_sentences, load_reldi_sentences
from diacritics import strip_diacritics

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
V5_MODEL = os.path.join(BASE_DIR, 'NLTK Treniranje', 'models_v5',
                        'perceptron-tagger-ud-clean.pickle')
SRLEX_PATH = os.path.join(BASE_DIR, 'POS-Aware-Stemmer', 'data', 'srLex_v1.3.gz')
SETIMES_PATH = os.path.join(BASE_DIR, 'data', 'SETimes.SR', 'set.sr.conll')
RELDI_PATH = os.path.join(BASE_DIR, 'data', 'ReLDI-NormTagNER-sr', 'reldi-normtagner-sr.conllup')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'real_errors_diacritics_v6.json')


def categorize(err_word_gold, restored, cg):
    """Diagnostic category for one error."""
    ascii_form = strip_diacritics(err_word_gold).lower()
    cands = cg.get_candidates(ascii_form)
    supp = cg.get_supplement(ascii_form)
    gold_lower = err_word_gold.lower()
    if not cands and not supp:
        if '-' in err_word_gold:
            return 'OOV_hyphenated'
        if any(c.isdigit() for c in err_word_gold):
            return 'OOV_numeric'
        if err_word_gold[:1].isupper():
            return 'OOV_capitalized(proper?)'
        return 'OOV_other'
    variants = {c[0].lower() for c in cands} if cands else set()
    if supp:
        variants.add(supp[0].lower())
    if gold_lower not in variants:
        return 'gold_variant_missing_from_dict'
    if len(variants) == 1:
        return 'in_dict_but_wrong(single_variant?)'
    return 'ambiguity_wrongly_resolved'


def run(name, gold_sents, system, cg):
    """Evaluate word-level and collect errors."""
    total = correct = 0
    errors = []
    for si, sent in enumerate(gold_sents):
        gold_text = sent if isinstance(sent, str) else ' '.join(sent)
        stripped = strip_diacritics(gold_text)
        restored = system.restore_text(stripped, method='pos_aware')
        gw, rw = gold_text.split(), restored.split()
        if len(gw) != len(rw):
            continue  # tokenization mismatch, skip (evaluator does same)
        for i, (g, r) in enumerate(zip(gw, rw)):
            total += 1
            if g == r:
                correct += 1
            else:
                errors.append({
                    'gold': g, 'restored': r,
                    'category': categorize(g, r, cg),
                    'context': ' '.join(gw[max(0, i-2):i+3]),
                })
    acc = 100.0 * correct / total if total else 0
    cat_counts = Counter(e['category'] for e in errors)
    top_words = Counter((e['gold'], e['restored']) for e in errors).most_common(30)
    print(f'\n=== {name}: word_acc={acc:.2f}% ({total-correct} errors / {total} words)')
    for c, n in cat_counts.most_common():
        print(f'  {c:<38} {n:>5}  ({100.0*n/len(errors):.1f}% of errors)')
    print('  TOP parovi (gold <- restored):')
    for (g, r), n in top_words[:15]:
        print(f'    {n:>4}x  {g}  <-  {r}')
    return {'accuracy': round(acc, 2), 'total_words': total,
            'n_errors': len(errors), 'categories': dict(cat_counts),
            'top_pairs': [{'gold': g, 'restored': r, 'n': n} for (g, r), n in top_words],
            'errors': errors}


def main():
    print('1. Loading srLex + supplement v2 + Np augmentation...', flush=True)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    cg = CandidateGenerator(
        SRLEX_PATH,
        supplement_path=os.path.join(data_dir, 'srwac_supplement_v2.json'),
        augment_np_path=os.path.join(data_dir, 'srwac_augment_np.json'))
    print('2. Loading v5 clean tagger...', flush=True)
    with open(V5_MODEL, 'rb') as f:
        tagger = pickle.load(f)
    system = POSDisambiguator(cg, tagger=tagger)

    out = {}
    print('3. SETimes...', flush=True)
    out['SETimes'] = run('SETimes (news)', load_setimes_sentences(SETIMES_PATH), system, cg)
    print('4. ReLDI...', flush=True)
    out['ReLDI'] = run('ReLDI (Twitter)', load_reldi_sentences(RELDI_PATH), system, cg)

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
