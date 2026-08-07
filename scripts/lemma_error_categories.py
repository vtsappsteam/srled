#!/usr/bin/env python3
"""
Table 10: categorization of lemmatization errors of the integrated system
on the UD Serbian-SET test split (gold MSD).

Categories (not mutually exclusive, matching the paper's definitions):
  - gold_lemma_missing: word form present in the combined dictionary but
    the gold lemma is not among its lemma candidates
  - oov: word form absent from the combined dictionary
  - wrong_selection: both the word form and the gold lemma are present,
    but a different candidate was selected
Also reports per-POS error counts.

"""
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'src'))
import evaluate_three_testsets as ev3

OUT = str(BASE / 'results' / 'lemma_error_categories.json')


def main():
    lem = ev3.Lemmatizer(ev3.SRLEX, ev3.EXTRA)
    sents = ev3.load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')

    n_err = 0
    cats = Counter()
    pos_counts = Counter()
    examples = []
    for s in sents:
        for w, gl, msd in s:
            pred = lem.lemmatize(w, msd)
            if pred and gl and pred.lower() == gl.lower():
                continue
            n_err += 1
            pos_counts[msd[0] if msd else '?'] += 1
            wl = w.lower()
            in_dict = wl in lem._wi
            lemmas = {l for l, f in lem._wi.get(wl, [])}
            gold_in = gl.lower() in lemmas
            # preklapajuće kategorije (kao u radu): OOV reč nema kandidate,
            # pa je gold lemma automatski "missing" za nju
            if not in_dict:
                cats['oov'] += 1
            if not gold_in:
                cats['gold_lemma_missing'] += 1
            if in_dict and gold_in:
                cats['wrong_selection'] += 1
            if len(examples) < 25:
                examples.append({'word': w, 'gold': gl, 'pred': pred, 'msd': msd,
                                 'in_dict': in_dict, 'gold_in_dict': gold_in})

    res = {'total_errors': n_err,
           'categories': dict(cats),
           'shares_pct': {k: round(100 * v / n_err, 1) for k, v in cats.items()},
           'per_pos': dict(pos_counts.most_common()),
           'examples': examples}
    print(json.dumps({k: res[k] for k in ('total_errors', 'categories', 'shares_pct', 'per_pos')},
                     ensure_ascii=False, indent=1))
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
