#!/usr/bin/env python3
"""
Stripped-ReLDI TEST split with PREDICTED MSD tags (reviewer-requested row):
the pipeline tags the stripped tokens with its own v7_pnpa perceptron
(single tagging pass, faithful to the Stage 1-4 architecture), then

  E) lemmatizer alone on stripped tokens, predicted MSD
  F) Stage 3 restoration guided by predicted MSD, then lemmatizer,
     predicted MSD
  D) CLASSLA-Stanza end-to-end on the same stripped tokens (rows from
     /tmp/classla_stripped_out.txt, as in stripped_test_split_eval.py)

Gold-MSD counterparts (B, C) are in results/stripped_reldi_test_split.json.
Output: results/stripped_reldi_predicted_msd.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'src'))

import evaluate_three_testsets as ev3
from experiment_stripped_reldi import strip_word
from stripped_test_split_eval import load_reldi_with_split, mcnemar_exact, RELDI, CL_OUT
from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator


def main():
    corpus = load_reldi_with_split(RELDI)
    cl_lines = [line.rstrip('\n').split('\t') for line in open(CL_OUT)]
    assert len(cl_lines) == len(corpus)
    test_idx = [i for i, (split, s) in enumerate(corpus) if split == 'test']
    n_tok = sum(len(corpus[i][1]) for i in test_idx)
    print(f'ReLDI test split: {len(test_idx)} sentences, {n_tok:,} tokens', flush=True)

    print('Loading v7_pnpa tagger...', flush=True)
    with open(ev3.V7_MODEL, 'rb') as f:
        tagger = pickle.load(f)

    print('Tagging stripped sentences...', flush=True)
    stripped_sents = [[strip_word(w) for (w, gl, msd) in corpus[i][1]]
                      for i in test_idx]
    pred = tagger.tag_sents(stripped_sents)
    pred_tags = [[ev3.normalize_pnpa(t) for (_, t) in sent] for sent in pred]

    # tagger accuracy on stripped test tokens (context number)
    gold_tags = [[msd for (_, _, msd) in corpus[i][1]] for i in test_idx]
    flat_p = [t for s in pred_tags for t in s]
    flat_g = [t for s in gold_tags for t in s]
    msd_acc = 100 * sum(1 for p, g in zip(flat_p, flat_g) if p == g) / len(flat_g)
    pos_acc = 100 * sum(1 for p, g in zip(flat_p, flat_g)
                        if p and g and p[0] == g[0]) / len(flat_g)
    print(f'Tagger on stripped test: MSD {msd_acc:.2f}%  POS {pos_acc:.2f}%', flush=True)

    print('Loading restoration resources...', flush=True)
    cg = CandidateGenerator(
        ev3.SRLEX,
        supplement_path=str(BASE / 'data' / 'srwac_supplement_v2.json'),
        augment_np_path=str(BASE / 'data' / 'srwac_augment_np.json'))
    disamb = POSDisambiguator(cg, tagger=None)

    print('Loading locked lemmatizer...', flush=True)
    lem = ev3.Lemmatizer(ev3.SRLEX, ev3.EXTRA)

    ve, vf, vd = [], [], []
    for si, i in enumerate(test_idx):
        sent = corpus[i][1]
        cl_sent = cl_lines[i]
        assert len(cl_sent) == len(sent)
        for ti, ((w, gl, msd), cl_lemma) in enumerate(zip(sent, cl_sent)):
            ws = strip_word(w)
            pm = pred_tags[si][ti]
            p = lem.lemmatize(ws, pm)
            ve.append(1 if p and gl and p.lower() == gl.lower() else 0)
            restored = disamb._restore_word_pos(ws, pm)
            p = lem.lemmatize(restored, pm)
            vf.append(1 if p and gl and p.lower() == gl.lower() else 0)
            vd.append(1 if gl and cl_lemma.lower() == gl.lower() else 0)

    ve, vf, vd = np.array(ve), np.array(vf), np.array(vd)
    res = {'tokens': n_tok, 'sentences': len(test_idx),
           'tagger_stripped_msd_acc': round(msd_acc, 2),
           'tagger_stripped_pos_acc': round(pos_acc, 2),
           'E_lemmatizer_alone_pred': round(100 * float(ve.mean()), 2),
           'F_stage3_pred': round(100 * float(vf.mean()), 2),
           'D_classla': round(100 * float(vd.mean()), 2)}
    for pname, (x, y) in {'E_vs_D': (ve, vd), 'F_vs_D': (vf, vd),
                          'F_vs_E': (vf, ve)}.items():
        p, f_only, s_only = mcnemar_exact(x, y)
        d = 100 * float(x.mean() - y.mean())
        res[f'mcnemar_{pname}'] = {'p': p, 'first_only': f_only,
                                   'second_only': s_only,
                                   'diff_pp': round(d, 2)}
    out = str(BASE / 'results' / 'stripped_reldi_predicted_msd.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
