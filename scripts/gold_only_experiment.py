#!/usr/bin/env python3
"""
Gold-only tagger experiment (mentor's question): does srWaC pre-training
matter for downstream lemmatization, or is training the averaged perceptron
purely on manually annotated corpora (UD train + SETimes.SR 2.0 train/dev +
ReLDI train) enough?

Compares, on the three held-out test sets (UD-SET news, ReLDI Twitter,
SrpKor lit+admin):
  1. MSD tagging accuracy (raw + Pn/Pa-normalized) and coarse POS accuracy:
     gold-only tagger vs v7 (srWaC pre-trained + fine-tuned) tagger.
  2. End-to-end lemmatization with predicted tags (our_pred), using the SAME
     lemmatizer (srLex + expanded supplement) so only the tagger differs.
  3. Exact McNemar + 95% CI + TOST (+-0.5 pp): gold-only vs CLASSLA-Stanza
     (existing per-token vectors from scispace_pertoken.npz; CLASSLA is NOT
     re-run) and gold-only vs v7 our_pred.

Sanity check: the v7 our_pred vector is recomputed and compared with the
stored npz vector; a mismatch is reported loudly.

Prerequisite: NLTK Treniranje/models_gold_only/perceptron-tagger-gold-only.pickle
(produced by NLTK Treniranje/train_gold_only.py).

Usage (SrpKor comes from the local HF cache):
    nohup env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python3 scripts/gold_only_experiment.py \
        > results/gold_only_experiment.log 2>&1 &
Output: results/gold_only_experiment.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402
# Importing scispace_experiments also fixes base.BASE/SRLEX/V7_MODEL paths
# (repo/scripts is one level deeper than the original scripts folder).
from scispace_experiments import mcnemar_exact, paired_stats  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # .../NLP-POS-Tagging
GOLD_ONLY_MODEL = (PROJECT_ROOT / 'NLTK Treniranje' / 'models_gold_only'
                   / 'perceptron-tagger-gold-only.pickle')
TRAIN_INFO = (PROJECT_ROOT / 'NLTK Treniranje' / 'models_gold_only'
              / 'training_info.json')
NPZ = SCRIPT_DIR.parent / 'results' / 'scispace_pertoken.npz'
OUT_JSON = SCRIPT_DIR.parent / 'results' / 'gold_only_experiment.json'


def normalize_pnpa(tag):
    """Pn*/Pa* -> generic P* (see train_gold_only.py for the rationale)."""
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag


def tag_and_score(tagger, sents, lem, full_msd_gold):
    """Tag sentences, return (lemma correctness vector, tagging accuracies).

    full_msd_gold=False (SrpKor): gold tag is a single coarse letter, so only
    coarse POS accuracy is meaningful.
    """
    pred_tags = tagger.tag_sents([[w for w, l, m in s] for s in sents])
    lemma_ok = []
    pos_ok = msd_ok = msd_norm_ok = total = 0
    for si, sent in enumerate(sents):
        for ti, (w, gl, gm) in enumerate(sent):
            pm = pred_tags[si][ti][1] or ''
            pl = lem.lemmatize(w, pm)
            lemma_ok.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
            total += 1
            if gm and pm and gm[0].upper() == pm[0].upper():
                pos_ok += 1
            if full_msd_gold:
                if gm == pm:
                    msd_ok += 1
                if normalize_pnpa(gm) == normalize_pnpa(pm):
                    msd_norm_ok += 1
    acc = {'pos_accuracy': round(100 * pos_ok / total, 2)}
    if full_msd_gold:
        acc['msd_accuracy'] = round(100 * msd_ok / total, 2)
        acc['msd_accuracy_pnpa_normalized'] = round(100 * msd_norm_ok / total, 2)
    return np.array(lemma_ok, dtype=np.int8), acc


def main():
    if not GOLD_ONLY_MODEL.exists():
        print(f'MISSING model: {GOLD_ONLY_MODEL} (run train_gold_only.py first)')
        return 1

    print('Učitavanje lematizatora (srLex + expanded supplement)...', flush=True)
    lem = base.Lemmatizer(base.SRLEX, base.EXTRA)
    with open(GOLD_ONLY_MODEL, 'rb') as f:
        gold_tagger = pickle.load(f)
    with open(base.V7_MODEL, 'rb') as f:
        v7_tagger = pickle.load(f)
    stored = np.load(NPZ)

    testsets = {}
    testsets['UD-SET (news)'] = (
        base.load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'), True)
    testsets['ReLDI (Twitter)'] = (base.load_reldi_test(
        str(base.BASE / 'data' / 'ReLDI-NormTagNER-sr'
            / 'reldi-normtagner-sr.conllup')), True)
    testsets['SrpKor (lit+admin)'] = (base.load_srpkor_test(), False)

    results = {}
    if TRAIN_INFO.exists():
        with open(TRAIN_INFO, encoding='utf-8') as f:
            results['_training_info'] = json.load(f)

    for name, (sents, full_msd) in testsets.items():
        tag = name.replace(' ', '_').replace('(', '').replace(')', '')
        n_tok = sum(len(s) for s in sents)
        print(f'\n=== {name}: {len(sents)} rečenica, {n_tok:,} tokena ===',
              flush=True)

        go_vec, go_acc = tag_and_score(gold_tagger, sents, lem, full_msd)
        v7_vec, v7_acc = tag_and_score(v7_tagger, sents, lem, full_msd)

        # Sanity: recomputed v7 our_pred vs stored npz vector
        v7_stored = stored[f'{tag}_our_pred']
        identical = bool(np.array_equal(v7_vec, v7_stored))
        if not identical:
            print(f'  UPOZORENJE: v7 our_pred se ne poklapa sa npz '
                  f'({100*v7_vec.mean():.2f}% vs {100*v7_stored.mean():.2f}%)',
                  flush=True)
        cl_vec = stored[f'{tag}_classla_std']

        entry = {
            'tokens': n_tok,
            'tagging': {'gold_only': go_acc, 'v7_combined': v7_acc},
            'lemmatization': {
                'gold_only_pred': round(100 * float(go_vec.mean()), 2),
                'v7_pred': round(100 * float(v7_vec.mean()), 2),
                'v7_pred_stored_npz': round(100 * float(v7_stored.mean()), 2),
                'classla_standard': round(100 * float(cl_vec.mean()), 2),
            },
            'v7_vector_identical_to_npz': identical,
            'gold_only_vs_classla': {**mcnemar_exact(go_vec, cl_vec),
                                     **paired_stats(go_vec, cl_vec)},
            'gold_only_vs_v7': {**mcnemar_exact(go_vec, v7_vec),
                                **paired_stats(go_vec, v7_vec)},
        }
        # ReLDI: i poređenje sa CLASSLA nonstandard (fer za tvitove)
        ns_key = f'{tag}_classla_ns'
        if ns_key in stored.files:
            ns_vec = stored[ns_key]
            entry['lemmatization']['classla_nonstandard'] = round(
                100 * float(ns_vec.mean()), 2)
            entry['gold_only_vs_classla_ns'] = {
                **mcnemar_exact(go_vec, ns_vec), **paired_stats(go_vec, ns_vec)}

        print(f'  lema gold-only: {entry["lemmatization"]["gold_only_pred"]}%'
              f'  | v7: {entry["lemmatization"]["v7_pred"]}%'
              f'  | CLASSLA: {entry["lemmatization"]["classla_standard"]}%',
              flush=True)
        print(f'  tagovanje gold-only: {go_acc} | v7: {v7_acc}', flush=True)

        results[name] = entry
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f'  checkpoint: {OUT_JSON}', flush=True)

    print('\nGOTOVO.', flush=True)
    print(json.dumps({k: v for k, v in results.items()
                      if not k.startswith('_')}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
