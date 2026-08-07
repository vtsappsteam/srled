#!/usr/bin/env python3
"""
2x2 tagger matrix experiment: {srWaC pre-training vs gold-only} x {Pn/Pa
pronoun subclassification vs plain P}. Extends gold_only_experiment.py.

Taggers:
  v7              srWaC + fine-tune, plain P   (models_v7, published system)
  gold_only       no pre-training, plain P     (models_gold_only)
  srwac_pnpa      srWaC + fine-tune, Pn/Pa     (models_v7_pnpa, train_pnpa_taggers.py)
  gold_only_pnpa  no pre-training, Pn/Pa       (models_gold_only_pnpa)

Per test set (UD-SET news, ReLDI Twitter, SrpKor lit+admin):
  1. MSD accuracy vs the gold of the tagger's OWN tagset (plain taggers vs
     plain-P gold; Pn/Pa taggers vs Pn/Pa-converted gold, conversion =
     classify_pronoun_msd(gold MSD, gold lemma), deterministic) + MSD accuracy
     on the NORMALIZED common tagset (both pred and gold Pn*/Pa* -> P*) for
     cross-tagger comparison + coarse POS. SrpKor gold is UPOS-coarse only.
  2. Downstream lemmatization with the SAME lemmatizer (srLex + expanded
     supplement). For the Pn/Pa taggers, predictions are NORMALIZED
     (Pn*/Pa* -> P*) BEFORE lemmatization: srLex v1.3 contains no Pn/Pa MSDs,
     so without normalization the exact-MSD lookup would artificially miss on
     every pronoun (raw-prediction numbers are also reported to quantify
     exactly that effect). v7 and gold_only feed raw predictions, matching
     the published pipeline. Exact McNemar + 95% CI + TOST (+-0.5 pp) for
     each Pn/Pa tagger vs v7, vs its plain-P counterpart, and vs
     CLASSLA-Stanza (stored per-token vectors from scispace_pertoken.npz;
     CLASSLA is NOT re-run).
  3. Sanity: recomputed v7 vector must equal the stored npz vector, and
     recomputed gold_only numbers must equal results/gold_only_experiment.json.

Usage:
    nohup env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python3 scripts/pnpa_2x2_experiment.py \
        > results/pnpa_2x2_experiment.log 2>&1 &
Output: results/pnpa_2x2_experiment.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402
# Importing scispace_experiments fixes base.BASE/SRLEX/V7_MODEL paths.
from scispace_experiments import mcnemar_exact, paired_stats  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # .../NLP-POS-Tagging
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from classify_pronouns_in_eval_corpus import classify_pronoun_msd  # noqa: E402

NLTK_DIR = PROJECT_ROOT / 'NLTK Treniranje'
MODELS = {
    'gold_only': NLTK_DIR / 'models_gold_only' / 'perceptron-tagger-gold-only.pickle',
    'srwac_pnpa': NLTK_DIR / 'models_v7_pnpa' / 'perceptron-tagger-srwac-pnpa.pickle',
    'gold_only_pnpa': NLTK_DIR / 'models_gold_only_pnpa'
                      / 'perceptron-tagger-gold-only-pnpa.pickle',
}
PNPA_TAGGERS = {'srwac_pnpa', 'gold_only_pnpa'}
NPZ = SCRIPT_DIR.parent / 'results' / 'scispace_pertoken.npz'
GOLD_ONLY_JSON = SCRIPT_DIR.parent / 'results' / 'gold_only_experiment.json'
OUT_JSON = SCRIPT_DIR.parent / 'results' / 'pnpa_2x2_experiment.json'


def normalize_pnpa(tag):
    """Pn*/Pa* -> generic P* (lossless; see train_gold_only.py)."""
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag


def tag_and_score(tagger, sents, lem, full_msd_gold, pnpa_tagger):
    """Tag sentences once; return dict with accuracies and lemma vectors.

    sents: [[(word, gold_lemma, gold_msd_plain), ...], ...]. The tagger's OWN
    gold is derived per token: pn_pa conversion for Pn/Pa taggers, plain
    otherwise. Lemma vectors: 'lemma_vec' uses normalized predictions for
    Pn/Pa taggers and raw for plain taggers (headline, deployment-matched);
    'lemma_vec_raw' always uses raw predictions.
    """
    pred_tags = tagger.tag_sents([[w for w, l, m in s] for s in sents])
    lemma_ok, lemma_raw_ok = [], []
    pos_ok = msd_own_ok = msd_norm_ok = n_pnpa_pred = total = 0
    for si, sent in enumerate(sents):
        for ti, (w, gl, gm) in enumerate(sent):
            pm = pred_tags[si][ti][1] or ''
            if pm[:2] in ('Pn', 'Pa'):
                n_pnpa_pred += 1
            pm_feed = normalize_pnpa(pm) if pnpa_tagger else pm
            pl = lem.lemmatize(w, pm_feed)
            lemma_ok.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
            pl_raw = pl if pm_feed == pm else lem.lemmatize(w, pm)
            lemma_raw_ok.append(
                1 if pl_raw and gl and pl_raw.lower() == gl.lower() else 0)
            total += 1
            if gm and pm and gm[0].upper() == pm[0].upper():
                pos_ok += 1
            if full_msd_gold:
                own_gold = classify_pronoun_msd(gm, gl) if pnpa_tagger else gm
                if own_gold == pm:
                    msd_own_ok += 1
                if normalize_pnpa(gm) == normalize_pnpa(pm):
                    msd_norm_ok += 1
    acc = {'pos_accuracy': round(100 * pos_ok / total, 2),
           'pnpa_predictions': n_pnpa_pred}
    if full_msd_gold:
        acc['msd_accuracy_own_gold'] = round(100 * msd_own_ok / total, 2)
        acc['msd_accuracy_pnpa_normalized'] = round(100 * msd_norm_ok / total, 2)
    return (np.array(lemma_ok, dtype=np.int8),
            np.array(lemma_raw_ok, dtype=np.int8), acc)


def main():
    for name, p in MODELS.items():
        if not p.exists():
            print(f'MISSING model {name}: {p}')
            return 1

    print('Učitavanje lematizatora (srLex + expanded supplement)...', flush=True)
    lem = base.Lemmatizer(base.SRLEX, base.EXTRA)
    stored = np.load(NPZ)
    with open(GOLD_ONLY_JSON, encoding='utf-8') as f:
        prev = json.load(f)

    testsets = {}
    testsets['UD-SET (news)'] = (
        base.load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'), True)
    testsets['ReLDI (Twitter)'] = (base.load_reldi_test(
        str(base.BASE / 'data' / 'ReLDI-NormTagNER-sr'
            / 'reldi-normtagner-sr.conllup')), True)
    testsets['SrpKor (lit+admin)'] = (base.load_srpkor_test(), False)

    results = {'_design': {
        'taggers': {
            'v7': 'srWaC pretrain + fine-tune, plain P (published system)',
            'gold_only': 'no pretrain, plain P',
            'srwac_pnpa': 'srWaC pretrain (v3) + same fine-tune procedure, Pn/Pa gold',
            'gold_only_pnpa': 'no pretrain, Pn/Pa gold',
        },
        'lemma_note': ('Pn/Pa taggers: predictions normalized Pn*/Pa*->P* '
                       'BEFORE lemmatization (srLex has no Pn/Pa MSDs; raw '
                       'feed reported as lemma_pred_raw). v7/gold_only: raw '
                       'feed (published pipeline).'),
    }}
    for m in ('models_gold_only_pnpa', 'models_v7_pnpa'):
        ti = NLTK_DIR / m / 'training_info.json'
        if ti.exists():
            with open(ti, encoding='utf-8') as f:
                results[f'_training_info_{m}'] = json.load(f)

    # Tag sequentially per tagger (two ~1 GB models must never be in RAM
    # at the same time on the 16 GB machine).
    all_paths = {'v7': Path(base.V7_MODEL), **MODELS}
    vec = {n: {} for n in testsets}      # testset -> tagger -> lemma vector
    vec_raw = {n: {} for n in testsets}
    acc = {n: {} for n in testsets}
    for tname, mpath in all_paths.items():
        print(f'\n--- Tager {tname}: učitavanje {mpath.name}...', flush=True)
        with open(mpath, 'rb') as f:
            tgg = pickle.load(f)
        for name, (sents, full_msd) in testsets.items():
            v, vr, a = tag_and_score(tgg, sents, lem, full_msd,
                                     tname in PNPA_TAGGERS)
            vec[name][tname], vec_raw[name][tname], acc[name][tname] = v, vr, a
            print(f'  {name}: {a} | lema {100 * v.mean():.2f}%'
                  f' (raw feed {100 * vr.mean():.2f}%)', flush=True)
        del tgg

    sanity_all = True
    for name, (sents, full_msd) in testsets.items():
        tag = name.replace(' ', '_').replace('(', '').replace(')', '')
        n_tok = sum(len(s) for s in sents)
        vec_t, vec_raw_t, acc_t = vec[name], vec_raw[name], acc[name]

        # --- Sanity 1: v7 vector vs stored npz ---
        v7_stored = stored[f'{tag}_our_pred']
        v7_ident = bool(np.array_equal(vec_t['v7'], v7_stored))
        # --- Sanity 2: gold_only lemma % vs gold_only_experiment.json ---
        prev_go = prev[name]['lemmatization']['gold_only_pred']
        go_now = round(100 * float(vec_t['gold_only'].mean()), 2)
        go_ident = (go_now == prev_go)
        if not v7_ident:
            print('  UPOZORENJE: v7 vektor != npz!', flush=True)
        if not go_ident:
            print(f'  UPOZORENJE: gold_only lema {go_now} != {prev_go} '
                  f'iz gold_only_experiment.json!', flush=True)
        sanity_all = sanity_all and v7_ident and go_ident

        cl_vec = stored[f'{tag}_classla_std']
        entry = {
            'tokens': n_tok,
            'tagging': acc_t,
            'lemmatization': {tn: round(100 * float(v.mean()), 2)
                              for tn, v in vec_t.items()},
            'lemmatization_raw_feed': {
                tn: round(100 * float(vec_raw_t[tn].mean()), 2)
                for tn in PNPA_TAGGERS},
            'classla_standard': round(100 * float(cl_vec.mean()), 2),
            'sanity': {'v7_vector_identical_to_npz': v7_ident,
                       'gold_only_matches_previous_json': go_ident},
            'stats': {},
        }
        pairs = [
            ('srwac_pnpa_vs_v7', vec_t['srwac_pnpa'], vec_t['v7']),
            ('gold_only_pnpa_vs_v7', vec_t['gold_only_pnpa'], vec_t['v7']),
            ('srwac_pnpa_vs_classla', vec_t['srwac_pnpa'], cl_vec),
            ('gold_only_pnpa_vs_classla', vec_t['gold_only_pnpa'], cl_vec),
            ('gold_only_pnpa_vs_gold_only', vec_t['gold_only_pnpa'],
             vec_t['gold_only']),
        ]
        ns_key = f'{tag}_classla_ns'
        if ns_key in stored.files:
            ns_vec = stored[ns_key]
            entry['classla_nonstandard'] = round(100 * float(ns_vec.mean()), 2)
            pairs += [('srwac_pnpa_vs_classla_ns', vec_t['srwac_pnpa'], ns_vec),
                      ('gold_only_pnpa_vs_classla_ns',
                       vec_t['gold_only_pnpa'], ns_vec)]
        for pname, a, b in pairs:
            entry['stats'][pname] = {**mcnemar_exact(a, b), **paired_stats(a, b)}

        results[name] = entry
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f'  checkpoint: {OUT_JSON}', flush=True)

    results['_sanity_overall'] = 'PASS' if sanity_all else 'FAIL'
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\nGOTOVO. Sanity: {results["_sanity_overall"]}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
