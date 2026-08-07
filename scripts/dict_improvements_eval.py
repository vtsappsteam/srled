#!/usr/bin/env python3
"""
Evaluation of the two dictionary improvements, each measured separately:

  A  baseline  srLex + expanded_supplement.json (v1) + 17 manual corrections
  B  +#1       srLex + expanded_supplement_v2.json (adds SrpKor train pairs)
  C  +#1+#2    as B + data-driven lemma corrections (lemma_corrections_v2.csv)

Per config and per test set (UD-SET news / ReLDI Twitter / SrpKor lit+admin):
  - lemmatization with GOLD tags (SrpKor gold = UPOS coarse char, as published)
  - END-TO-END with the srwac_pnpa tagger (models_v7_pnpa); predictions are
    normalized Pn*/Pa* -> P* BEFORE lemmatization (srLex has no Pn/Pa MSDs),
    exactly as in pnpa_2x2_experiment.py. Tagging is done ONCE (predictions
    do not depend on the lemmatizer config).
Statistics: exact McNemar + 95% CI + TOST (+-0.5 pp) for B vs A, C vs A,
C vs B, and each config vs CLASSLA standard (stored per-token vectors from
scispace_pertoken.npz; nonstandard variant on ReLDI). CLASSLA is NOT re-run.

Sanity (must PASS):
  - baseline GOLD vectors identical to scispace_pertoken.npz *_our_gold
  - baseline end-to-end accuracies equal pnpa_2x2_experiment.json
    lemmatization['srwac_pnpa'] per test set
  - the 17 manual corrections in lemma_corrections_v2.csv equal the
    hardcoded LEMMA_CORRECTIONS table

No test split is used in any resource: supplement v2 and corrections were
mined from train splits only (build_supplement_v2.py, mine_lemma_corrections.py).

Usage (long-running, detached):
    nohup env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python3 scripts/dict_improvements_eval.py \
        > results/dict_improvements.log 2>&1 &
Output: results/dict_improvements.json
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402
from scispace_experiments import mcnemar_exact, paired_stats  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
EXTRA_V1 = str(SCRIPT_DIR.parent / 'data' / 'expanded_supplement.json')
EXTRA_V2 = str(SCRIPT_DIR.parent / 'data' / 'expanded_supplement_v2.json')
CORR_V2_CSV = SCRIPT_DIR.parent / 'data' / 'lemma_corrections_v2.csv'
PNPA_MODEL = (PROJECT_ROOT / 'NLTK Treniranje' / 'models_v7_pnpa'
              / 'perceptron-tagger-srwac-pnpa.pickle')
NPZ = SCRIPT_DIR.parent / 'results' / 'scispace_pertoken.npz'
PNPA_JSON = SCRIPT_DIR.parent / 'results' / 'pnpa_2x2_experiment.json'
OUT_JSON = SCRIPT_DIR.parent / 'results' / 'dict_improvements.json'


def normalize_pnpa(tag):
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag


def safe_paired_stats(a, b):
    if np.array_equal(a, b):
        return {'diff_pp': 0.0, 'ci95_pp': [0.0, 0.0], 'tost_margin_pp': 0.5,
                'tost_p': 0.0, 'equivalent_at_margin': True,
                'identical_vectors': True}
    return paired_stats(a, b)


def main():
    manual17 = dict(base.LEMMA_CORRECTIONS_MANUAL17)
    with open(CORR_V2_CSV, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    corr_v2 = {r['source_lemma']: r['corrected_lemma'] for r in rows}
    first17 = {r['source_lemma']: r['corrected_lemma'] for r in rows[:17]}
    assert first17 == manual17, 'CSV prvih 17 != hardkodovana tabela!'
    n_new_corr = len(corr_v2) - len(manual17)
    print(f'Korekcije: 17 rucnih + {n_new_corr} novih = {len(corr_v2)}',
          flush=True)

    print('Test setovi...', flush=True)
    testsets = {}
    testsets['UD-SET (news)'] = base.load_conllu(
        '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    testsets['ReLDI (Twitter)'] = base.load_reldi_test(
        str(PROJECT_ROOT / 'data' / 'ReLDI-NormTagNER-sr'
            / 'reldi-normtagner-sr.conllup'))
    testsets['SrpKor (lit+admin)'] = base.load_srpkor_test()
    for n, s in testsets.items():
        print(f'  {n}: {len(s)} sent, {sum(len(x) for x in s):,} tok', flush=True)

    print('Tagovanje (srwac_pnpa, jednom; Pn*/Pa* -> P*)...', flush=True)
    with open(PNPA_MODEL, 'rb') as f:
        tagger = pickle.load(f)
    pred_msd = {}
    for name, sents in testsets.items():
        tags = tagger.tag_sents([[w for w, l, m in s] for s in sents])
        pred_msd[name] = [[normalize_pnpa(t[1] or '') for t in sent]
                          for sent in tags]
        print(f'  {name}: tagovano', flush=True)
    del tagger

    stored = np.load(NPZ)
    with open(PNPA_JSON, encoding='utf-8') as f:
        pnpa_prev = json.load(f)

    configs = [
        ('baseline', EXTRA_V1, manual17,
         'srLex + expanded_supplement v1 + 17 manual corrections (published)'),
        ('v2_dict', EXTRA_V2, manual17,
         '#1: + SrpKor train pairs in expanded_supplement_v2'),
        ('v2_dict_corr', EXTRA_V2, corr_v2,
         '#1+#2: + data-driven lemma corrections (lemma_corrections_v2)'),
    ]

    results = {'_design': {c[0]: c[3] for c in configs}}
    results['_design']['end_to_end'] = ('srwac_pnpa tagger, predictions '
                                        'normalized Pn*/Pa*->P* before lookup')
    vec = {n: {} for n in testsets}   # testset -> config -> {'gold','pred'}
    sanity_all = True

    for cname, extra_path, corr, desc in configs:
        print(f'\n=== Config {cname}: {desc}', flush=True)
        base.LEMMA_CORRECTIONS.clear()
        base.LEMMA_CORRECTIONS.update(corr)
        lem = base.Lemmatizer(base.SRLEX, extra_path)
        for name, sents in testsets.items():
            g, p = [], []
            for si, sent in enumerate(sents):
                for ti, (w, gl, msd) in enumerate(sent):
                    pl = lem.lemmatize(w, msd)
                    g.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
                    pl2 = lem.lemmatize(w, pred_msd[name][si][ti])
                    p.append(1 if pl2 and gl and pl2.lower() == gl.lower() else 0)
            vec[name][cname] = {'gold': np.array(g, dtype=np.int8),
                                'pred': np.array(p, dtype=np.int8)}
            print(f'  {name}: gold {100 * np.mean(g):.2f}% | '
                  f'end-to-end {100 * np.mean(p):.2f}%', flush=True)
        del lem
    base.LEMMA_CORRECTIONS.clear()
    base.LEMMA_CORRECTIONS.update(manual17)

    print('\nSanity provere...', flush=True)
    for name, sents in testsets.items():
        tag = name.replace(' ', '_').replace('(', '').replace(')', '')
        ok_gold = bool(np.array_equal(vec[name]['baseline']['gold'],
                                      stored[f'{tag}_our_gold']))
        prev_pred = pnpa_prev[name]['lemmatization']['srwac_pnpa']
        now_pred = round(100 * float(vec[name]['baseline']['pred'].mean()), 2)
        ok_pred = (now_pred == prev_pred)
        print(f'  {name}: baseline gold == npz: {ok_gold} | '
              f'baseline e2e {now_pred} == pnpa_2x2 {prev_pred}: {ok_pred}',
              flush=True)
        sanity_all = sanity_all and ok_gold and ok_pred

    for name, sents in testsets.items():
        tag = name.replace(' ', '_').replace('(', '').replace(')', '')
        cl_std = stored[f'{tag}_classla_std']
        entry = {'tokens': sum(len(s) for s in sents),
                 'accuracy': {}, 'stats': {}}
        for cname, _, _, _ in configs:
            entry['accuracy'][cname] = {
                'gold': round(100 * float(vec[name][cname]['gold'].mean()), 2),
                'end_to_end': round(100 * float(vec[name][cname]['pred'].mean()), 2),
            }
        entry['accuracy']['classla_standard'] = round(100 * float(cl_std.mean()), 2)
        ns_key = f'{tag}_classla_ns'
        cl_ns = stored[ns_key] if ns_key in stored.files else None
        if cl_ns is not None:
            entry['accuracy']['classla_nonstandard'] = round(
                100 * float(cl_ns.mean()), 2)

        pairs = []
        for mode in ('gold', 'pred'):
            mlab = 'gold' if mode == 'gold' else 'e2e'
            pairs += [
                (f'{mlab}_v2_vs_baseline',
                 vec[name]['v2_dict'][mode], vec[name]['baseline'][mode]),
                (f'{mlab}_v2corr_vs_baseline',
                 vec[name]['v2_dict_corr'][mode], vec[name]['baseline'][mode]),
                (f'{mlab}_v2corr_vs_v2',
                 vec[name]['v2_dict_corr'][mode], vec[name]['v2_dict'][mode]),
            ]
            for cname, _, _, _ in configs:
                pairs.append((f'{mlab}_{cname}_vs_classla_std',
                              vec[name][cname][mode], cl_std))
                if cl_ns is not None:
                    pairs.append((f'{mlab}_{cname}_vs_classla_ns',
                                  vec[name][cname][mode], cl_ns))
        for pname, a, b in pairs:
            entry['stats'][pname] = {**mcnemar_exact(a, b),
                                     **safe_paired_stats(a, b)}
        results[name] = entry
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f'checkpoint: {OUT_JSON} ({name})', flush=True)

    results['_sanity_overall'] = 'PASS' if sanity_all else 'FAIL'
    results['_corrections'] = {'manual': len(manual17),
                               'data_driven_applied': n_new_corr}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\nGOTOVO. Sanity: {results["_sanity_overall"]}', flush=True)
    return 0 if sanity_all else 1


if __name__ == '__main__':
    sys.exit(main())
