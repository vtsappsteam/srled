#!/usr/bin/env python3
"""
Final reviewed lemma corrections: replace the mechanically applied 101-pair
set with the linguist-reviewed decision over all 207 candidates
(results/lemma_correction_candidates_reviewed.csv, column ODLUKA:
131 approved / 76 rejected).

Steps:
  1. Rebuild data/lemma_corrections_v2.csv = 17 manual + 131 approved pairs.
  2. Evaluate on the three test sets (lemmatization with GOLD tags and
     END-TO-END with the srwac_pnpa tagger, predictions normalized
     Pn*/Pa* -> P* before lookup), three lemmatizer configs:
       baseline     srLex + expanded_supplement v1 + 17 manual corrections
       old_applied  srLex + supplement v2 + 17+101 (pre-review, sanity only)
       reviewed     srLex + supplement v2 + 17+131 (FINAL)
  3. Statistics (exact McNemar + 95% CI + TOST +-0.5 pp): reviewed vs
     baseline, reviewed vs old_applied, reviewed vs CLASSLA standard
     (stored per-token vectors from scispace_pertoken.npz; nonstandard
     variant on ReLDI). CLASSLA is NOT re-run.
  4. Merge the 'reviewed' accuracies and stats into
     results/dict_improvements.json (existing content preserved).

Sanity (must PASS):
  - baseline GOLD vectors identical to scispace_pertoken.npz *_our_gold
  - baseline end-to-end equal to pnpa_2x2_experiment.json srwac_pnpa
  - old_applied gold/e2e equal to stored dict_improvements.json v2_dict_corr
  - first 17 rows of the rebuilt CSV equal the hardcoded table

Usage (long-running, detached):
    cd repo && nohup env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python3 scripts/reviewed_corrections_eval.py \
        > results/reviewed_corrections.log 2>&1 &
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
import scispace_experiments  # noqa: E402,F401  (fixes base.BASE/SRLEX paths)
from dict_improvements_eval import normalize_pnpa, safe_paired_stats  # noqa: E402
from scispace_experiments import mcnemar_exact  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
EXTRA_V1 = str(SCRIPT_DIR.parent / 'data' / 'expanded_supplement.json')
EXTRA_V2 = str(SCRIPT_DIR.parent / 'data' / 'expanded_supplement_v2.json')
ORIG_CSV = SCRIPT_DIR.parent / 'data' / 'lemma_corrections.csv'
CORR_V2_CSV = SCRIPT_DIR.parent / 'data' / 'lemma_corrections_v2.csv'
REVIEWED_CSV = (SCRIPT_DIR.parent / 'results'
                / 'lemma_correction_candidates_reviewed.csv')
PNPA_MODEL = (PROJECT_ROOT / 'NLTK Treniranje' / 'models_v7_pnpa'
              / 'perceptron-tagger-srwac-pnpa.pickle')
NPZ = SCRIPT_DIR.parent / 'results' / 'scispace_pertoken.npz'
PNPA_JSON = SCRIPT_DIR.parent / 'results' / 'pnpa_2x2_experiment.json'
OUT_JSON = SCRIPT_DIR.parent / 'results' / 'dict_improvements.json'


def rebuild_corrections_csv():
    """data/lemma_corrections_v2.csv = 17 manual + 131 reviewed 'da' pairs.

    Returns (manual17, old_corr, reviewed_corr) as source->target dicts.
    """
    manual17 = dict(base.LEMMA_CORRECTIONS_MANUAL17)
    with open(CORR_V2_CSV, encoding='utf-8') as f:
        old_corr = {r['source_lemma']: r['corrected_lemma']
                    for r in csv.DictReader(f)}
    with open(REVIEWED_CSV, encoding='utf-8') as f:
        reviewed_rows = list(csv.DictReader(f))
    approved = [r for r in reviewed_rows
                if r['ODLUKA'].strip().lower() == 'da']
    rejected = [r for r in reviewed_rows
                if r['ODLUKA'].strip().lower() == 'ne']
    assert len(approved) + len(rejected) == len(reviewed_rows), \
        'ODLUKA kolona ima vrednosti van {da, ne}!'
    assert not any(r['in_manual_17'] == 'True' for r in approved), \
        'Odobren kandidat vec postoji medju 17 rucnih!'
    assert len({r['source_lemma'] for r in approved}) == len(approved), \
        'Dupli source_lemma medju odobrenima!'
    print(f'Pregledano: {len(reviewed_rows)} kandidata -> '
          f'{len(approved)} da / {len(rejected)} ne '
          f'(ranije primenjeno mehanicki: {len(old_corr) - len(manual17)})',
          flush=True)

    with open(ORIG_CSV, encoding='utf-8') as f:
        orig_lines = f.read().rstrip('\n')
    with open(CORR_V2_CSV, 'w', encoding='utf-8', newline='') as f:
        f.write(orig_lines + '\n')
        w = csv.writer(f)
        for r in approved:  # candidate order = descending train frequency
            w.writerow([r['source_lemma'], r['target_lemma'],
                        f"data-driven from train splits (n={r['total']}, "
                        f"agreement={float(r['agreement']):.0%}), "
                        f"manually reviewed"])
    reviewed_corr = dict(manual17)
    reviewed_corr.update({r['source_lemma']: r['target_lemma']
                          for r in approved})
    # round-trip check: the rebuilt CSV parses back to exactly this mapping
    with open(CORR_V2_CSV, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    parsed = {r['source_lemma']: r['corrected_lemma'] for r in rows}
    first17 = {r['source_lemma']: r['corrected_lemma'] for r in rows[:17]}
    assert first17 == manual17, 'CSV prvih 17 != hardkodovana tabela!'
    assert parsed == reviewed_corr, 'CSV round-trip != ocekivana mapa!'
    print(f'lemma_corrections_v2.csv REGENERISAN: 17 rucnih + {len(approved)} '
          f'pregledanih = {len(reviewed_corr)} korekcija', flush=True)
    return manual17, old_corr, reviewed_corr


def main():
    manual17, old_corr, reviewed_corr = rebuild_corrections_csv()

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
    with open(OUT_JSON, encoding='utf-8') as f:
        results = json.load(f)

    configs = [
        ('baseline', EXTRA_V1, manual17,
         'srLex + expanded_supplement v1 + 17 manual corrections (published)'),
        ('old_applied', EXTRA_V2, old_corr,
         'supplement v2 + 17+101 mechanically applied (pre-review, sanity)'),
        ('reviewed', EXTRA_V2, reviewed_corr,
         'FINAL: supplement v2 + 17 manual + 131 reviewed corrections'),
    ]

    vec = {n: {} for n in testsets}
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
    base.LEMMA_CORRECTIONS.update(reviewed_corr)

    print('\nSanity provere...', flush=True)
    sanity_all = True
    for name in testsets:
        tag = name.replace(' ', '_').replace('(', '').replace(')', '')
        ok_gold = bool(np.array_equal(vec[name]['baseline']['gold'],
                                      stored[f'{tag}_our_gold']))
        prev_pred = pnpa_prev[name]['lemmatization']['srwac_pnpa']
        now_pred = round(100 * float(vec[name]['baseline']['pred'].mean()), 2)
        ok_pred = (now_pred == prev_pred)
        prev_old = results[name]['accuracy']['v2_dict_corr']
        old_g = round(100 * float(vec[name]['old_applied']['gold'].mean()), 2)
        old_p = round(100 * float(vec[name]['old_applied']['pred'].mean()), 2)
        ok_old = (old_g == prev_old['gold'] and old_p == prev_old['end_to_end'])
        print(f'  {name}: baseline gold == npz: {ok_gold} | '
              f'baseline e2e {now_pred} == pnpa_2x2 {prev_pred}: {ok_pred} | '
              f'old_applied {old_g}/{old_p} == dict_improvements '
              f"{prev_old['gold']}/{prev_old['end_to_end']}: {ok_old}",
              flush=True)
        sanity_all = sanity_all and ok_gold and ok_pred and ok_old

    print('\nStatistika i upis...', flush=True)
    for name in testsets:
        tag = name.replace(' ', '_').replace('(', '').replace(')', '')
        cl_std = stored[f'{tag}_classla_std']
        ns_key = f'{tag}_classla_ns'
        cl_ns = stored[ns_key] if ns_key in stored.files else None
        entry = results[name]
        entry['accuracy']['reviewed'] = {
            'gold': round(100 * float(vec[name]['reviewed']['gold'].mean()), 2),
            'end_to_end': round(
                100 * float(vec[name]['reviewed']['pred'].mean()), 2),
        }
        for mode, mlab in (('gold', 'gold'), ('pred', 'e2e')):
            r = vec[name]['reviewed'][mode]
            pairs = [
                (f'{mlab}_reviewed_vs_baseline', r, vec[name]['baseline'][mode]),
                (f'{mlab}_reviewed_vs_old101', r, vec[name]['old_applied'][mode]),
                (f'{mlab}_reviewed_vs_classla_std', r, cl_std),
            ]
            if cl_ns is not None:
                pairs.append((f'{mlab}_reviewed_vs_classla_ns', r, cl_ns))
            for pname, a, b in pairs:
                entry['stats'][pname] = {**mcnemar_exact(a, b),
                                         **safe_paired_stats(a, b)}
                s = entry['stats'][pname]
                print(f"  {name} {pname}: diff {s['diff_pp']:+.3f} pp, "
                      f"fix/break {s['only_ours']}/{s['only_classla']}, "
                      f"p={s['p_exact']}, TOST ekv.: "
                      f"{s['equivalent_at_margin']}", flush=True)

    results['_design']['reviewed'] = (
        'FINAL: #1 + linguist-reviewed corrections (131 of 207 candidates '
        'approved, column ODLUKA in lemma_correction_candidates_reviewed.csv); '
        'replaces the mechanical 101-pair set in lemma_corrections_v2.csv')
    results['_corrections'] = {'manual': 17,
                               'data_driven_applied_pre_review': 101,
                               'reviewed_applied': len(reviewed_corr) - 17}
    results['_reviewed_sanity'] = 'PASS' if sanity_all else 'FAIL'
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\nGOTOVO. Sanity (baseline==npz/pnpa_2x2, old_applied==stored): '
          f'{results["_reviewed_sanity"]}', flush=True)
    return 0 if sanity_all else 1


if __name__ == '__main__':
    sys.exit(main())
