#!/usr/bin/env python3
"""
Consolidated re-verification of the FINAL locked system configuration
(srwac_pnpa tagger + srLex + expanded_supplement_v2 + 148 lemma corrections
+ Pn*/Pa* -> P* normalization inside the lemmatizer).

Produces every accuracy-related number of the paper in one place:
  T2   gold-tag lemmatization on the three test sets + stats vs CLASSLA std
  T3   per-POS accuracy on UD test vs CLASSLA std + Holm-Bonferroni
  T4   end-to-end (predicted MSD) on the three test sets + CLASSLA std/ns
       + exact McNemar + 95% CI + TOST (+-0.5 pp)
  T7   MSD granularity (full / coarse / none) on UD test
  T9   step-7 value (= T2 UD gold with the final configuration)
  S42  tagger accuracies: MSD (Pn/Pa-normalized) and coarse POS on
       UD / ReLDI / SrpKor test + degradation on SETimes with stripped
       diacritics (overall and per POS)

CLASSLA is NOT re-run: stored per-token correctness vectors from
results/scispace_pertoken.npz are used (CLASSLA and the test sets are
unchanged since that run).

Sanity (must PASS): gold and end-to-end accuracies of the locked system
must equal the 'reviewed' configuration in results/dict_improvements.json.

Usage (long-running, detached):
    nohup env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python3 scripts/final_system_eval.py \
        > results/final_system_eval.log 2>&1 &
Output: results/final_system_eval.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402  (locked configuration)

PROJECT_ROOT = base.BASE
NPZ = SCRIPT_DIR.parent / 'results' / 'scispace_pertoken.npz'
DICT_IMPR = SCRIPT_DIR.parent / 'results' / 'dict_improvements.json'
OUT_JSON = SCRIPT_DIR.parent / 'results' / 'final_system_eval.json'
SETIMES = PROJECT_ROOT / 'data' / 'SETimes.SR' / 'set.sr.conll'
TOST_MARGIN = 0.005  # +-0.5 pp

_D2A = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')

POS_ORDER = ['N', 'V', 'A', 'P', 'R', 'S', 'C', 'M', 'Q']


def mcnemar_exact(a, b):
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0:
        return {'p_exact': 1.0, 'only_ours': n01, 'only_other': n10}
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return {'p_exact': round(float(p), 6), 'only_ours': n01, 'only_other': n10}


def paired_stats(a, b):
    if np.array_equal(a, b):
        return {'diff_pp': 0.0, 'ci95_pp': [0.0, 0.0], 'tost_p': 0.0,
                'equivalent_at_margin': True, 'identical_vectors': True}
    d = a.astype(float) - b.astype(float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    p_low = 1 - stats.t.cdf((mean + TOST_MARGIN) / se, n - 1)
    p_high = stats.t.cdf((mean - TOST_MARGIN) / se, n - 1)
    p_tost = max(p_low, p_high)
    return {'diff_pp': round(mean * 100, 3),
            'ci95_pp': [round(ci[0] * 100, 3), round(ci[1] * 100, 3)],
            'tost_margin_pp': TOST_MARGIN * 100,
            'tost_p': round(float(p_tost), 5),
            'equivalent_at_margin': bool(p_tost < 0.05)}


def acc(v):
    return round(100 * float(np.mean(v)), 2)


def load_setimes(path):
    """set.sr.conll: id, word, lemma, POS, MSD, ... -> (word, lemma, msd)."""
    sentences, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 5 and '-' not in parts[0] and '.' not in parts[0]:
                current.append((parts[1], parts[2], parts[4]))
    if current:
        sentences.append(current)
    return sentences


def main():
    print('Konfiguracija (zaključani sistem):', flush=True)
    print(f'  EXTRA   = {base.EXTRA}', flush=True)
    print(f'  MODEL   = {base.V7_MODEL}', flush=True)
    print(f'  korekcije = {len(base.LEMMA_CORRECTIONS)}', flush=True)

    print('\nUčitavanje lematizatora i tagera...', flush=True)
    lem = base.Lemmatizer(base.SRLEX, base.EXTRA)
    with open(base.V7_MODEL, 'rb') as f:
        tagger = pickle.load(f)
    n_classes = len(getattr(tagger, 'classes', []) or [])

    testsets = {}
    testsets['UD-SET (news)'] = base.load_conllu(
        '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    testsets['ReLDI (Twitter)'] = base.load_reldi_test(
        str(PROJECT_ROOT / 'data' / 'ReLDI-NormTagNER-sr'
            / 'reldi-normtagner-sr.conllup'))
    testsets['SrpKor (lit+admin)'] = base.load_srpkor_test()

    stored = np.load(NPZ)
    with open(DICT_IMPR, encoding='utf-8') as f:
        reviewed_ref = json.load(f)

    results = {'_config': {
        'tagger': str(base.V7_MODEL), 'tagger_classes': n_classes,
        'supplement': str(base.EXTRA),
        'lemma_corrections': len(base.LEMMA_CORRECTIONS),
        'normalization': 'Pn*/Pa* -> P* inside Lemmatizer.lemmatize',
        'classla_vectors': 'stored scispace_pertoken.npz (not re-run)'}}
    sanity_all = True

    # ---------- T2 + T4 + tagger accuracies ----------
    perpos_vectors = None
    for name, sents in testsets.items():
        tag = name.replace(' ', '_').replace('(', '').replace(')', '')
        n_tok = sum(len(s) for s in sents)
        print(f'\n=== {name}: {len(sents)} rečenica, {n_tok:,} tokena ===',
              flush=True)

        pred_tags = tagger.tag_sents([[w for w, l, m in s] for s in sents])

        our_gold, our_pred, tag_ok_msd, tag_ok_pos, pos_labels = [], [], [], [], []
        for si, sent in enumerate(sents):
            for ti, (w, gl, gm) in enumerate(sent):
                pm_raw = pred_tags[si][ti][1] or ''
                pm = base.normalize_pnpa(pm_raw)
                pl = lem.lemmatize(w, gm)
                our_gold.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
                pl2 = lem.lemmatize(w, pm)
                our_pred.append(1 if pl2 and gl and pl2.lower() == gl.lower() else 0)
                if name.startswith('SrpKor'):
                    tag_ok_msd.append(1 if pm[:1] == gm else 0)
                    tag_ok_pos.append(1 if pm[:1] == gm else 0)
                else:
                    tag_ok_msd.append(1 if pm == gm else 0)
                    tag_ok_pos.append(1 if pm[:1] == gm[:1] else 0)
                pos_labels.append(gm[0] if gm else '?')
        our_gold = np.array(our_gold, dtype=np.int8)
        our_pred = np.array(our_pred, dtype=np.int8)

        entry = {'tokens': n_tok,
                 'gold': acc(our_gold), 'end_to_end': acc(our_pred),
                 'tagger': {'msd_accuracy_normalized': acc(np.array(tag_ok_msd)),
                            'pos_accuracy': acc(np.array(tag_ok_pos))}}
        if name.startswith('SrpKor'):
            entry['tagger'] = {'pos_accuracy': acc(np.array(tag_ok_pos)),
                               'note': 'SrpKor gold je UPOS->grubi tag'}
        print(f"  gold {entry['gold']} | e2e {entry['end_to_end']} | "
              f"tagger {entry['tagger']}", flush=True)

        # sanity vs dict_improvements.json 'reviewed'
        ref = reviewed_ref[name]['accuracy']['reviewed']
        ok = (entry['gold'] == ref['gold']
              and entry['end_to_end'] == ref['end_to_end'])
        entry['sanity_vs_dict_improvements_reviewed'] = bool(ok)
        sanity_all = sanity_all and ok
        print(f"  SANITY vs reviewed ({ref['gold']}/{ref['end_to_end']}): "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

        cl_std = stored[f'{tag}_classla_std']
        entry['classla_standard'] = acc(cl_std)
        entry['stats'] = {
            'gold_vs_classla_std': {**mcnemar_exact(our_gold, cl_std),
                                    **paired_stats(our_gold, cl_std)},
            'e2e_vs_classla_std': {**mcnemar_exact(our_pred, cl_std),
                                   **paired_stats(our_pred, cl_std)},
        }
        ns_key = f'{tag}_classla_ns'
        if ns_key in stored.files:
            cl_ns = stored[ns_key]
            entry['classla_nonstandard'] = acc(cl_ns)
            entry['stats']['gold_vs_classla_ns'] = {
                **mcnemar_exact(our_gold, cl_ns), **paired_stats(our_gold, cl_ns)}
            entry['stats']['e2e_vs_classla_ns'] = {
                **mcnemar_exact(our_pred, cl_ns), **paired_stats(our_pred, cl_ns)}

        if name.startswith('UD'):
            perpos_vectors = (our_gold, cl_std, np.array(pos_labels))

        results[name] = entry
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f'  checkpoint: {OUT_JSON}', flush=True)

    # ---------- T3: per-POS UD + Holm-Bonferroni ----------
    print('\n=== T3: per-POS (UD test, gold tagovi, vs CLASSLA std) ===',
          flush=True)
    ours_v, cl_v, pos_v = perpos_vectors
    perpos = {}
    pvals = []
    for pos in POS_ORDER:
        mask = pos_v == pos
        n = int(mask.sum())
        if n == 0:
            continue
        o, c = ours_v[mask], cl_v[mask]
        mc = mcnemar_exact(o, c)
        perpos[pos] = {'n': n, 'ours': acc(o), 'classla': acc(c),
                       'diff_pp': round(acc(o) - acc(c), 2), **mc}
        pvals.append((pos, mc['p_exact']))
        print(f"  {pos}: n={n} ours {acc(o)} classla {acc(c)} "
              f"p={mc['p_exact']}", flush=True)
    # Holm-Bonferroni over the m simultaneous tests (step-down, monotone
    # adjusted p-values: adj_i = max_{j<=i} min(1, (m-j) * p_(j)))
    m = len(pvals)
    running_max = 0.0
    sig = True
    for rank, (pos, p) in enumerate(sorted(pvals, key=lambda x: x[1])):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        perpos[pos]['p_holm_adjusted'] = round(running_max, 6)
        if sig and p <= 0.05 / (m - rank):
            perpos[pos]['holm_significant'] = True
        else:
            sig = False
            perpos[pos]['holm_significant'] = False
    results['per_pos_UD'] = perpos

    # ---------- T7: granularity ----------
    print('\n=== T7: granularnost (UD test, zaključani sistem) ===',
          flush=True)
    ud = testsets['UD-SET (news)']
    gran = {}
    gran_vec = {}
    for mode, mk in [('full', lambda m: m),
                     ('coarse', lambda m: m[0] if m else ''),
                     ('none', lambda m: '')]:
        v = []
        for s in ud:
            for w, gl, msd in s:
                p = lem.lemmatize(w, mk(msd))
                v.append(1 if p and gl and p.lower() == gl.lower() else 0)
        v = np.array(v, dtype=np.int8)
        gran_vec[mode] = v
        gran[mode] = acc(v)
        print(f'  {mode}: {gran[mode]}', flush=True)
    gran['coarse_vs_full'] = mcnemar_exact(gran_vec['coarse'], gran_vec['full'])
    gran['none_vs_full'] = mcnemar_exact(gran_vec['none'], gran_vec['full'])
    results['granularity_UD'] = gran

    # ---------- S42: tagger degradation on stripped SETimes ----------
    print('\n=== S4.2: degradacija tagera na SETimes bez dijakritika ===',
          flush=True)
    setimes = load_setimes(SETIMES)
    n_tok = sum(len(s) for s in setimes)
    print(f'  SETimes: {len(setimes)} rečenica, {n_tok:,} tokena', flush=True)
    words = [[w for w, l, m in s] for s in setimes]
    words_stripped = [[w.translate(_D2A) for w in sent] for sent in words]
    preds_orig = tagger.tag_sents(words)
    preds_strip = tagger.tag_sents(words_stripped)
    deg = {'tokens': n_tok}
    per_pos = {}
    ok_msd_o = ok_msd_s = ok_pos_o = ok_pos_s = 0
    from collections import defaultdict
    pp = defaultdict(lambda: [0, 0, 0])  # n, msd_ok_orig, msd_ok_strip
    diac_tokens = defaultdict(lambda: [0, 0])  # per POS: n, with_diac
    for si, sent in enumerate(setimes):
        for ti, (w, gl, gm) in enumerate(sent):
            po = base.normalize_pnpa(preds_orig[si][ti][1] or '')
            ps = base.normalize_pnpa(preds_strip[si][ti][1] or '')
            ok_msd_o += po == gm
            ok_msd_s += ps == gm
            ok_pos_o += po[:1] == gm[:1]
            ok_pos_s += ps[:1] == gm[:1]
            g0 = gm[0] if gm else '?'
            pp[g0][0] += 1
            pp[g0][1] += po == gm
            pp[g0][2] += ps == gm
            diac_tokens[g0][0] += 1
            diac_tokens[g0][1] += (w != w.translate(_D2A))
    deg['msd_original'] = round(100 * ok_msd_o / n_tok, 2)
    deg['msd_stripped'] = round(100 * ok_msd_s / n_tok, 2)
    deg['msd_drop_pp'] = round(deg['msd_stripped'] - deg['msd_original'], 2)
    deg['pos_original'] = round(100 * ok_pos_o / n_tok, 2)
    deg['pos_stripped'] = round(100 * ok_pos_s / n_tok, 2)
    deg['pos_drop_pp'] = round(deg['pos_stripped'] - deg['pos_original'], 2)
    for pos, (n, a, b) in sorted(pp.items(), key=lambda x: -x[1][0]):
        if n < 50:
            continue
        per_pos[pos] = {'n': n, 'msd_original': round(100 * a / n, 2),
                        'msd_stripped': round(100 * b / n, 2),
                        'drop_pp': round(100 * (b - a) / n, 2),
                        'pct_tokens_with_diacritics':
                            round(100 * diac_tokens[pos][1] / n, 1)}
    deg['per_pos_msd'] = per_pos
    results['tagger_degradation_SETimes'] = deg
    print(f"  MSD {deg['msd_original']} -> {deg['msd_stripped']} "
          f"({deg['msd_drop_pp']} pp) | POS {deg['pos_original']} -> "
          f"{deg['pos_stripped']} ({deg['pos_drop_pp']} pp)", flush=True)

    results['_sanity_overall'] = 'PASS' if sanity_all else 'FAIL'
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nGOTOVO. Sanity vs dict_improvements 'reviewed': "
          f"{results['_sanity_overall']}", flush=True)
    print(f'Sačuvano: {OUT_JSON}', flush=True)
    return 0 if sanity_all else 1


if __name__ == '__main__':
    sys.exit(main())
