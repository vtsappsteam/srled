#!/usr/bin/env python3
"""Robusnost tagera na tekst bez dijakritika, mereno na HELD-OUT podacima.

Rad je taj broj do sada davao na CELOM SETimes.SR korpusu, a isporucena
(combined) varijanta tagera je fine-tunovana na SETimes 2.0 train+dev delovima,
pa taj korpus sadrzi materijal iz treninga: apsolutne tacnosti (98,02 / 99,43)
nisu procena generalizacije, nego opisna dijagnostika perturbacije.

Ovde se ista perturbacija racuna na UD Serbian-SET TEST splitu (520 recenica,
11.421 token), koji je izvan svih faza treninga te varijante (fine-tune je isao
na UD train, SET2 train+dev i ReLDI train).

Postupak je identican onome u final_system_eval (isto stripovanje `_D2A`, ista
Pn/Pa normalizacija, isti per-POS prag od 50 tokena), pa su brojevi uporedivi.

IDENTITY: tacnost na neizmenjenom ulazu mora da bude jednaka vrednosti koju rad
vec navodi za taj tager na UD testu (95,02 MSD / 98,72 POS). Bez PASS-a se nista
ne pise.

Izlaz: results/tagger_degradation_heldout_v2p.json
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402

TAGGER = base.BASE / 'NLTK Treniranje' / 'models_v2p_pnpa' / 'perceptron-tagger-v2p-combined.pickle'
UD_TEST = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'
OUT = SCRIPT_DIR.parent / 'results' / 'tagger_degradation_heldout_v2p.json'
REF = SCRIPT_DIR.parent / 'results' / 'final_system_eval_v2p.json'
_D2A = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')


def main():
    ref = json.loads(REF.read_text(encoding='utf-8'))['UD-SET (news)']['tagger']
    sents = base.load_conllu(UD_TEST)
    n_tok = sum(len(s) for s in sents)
    print(f'UD test (held out): {len(sents)} recenica, {n_tok:,} tokena', flush=True)

    with open(TAGGER, 'rb') as f:
        tagger = pickle.load(f)
    words = [[w for w, l, m in s] for s in sents]
    stripped = [[w.translate(_D2A) for w in sent] for sent in words]
    po_all = tagger.tag_sents(words)
    ps_all = tagger.tag_sents(stripped)

    ok = dict(msd_o=0, msd_s=0, pos_o=0, pos_s=0)
    pp = defaultdict(lambda: [0, 0, 0])
    diac = defaultdict(lambda: [0, 0])
    for si, sent in enumerate(sents):
        for ti, (w, gl, gm) in enumerate(sent):
            po = base.normalize_pnpa(po_all[si][ti][1] or '')
            ps = base.normalize_pnpa(ps_all[si][ti][1] or '')
            ok['msd_o'] += po == gm
            ok['msd_s'] += ps == gm
            ok['pos_o'] += po[:1] == gm[:1]
            ok['pos_s'] += ps[:1] == gm[:1]
            g0 = gm[0] if gm else '?'
            pp[g0][0] += 1
            pp[g0][1] += po == gm
            pp[g0][2] += ps == gm
            diac[g0][0] += 1
            diac[g0][1] += (w != w.translate(_D2A))

    r = lambda x: round(100 * x / n_tok, 2)  # noqa: E731
    out = {'corpus': 'UD Serbian-SET test split (held out)', 'tokens': n_tok,
           'msd_original': r(ok['msd_o']), 'msd_stripped': r(ok['msd_s']),
           'msd_drop_pp': round(r(ok['msd_s']) - r(ok['msd_o']), 2),
           'pos_original': r(ok['pos_o']), 'pos_stripped': r(ok['pos_s']),
           'pos_drop_pp': round(r(ok['pos_s']) - r(ok['pos_o']), 2)}
    per_pos = {}
    for pos, (n, a, b) in sorted(pp.items(), key=lambda x: -x[1][0]):
        if n < 50:
            continue
        per_pos[pos] = {'n': n, 'msd_original': round(100 * a / n, 2),
                        'msd_stripped': round(100 * b / n, 2),
                        'drop_pp': round(100 * (b - a) / n, 2),
                        'pct_tokens_with_diacritics': round(100 * diac[pos][1] / n, 1)}
    out['per_pos_msd'] = per_pos

    print(f"  MSD {out['msd_original']} -> {out['msd_stripped']} "
          f"({out['msd_drop_pp']} pp) | POS {out['pos_original']} -> "
          f"{out['pos_stripped']} ({out['pos_drop_pp']} pp)")
    for pos, v in per_pos.items():
        print(f"    {pos}: n={v['n']:>5}  {v['msd_original']:>6} -> "
              f"{v['msd_stripped']:>6}  ({v['drop_pp']:+.2f} pp, "
              f"{v['pct_tokens_with_diacritics']}% sa dijakritikom)")

    fails = []
    if out['msd_original'] != ref['msd_accuracy_normalized']:
        fails.append(f"MSD {out['msd_original']} != {ref['msd_accuracy_normalized']}")
    if out['pos_original'] != ref['pos_accuracy']:
        fails.append(f"POS {out['pos_original']} != {ref['pos_accuracy']}")
    if fails:
        raise SystemExit('IDENTITY FAIL: ' + '; '.join(fails) + '; nista nije upisano')
    print('\nIDENTITY: tacnost na neizmenjenom ulazu se poklapa sa radom PASS')

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Sacuvano: {OUT}')


if __name__ == '__main__':
    main()
