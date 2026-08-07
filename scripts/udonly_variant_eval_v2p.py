#!/usr/bin/env python3
"""Split-clean UD-only varijanta v2p tagera na UD test splitu.

Rad tvrdi dve stvari o toj varijanti (Sekcije 4.2 i 5.3):
  1. njena MSD i gruba POS tacnost na UD testu,
  2. da joj se end-to-end lematizacija poklapa sa isporucenom combined
     varijantom na dve decimale.
Obe su posle retreninga tagera ostale nemerene, pa se mere ovde.

Konfiguracija je identicna final_system_eval-u (isti leksikon, dopuna,
148 korekcija, Pn*/Pa* -> P* normalizacija); menja se SAMO tager.

IDENTITY: gold-tag tacnost ne dodiruje tager i mora ispasti 98,48 kao u
final_system_eval_v2p.json. Bez tog PASS-a skripta ne pise izlaz.

Izlaz: results/udonly_variant_eval_v2p.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402

UDONLY = base.BASE / 'NLTK Treniranje' / 'models_v2p_udonly' / 'perceptron-tagger-v2p-udonly.pickle'
COMBINED_JSON = SCRIPT_DIR.parent / 'results' / 'final_system_eval_v2p.json'
OUT = SCRIPT_DIR.parent / 'results' / 'udonly_variant_eval_v2p.json'
UD_TEST = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'


def acc(v):
    return round(100 * float(np.mean(v)), 2)


def main():
    if not UDONLY.exists():
        raise SystemExit(f'nema modela: {UDONLY}')
    ref = json.loads(COMBINED_JSON.read_text(encoding='utf-8'))['UD-SET (news)']

    print(f'TAGER: {UDONLY}', flush=True)
    print('Ucitavam lematizator...', flush=True)
    lem = base.Lemmatizer(base.SRLEX, base.EXTRA)
    with open(UDONLY, 'rb') as f:
        tagger = pickle.load(f)
    n_classes = len(getattr(tagger, 'classes', []) or [])

    sents = base.load_conllu(UD_TEST)
    n_tok = sum(len(s) for s in sents)
    print(f'UD test: {len(sents)} recenica, {n_tok:,} tokena, '
          f'{n_classes} klasa u modelu\n', flush=True)

    pred_tags = tagger.tag_sents([[w for w, l, m in s] for s in sents])
    gold_ok, e2e_ok, msd_ok, pos_ok = [], [], [], []
    for si, sent in enumerate(sents):
        for ti, (w, gl, gm) in enumerate(sent):
            pm = base.normalize_pnpa(pred_tags[si][ti][1] or '')
            pl = lem.lemmatize(w, gm)
            gold_ok.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
            pl2 = lem.lemmatize(w, pm)
            e2e_ok.append(1 if pl2 and gl and pl2.lower() == gl.lower() else 0)
            msd_ok.append(1 if pm == gm else 0)
            pos_ok.append(1 if pm[:1] == gm[:1] else 0)

    out = {'tagger': str(UDONLY), 'tagger_classes': n_classes,
           'tokens': n_tok,
           'gold': acc(gold_ok), 'end_to_end': acc(e2e_ok),
           'msd_accuracy_normalized': acc(msd_ok), 'pos_accuracy': acc(pos_ok),
           'combined_variant': {'gold': ref['gold'], 'end_to_end': ref['end_to_end'],
                                **ref['tagger']}}

    print(f"  gold      {out['gold']}  (combined {ref['gold']})")
    print(f"  e2e       {out['end_to_end']}  (combined {ref['end_to_end']})")
    print(f"  MSD norm  {out['msd_accuracy_normalized']}  "
          f"(combined {ref['tagger']['msd_accuracy_normalized']})")
    print(f"  POS grubi {out['pos_accuracy']}  "
          f"(combined {ref['tagger']['pos_accuracy']})")

    if out['gold'] != ref['gold']:
        raise SystemExit(f'IDENTITY FAIL: gold {out["gold"]} != {ref["gold"]} - '
                         f'promenjeno je nesto osim tagera; nista nije upisano')
    print('\nIDENTITY: gold-tag tacnost identicna combined varijanti PASS')
    out['e2e_matches_combined_2dp'] = out['end_to_end'] == ref['end_to_end']
    print(f"E2E paritet na dve decimale: "
          f"{'PASS' if out['e2e_matches_combined_2dp'] else 'NE VAZI'}")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Sacuvano: {OUT}')


if __name__ == '__main__':
    main()
