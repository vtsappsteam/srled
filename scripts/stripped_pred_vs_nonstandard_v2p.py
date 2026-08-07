#!/usr/bin/env python3
"""Automatski pipeline (F) naspram CLASSLA nonstandard varijante (N).

Rad na dva mesta (Sekcija 5.4 i Zakljucak) kaze da automatski pipeline
zaostaje "0,69 poena" za ReLDI-treniranom nonstandard varijantom, sa
p = 0,008. Posle retreninga tagera F je porastao (93,48 -> u razmeri rada
93,69), pa je i razlika i njena p-vrednost morala da se ponovo izmeri;
per-token vektori iz julskog racuna nisu sacuvani.

Sta radi:
  1. F: v2p tager taguje ogoljene tokene ReLDI TEST splita, Stage 3
     restauracija pod predvidjenim tagovima, pa lematizacija - isti kod
     kao stripped_predicted_msd_eval.py.
  2. N: CLASSLA-Stanza nonstandard nad istim ogoljenim tokenima. Izlaz se
     cuva TRAJNO (data/evaluation/), ne u /tmp, jer je julski izlaz izgubljen
     brisanjem /tmp-a.
  3. Uparen McNemar F vs N + razlika, u obe razmere (9.108 i 9.088 tokena;
     rad koristi 9.088 jer 20 tokena nema zlatnu lemu).

IDENTITY (bez PASS-a se nista ne pise):
  - F mora reprodukovati 93,48 iz stripped_reldi_predicted_msd.json,
  - N mora reprodukovati 93,84 iz julskog merenja.

Izlaz: results/stripped_pred_vs_nonstandard_v2p.json
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'src'))

import evaluate_three_testsets as ev3                                # noqa: E402
ev3.V7_MODEL = str(ev3.BASE / 'NLTK Treniranje' / 'models_v2p_pnpa'
                   / 'perceptron-tagger-v2p-combined.pickle')

from experiment_stripped_reldi import strip_word                    # noqa: E402
from stripped_test_split_eval import load_reldi_with_split, mcnemar_exact, RELDI  # noqa: E402
from candidate_generator import CandidateGenerator                  # noqa: E402
from pos_disambiguator import POSDisambiguator                      # noqa: E402

CLASSLA_PY = '/tmp/classla_env/bin/python3'
NS_OUT = BASE / 'data' / 'evaluation' / 'classla_stripped_nonstd_test.tsv'
OUT = BASE / 'results' / 'stripped_pred_vs_nonstandard_v2p.json'
REF_F = BASE / 'results' / 'stripped_reldi_predicted_msd.json'
REF_N_JULY = 93.84          # nonstandard, test split, imenilac 9.108 (jul 2026)
PAPER_DEN = 9088            # imenilac rada (20 tokena bez zlatne leme)


def run_classla_nonstandard(sentences):
    """Lemmatizuj vec tokenizovane recenice CLASSLA nonstandard varijantom."""
    if NS_OUT.exists():
        print(f'Koristim postojeci izlaz: {NS_OUT}', flush=True)
        return [line.rstrip('\n').split('\t') for line in
                NS_OUT.open(encoding='utf-8')]
    NS_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_in = '/tmp/reldi_stripped_test_tokens.txt'
    with open(tmp_in, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write('\n'.join(sent) + '\n\n')
    script = f'''
import classla
nlp = classla.Pipeline("sr", processors="tokenize,pos,lemma", type="nonstandard",
                       tokenize_pretokenized=True, use_gpu=False)
sentences, current = [], []
with open("{tmp_in}") as f:
    for line in f:
        line = line.rstrip("\\n")
        if not line:
            if current: sentences.append(current); current = []
            continue
        current.append(line)
if current: sentences.append(current)
doc = nlp(sentences)
with open("{NS_OUT}", "w") as f:
    for sent in doc.sentences:
        f.write("\\t".join(w.lemma if w.lemma else "_" for w in sent.words) + "\\n")
print("DONE")
'''
    runner = '/tmp/classla_nonstd_run.py'
    with open(runner, 'w', encoding='utf-8') as f:
        f.write(script)
    print('Pokrecem CLASSLA nonstandard...', flush=True)
    r = subprocess.run([CLASSLA_PY, runner], capture_output=True, text=True,
                       timeout=7200)
    if 'DONE' not in r.stdout:
        print('CLASSLA STDERR:', r.stderr[-800:])
        raise SystemExit('CLASSLA nonstandard nije zavrsio')
    return [line.rstrip('\n').split('\t') for line in NS_OUT.open(encoding='utf-8')]


def main():
    corpus = load_reldi_with_split(RELDI)
    test_idx = [i for i, (split, s) in enumerate(corpus) if split == 'test']
    sents = [corpus[i][1] for i in test_idx]
    n_tok = sum(len(s) for s in sents)
    print(f'ReLDI test split: {len(sents)} recenica, {n_tok:,} tokena', flush=True)

    stripped = [[strip_word(w) for (w, gl, msd) in s] for s in sents]

    print('Ucitavam v2p combined tager...', flush=True)
    with open(ev3.V7_MODEL, 'rb') as f:
        tagger = pickle.load(f)
    pred_tags = [[ev3.normalize_pnpa(t) for (_, t) in s]
                 for s in tagger.tag_sents(stripped)]
    del tagger

    print('Ucitavam restauracione resurse i lematizator...', flush=True)
    cg = CandidateGenerator(
        ev3.SRLEX,
        supplement_path=str(BASE / 'data' / 'srwac_supplement_v2.json'),
        augment_np_path=str(BASE / 'data' / 'srwac_augment_np.json'))
    disamb = POSDisambiguator(cg, tagger=None)
    lem = ev3.Lemmatizer(ev3.SRLEX, ev3.EXTRA)

    ns_lemmas = run_classla_nonstandard(stripped)
    if len(ns_lemmas) != len(sents):
        raise SystemExit(f'CLASSLA je vratio {len(ns_lemmas)} recenica, '
                         f'ocekivano {len(sents)}')

    vf, vn = [], []
    for si, sent in enumerate(sents):
        if len(ns_lemmas[si]) != len(sent):
            raise SystemExit(f'recenica {si}: {len(ns_lemmas[si])} lema vs '
                             f'{len(sent)} tokena')
        for ti, (w, gl, msd) in enumerate(sent):
            pm = pred_tags[si][ti]
            ws = stripped[si][ti]
            p = lem.lemmatize(disamb._restore_word_pos(ws, pm), pm)
            vf.append(1 if p and gl and p.lower() == gl.lower() else 0)
            nl = ns_lemmas[si][ti]
            vn.append(1 if gl and nl.lower() == gl.lower() else 0)
    vf, vn = np.array(vf), np.array(vn)

    f_acc = 100 * float(vf.mean())
    n_acc = 100 * float(vn.mean())
    ref_f = json.loads(REF_F.read_text(encoding='utf-8'))['F_stage3_pred']
    print(f'\n  F (nas, predvidjeni tagovi) {f_acc:.2f}  (referenca {ref_f})')
    print(f'  N (CLASSLA nonstandard)     {n_acc:.2f}  (jul {REF_N_JULY})')
    fails = []
    if round(f_acc, 2) != ref_f:
        fails.append(f'F {f_acc:.2f} != {ref_f}')
    if abs(n_acc - REF_N_JULY) > 0.005:
        fails.append(f'N {n_acc:.2f} != {REF_N_JULY}')
    if fails:
        raise SystemExit('IDENTITY FAIL: ' + '; '.join(fails) + '; nista nije upisano')
    print('IDENTITY: F i N reprodukovani PASS')

    p, f_only, n_only = mcnemar_exact(vf, vn)
    scale = len(vf) / PAPER_DEN
    res = {'tokens': int(len(vf)), 'paper_denominator': PAPER_DEN,
           'F_correct_tokens': int(vf.sum()), 'N_correct_tokens': int(vn.sum()),
           'F_stage3_pred': round(f_acc, 2),
           'N_classla_nonstandard': round(n_acc, 2),
           'F_paper_scale': round(f_acc * scale, 2),
           'N_paper_scale': round(n_acc * scale, 2),
           'gap_pp': round(n_acc - f_acc, 2),
           'gap_pp_paper_scale': round((n_acc - f_acc) * scale, 2),
           'mcnemar_F_vs_N': {'p': p, 'F_only': f_only, 'N_only': n_only},
           'classla_output': str(NS_OUT)}
    print(f'  zaostatak F za N: {res["gap_pp"]:.2f} pp '
          f'({res["gap_pp_paper_scale"]:.2f} u razmeri rada), '
          f'p = {p}, samo F tacan {f_only}, samo N tacan {n_only}')
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Sacuvano: {OUT}')


if __name__ == '__main__':
    main()
