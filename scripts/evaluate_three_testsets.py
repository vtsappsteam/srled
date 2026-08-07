#!/usr/bin/env python3
"""
Comprehensive evaluation on THREE independent test sets from different domains.

1. UD Serbian-SET test (news) - MSD tags
2. ReLDI test (Twitter) - MSD tags
3. SrpKor4Tagging test (literary + admin) - UPOS tags mapped to MSD

Reference implementation of the FINAL system configuration:
  - lexicon: srLex + expanded_supplement_v2.json (SETimes.SR 2.0 + ReLDI
    + SrpKor4Tagging train splits)
  - lemma corrections: data/lemma_corrections_v2.csv (17 manual + 131
    linguist-reviewed data-driven pairs = 148)
  - tagger: models_v7_pnpa (srWaC pretrain + Pn/Pa fine-tune); predicted
    Pn*/Pa* MSDs are normalized to P* inside Lemmatizer.lemmatize (srLex
    has no Pn/Pa tags; gold evaluation tags are unaffected)

"""

import csv
import gzip
import json
import pickle
import subprocess
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
from datasets import load_dataset

BASE = Path(__file__).resolve().parent.parent.parent
if not (BASE / 'POS-Aware-Stemmer').exists() and (BASE.parent / 'POS-Aware-Stemmer').exists():
    BASE = BASE.parent  # repo/scripts layout is one level deeper than scripts/
SRLEX = str(BASE / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
EXTRA = str(Path(__file__).parent.parent / 'data' / 'expanded_supplement_v2.json')
V7_MODEL = str(BASE / 'NLTK Treniranje' / 'models_v7_pnpa' / 'perceptron-tagger-srwac-pnpa.pickle')
CORRECTIONS_CSV = Path(__file__).parent.parent / 'data' / 'lemma_corrections_v2.csv'
CLASSLA_PY = '/tmp/classla_env/bin/python3'

# UPOS to MSD first-character mapping
UPOS_TO_MSD = {
    'NOUN': 'N', 'PROPN': 'N', 'VERB': 'V', 'AUX': 'V',
    'ADJ': 'A', 'ADV': 'R', 'PRON': 'P', 'DET': 'P',
    'ADP': 'S', 'CCONJ': 'C', 'SCONJ': 'C', 'NUM': 'M',
    'PART': 'Q', 'INTJ': 'I', 'PUNCT': 'Z', 'X': 'X',
}

EKAVIZATION = {
    'također': 'takođe', 'usprkos': 'uprkos', 'unatoč': 'uprkos',
    'željeti': 'želeti', 'vidjeti': 'videti', 'htjeti': 'hteti',
    'razumjeti': 'razumeti', 'smjeti': 'smeti', 'voljeti': 'voleti',
    'živjeti': 'živeti', 'trpjeti': 'trpeti', 'letjeti': 'leteti',
    'gorjeti': 'goreti', 'sjesti': 'sesti',
    'doživiti': 'doživeti', 'voliti': 'voleti', 'resiti': 'rešiti',
}
# The original 17 manually curated corrections (kept for scripts that
# reproduce historical configurations; the deployed table is the CSV below).
LEMMA_CORRECTIONS_MANUAL17 = {
    'premer': 'premijer', 'mišlenje': 'mišljenje', 'skopje': 'skoplje',
    'k': 'ka', 'tko': 'ko', 'netko': 'neko', 'nitko': 'niko',
    'cel': 'ceo', 'efekt': 'efekat', 'projekt': 'projekat', 'objekt': 'objekat',
    'ambient': 'ambijent', 'ekonomist': 'ekonomista', 'terorist': 'terorista',
    'gardist': 'gardista', 'alijas': 'alijansa', 'nauk': 'nauka',
}


def load_lemma_corrections(path=CORRECTIONS_CSV):
    """Final correction table: 17 manual + 131 reviewed data-driven pairs."""
    with open(path, encoding='utf-8') as f:
        return {r['source_lemma']: r['corrected_lemma']
                for r in csv.DictReader(f)}


LEMMA_CORRECTIONS = load_lemma_corrections()


def normalize_pnpa(tag):
    """Pn*/Pa* -> P* (srLex and the supplement have no Pn/Pa MSDs)."""
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag


_HTETI = {'neće', 'neću', 'nećemo', 'nećete'}
_D2A = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DC = set('čćžšđČĆŽŠĐ')


class Lemmatizer:
    def __init__(self, srlex_path, extra_path=None, v6_restorer=None):
        self._mi = defaultdict(dict)
        self._pi = defaultdict(lambda: defaultdict(list))
        self._wi = defaultdict(list)
        self._ai = defaultdict(list)
        # Layer 0 = puni v6 modul (Sekcija 4.3); deljen ili kreiran ovde
        if v6_restorer is not None:
            self._v6 = v6_restorer
        else:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
            from v6_restorer import V6Restorer
            self._v6 = V6Restorer(srlex_path, extra_dict_path=extra_path)
        md = defaultdict(lambda: defaultdict(lambda: (None, 0)))
        pd = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        wd = defaultdict(lambda: defaultdict(int))
        with gzip.open(srlex_path, 'rt', encoding='utf-8') as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) < 7: continue
                w, l, m = p[0].lower(), p[1].lower(), p[2]
                l = EKAVIZATION.get(l, l)
                try: fr = int(p[6])
                except: fr = 0
                pos = m[0] if m else '?'
                ex = md[w][m]
                if fr > ex[1]: md[w][m] = (l, fr)
                pd[w][pos][l] += fr; wd[w][l] += fr
        if extra_path and Path(extra_path).exists():
            with open(extra_path) as f:
                extra = json.load(f)
            for w, entries in extra.items():
                w = w.lower()
                for lemma, msd, freq in entries:
                    lemma = lemma.lower()
                    pos = msd[0] if msd else '?'
                    ex = md[w].get(msd, (None, 0))
                    if freq > ex[1]: md[w][msd] = (lemma, freq)
                    pd[w][pos][lemma] += freq; wd[w][lemma] += freq
        af2 = defaultdict(lambda: defaultdict(int))
        for w, lf in wd.items():
            af = w.translate(_D2A); af2[af][w] += sum(lf.values())
        for af, wf in af2.items():
            self._ai[af] = sorted(wf.items(), key=lambda x: -x[1])
        for w, ms in md.items():
            self._mi[w] = {m: l for m, (l, f) in ms.items()}
        for w, ps in pd.items():
            for pos, lf in ps.items():
                self._pi[w][pos] = sorted(lf.items(), key=lambda x: -x[1])
        for w, lf in wd.items():
            self._wi[w] = sorted(lf.items(), key=lambda x: -x[1])

    def _correct(self, lemma):
        if not lemma: return lemma
        lemma = LEMMA_CORRECTIONS.get(lemma, lemma)
        return EKAVIZATION.get(lemma, lemma)

    def lemmatize(self, word, msd=""):
        msd = normalize_pnpa(msd)
        w = word.lower(); pos = msd[0] if msd else ""
        if w in _HTETI and pos == 'V': return 'hteti'
        # Layer 0: puni modul za restauraciju dijakritika (Sekcija 4.3)
        if w not in self._wi and not (_DC & set(word)):
            restored = self._v6.restore(word, msd)
            if restored != word:
                w = restored.lower()
        if msd and w in self._mi:
            l = self._mi[w].get(msd)
            if l: return self._correct(l)
        if pos and w in self._pi and pos in self._pi[w]:
            cands = self._pi[w][pos]
            if msd.startswith(('Ap', 'Ag')):
                for l, f in cands:
                    if not l.endswith(('ti', 'ći', 'ci')): return self._correct(l)
            for l, f in cands:
                if l == w: return w
            return self._correct(cands[0][0])
        if msd and w in self._mi:
            for plen in range(len(msd)-1, 0, -1):
                for fm, l in self._mi[w].items():
                    if fm.startswith(msd[:plen]): return self._correct(l)
        if w in self._wi:
            for l, f in self._wi[w]:
                if l == w: return w
            return self._correct(self._wi[w][0][0])
        if pos == 'N' and len(w) > 3:
            g = msd[2] if len(msd) > 2 else ''
            n = msd[3] if len(msd) > 3 else ''
            c = msd[4] if len(msd) > 4 else ''
            if g == 'm' and n == 's' and c == 'g' and w.endswith('a'):
                guess = w[:-1]
                if guess in self._wi: return guess
            if g == 'f' and n == 's' and c == 'g' and w.endswith('e'):
                guess = w[:-1] + 'a'
                if guess in self._wi: return guess
        if word[0].isupper() and len(word) > 1 and word[1:].islower(): return word
        return w


def load_conllu(path):
    sentences, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current: sentences.append(current); current = []
                continue
            if line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) >= 5 and '-' not in parts[0] and '.' not in parts[0]:
                if parts[2] != '_':
                    current.append((parts[1], parts[2], parts[4]))
    if current: sentences.append(current)
    return sentences


def load_reldi_test(path):
    test, current, is_test = [], [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current and is_test: test.append(current)
                current = []
                continue
            if line.startswith('# contained_in_datasets'):
                is_test = 'test' in line; continue
            if line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) >= 5 and '-' not in parts[0] and '.' not in parts[0]:
                if parts[2] != '_':
                    current.append((parts[1], parts[2], parts[4]))
    if current and is_test: test.append(current)
    return test


def load_srpkor_test():
    """Load SrpKor4Tagging last 10% as test, map UPOS to MSD."""
    ds = load_dataset('jerteh/SrpKor4Tagging')['train']
    test_size = len(ds) // 10
    test_indices = range(len(ds) - test_size, len(ds))

    sentences = []
    for idx in test_indices:
        sent = []
        tokens = ds['token'][idx]
        lemmas = ds['lemma'][idx]
        upos_tags = ds['ud'][idx]
        for tok, lem, upos in zip(tokens, lemmas, upos_tags):
            msd_char = UPOS_TO_MSD.get(upos, 'X')
            if upos != 'PUNCT':
                sent.append((tok, lem, msd_char))
        if sent:
            sentences.append(sent)
    return sentences


def run_classla(test_sents, name):
    """Run CLASSLA on test set."""
    tmp_input = f'/tmp/classla_{name}.conllu'
    with open(tmp_input, 'w', encoding='utf-8') as f:
        for i, sent in enumerate(test_sents):
            f.write(f'# sent_id = {i}\n')
            for j, (w, l, m) in enumerate(sent, 1):
                f.write(f'{j}\t{w}\t_\t_\t_\t_\t_\t_\t_\t_\n')
            f.write('\n')

    script = f'''
import classla
nlp = classla.Pipeline("sr", processors="tokenize,pos,lemma",
                        tokenize_pretokenized=True, use_gpu=False)
sentences = []
current = []
with open("{tmp_input}") as f:
    for line in f:
        line = line.strip()
        if not line:
            if current: sentences.append(current); current = []
            continue
        if line.startswith("#"): continue
        parts = line.split("\\t")
        if len(parts) >= 2: current.append(parts[1])
if current: sentences.append(current)
doc = nlp([[w for w in sent] for sent in sentences])
for sent in doc.sentences:
    print("\\t".join([w.lemma for w in sent.words]))
'''
    tmp_script = f'/tmp/classla_{name}.py'
    with open(tmp_script, 'w') as f:
        f.write(script)

    result = subprocess.run([CLASSLA_PY, tmp_script],
                           capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  CLASSLA error: {result.stderr[:200]}")
        return None
    return [line.split('\t') for line in result.stdout.strip().split('\n') if line.strip()]


def mcnemar(a, b):
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    if (bb + cc) == 0: return 1.0, bb, cc
    chi2 = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return round(1 - stats.chi2.cdf(chi2, 1), 4), bb, cc


def main():
    print("=" * 70)
    print("EVALUATION ON THREE INDEPENDENT TEST SETS")
    print("=" * 70)

    # Load lemmatizer
    print("\n1. Loading lemmatizer (expanded dict)...", flush=True)
    lem = Lemmatizer(SRLEX, EXTRA)

    # Load tagger
    print("   Loading srwac_pnpa tagger...", flush=True)
    with open(V7_MODEL, 'rb') as f:
        tagger = pickle.load(f)

    # Load all test sets
    print("\n2. Loading test sets...", flush=True)
    testsets = {}

    # UD Serbian-SET
    ud_test = load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    testsets['UD-SET (news)'] = ud_test
    print(f"   UD-SET: {len(ud_test)} sent, {sum(len(s) for s in ud_test):,} tokens")

    # ReLDI
    reldi_path = str(BASE / 'data' / 'ReLDI-NormTagNER-sr' / 'reldi-normtagner-sr.conllup')
    reldi_test = load_reldi_test(reldi_path)
    testsets['ReLDI (Twitter)'] = reldi_test
    print(f"   ReLDI: {len(reldi_test)} sent, {sum(len(s) for s in reldi_test):,} tokens")

    # SrpKor4Tagging
    print("   Loading SrpKor4Tagging from HuggingFace...", flush=True)
    srpkor_test = load_srpkor_test()
    testsets['SrpKor (lit+admin)'] = srpkor_test
    print(f"   SrpKor: {len(srpkor_test)} sent, {sum(len(s) for s in srpkor_test):,} tokens")

    # Evaluate each
    all_results = {}

    for ts_name, test_sents in testsets.items():
        print(f"\n{'='*70}")
        print(f"  {ts_name}")
        print(f"{'='*70}")

        n_tokens = sum(len(s) for s in test_sents)
        is_msd = ts_name != 'SrpKor (lit+admin)'

        # Our system with gold tags
        our = []
        for sent in test_sents:
            for w, gl, msd in sent:
                pred = lem.lemmatize(w, msd)
                our.append(1 if pred and gl and pred.lower() == gl.lower() else 0)
        our = np.array(our)
        our_acc = round(100 * our.mean(), 2)
        print(f"  Our system (gold tags):  {our_acc}%  ({int(our.sum())}/{len(our)})", flush=True)

        # Our system with predicted tags (only for MSD datasets)
        our_pred_acc = None
        if is_msd:
            words_by_sent = [[w for w, l, m in s] for s in test_sents]
            pred_tags = tagger.tag_sents(words_by_sent)
            our_pred = []
            for si, sent in enumerate(test_sents):
                for ti, (w, gl, msd) in enumerate(sent):
                    pm = pred_tags[si][ti][1]
                    pl = lem.lemmatize(w, pm)
                    our_pred.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
            our_pred = np.array(our_pred)
            our_pred_acc = round(100 * our_pred.mean(), 2)
            print(f"  Our system (pred tags):  {our_pred_acc}%", flush=True)

        # CLASSLA
        print(f"  Running CLASSLA...", flush=True)
        cl_lemmas = run_classla(test_sents, ts_name.replace(' ', '_').replace('(', '').replace(')', ''))

        cl_acc = None
        p_val = None
        if cl_lemmas:
            cl = []
            for i, sent in enumerate(test_sents):
                if i >= len(cl_lemmas): break
                for j, (w, gl, msd) in enumerate(sent):
                    if j < len(cl_lemmas[i]):
                        ok = 1 if cl_lemmas[i][j] and gl and cl_lemmas[i][j].lower() == gl.lower() else 0
                    else:
                        ok = 0
                    cl.append(ok)
            cl = np.array(cl)
            cl_acc = round(100 * cl.mean(), 2)

            min_len = min(len(our), len(cl))
            p_val, b, c = mcnemar(our[:min_len], cl[:min_len])

            print(f"  CLASSLA-Stanza:          {cl_acc}%", flush=True)
            print(f"  Diff (ours - CL):        {our_acc - cl_acc:+.2f} pp", flush=True)
            print(f"  McNemar p:               {p_val}", flush=True)
            print(f"  Only ours correct:       {b}", flush=True)
            print(f"  Only CLASSLA correct:    {c}", flush=True)

        all_results[ts_name] = {
            'domain': ts_name,
            'tokens': n_tokens,
            'our_gold': our_acc,
            'our_pred': our_pred_acc,
            'classla': cl_acc,
            'p_value': p_val,
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Test set':<22} {'Domain':<15} {'N':>7} {'Ours':>8} {'CLASSLA':>9} {'Δ':>7} {'p':>8}")
    print(f"  {'-'*78}")
    for name, r in all_results.items():
        cl = f"{r['classla']}%" if r['classla'] else "N/A"
        delta = f"{r['our_gold'] - r['classla']:+.2f}" if r['classla'] else "N/A"
        p = f"{r['p_value']}" if r['p_value'] is not None else "N/A"
        domain = 'News' if 'UD' in name else ('Twitter' if 'ReLDI' in name else 'Lit+Admin')
        print(f"  {name:<22} {domain:<15} {r['tokens']:>7,} {r['our_gold']:>7}% {cl:>9} {delta:>7} {p:>8}")

    # Save
    out = Path(__file__).parent.parent / 'results' / 'three_testsets_evaluation.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
