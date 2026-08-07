#!/usr/bin/env python3
"""
Robustness experiment: lemmatization on stripped (diacritics-removed) ReLDI
text. Compares the proposed system (with the restoration Layer 0) against
CLASSLA-Stanza on input from which all diacritics have been removed,
simulating social media text typed without diacritics.

Also runs: bootstrap CI for the accuracy difference, coarse POS vs full MSD.

"""

import gzip
import pickle
import time
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================
# Constants (same as pos_lemmatizer_v2.py)
# ============================================================
EKAVIZATION = {
    'također': 'takođe', 'usprkos': 'uprkos', 'unatoč': 'uprkos',
    'željeti': 'želeti', 'vidjeti': 'videti', 'htjeti': 'hteti',
    'razumjeti': 'razumeti', 'smjeti': 'smeti', 'voljeti': 'voleti',
    'živjeti': 'živeti', 'trpjeti': 'trpeti', 'letjeti': 'leteti',
    'gorjeti': 'goreti', 'sjesti': 'sesti',
    'doživiti': 'doživeti', 'voliti': 'voleti', 'resiti': 'rešiti',
}
LEMMA_CORRECTIONS = {
    'premer': 'premijer', 'mišlenje': 'mišljenje', 'skopje': 'skoplje',
    'k': 'ka', 'tko': 'ko', 'netko': 'neko', 'nitko': 'niko',
    'cel': 'ceo', 'efekt': 'efekat', 'projekt': 'projekat', 'objekt': 'objekat',
    'ambient': 'ambijent', 'ekonomist': 'ekonomista', 'terorist': 'terorista',
    'gardist': 'gardista', 'alijas': 'alijansa', 'nauk': 'nauka',
}
_HTETI_FORMS = {'neće', 'neću', 'nećemo', 'nećete'}
_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


# ============================================================
# Lemmatizer (same as v2)
# ============================================================
class Lemmatizer:
    def __init__(self, srlex_path, v6_restorer=None):
        # Layer 0 = puni v6 modul (Sekcija 4.3); deljen ili kreiran ovde
        if v6_restorer is not None:
            self._v6 = v6_restorer
        else:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
            from v6_restorer import V6Restorer
            self._v6 = V6Restorer(srlex_path)
        self._msd_index = defaultdict(dict)
        self._pos_index = defaultdict(lambda: defaultdict(list))
        self._word_index = defaultdict(list)
        self._ascii_index = defaultdict(list)

        msd_d = defaultdict(lambda: defaultdict(lambda: (None, 0)))
        pos_d = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        word_d = defaultdict(lambda: defaultdict(int))

        with gzip.open(srlex_path, 'rt', encoding='utf-8') as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) < 7: continue
                w, l, m = p[0].lower(), p[1].lower(), p[2]
                l = EKAVIZATION.get(l, l)
                try: fr = int(p[6])
                except: fr = 0
                pos = m[0] if m else '?'
                ex = msd_d[w][m]
                if fr > ex[1]: msd_d[w][m] = (l, fr)
                pos_d[w][pos][l] += fr
                word_d[w][l] += fr

        ascii_freq = defaultdict(lambda: defaultdict(int))
        for w, lf in word_d.items():
            af = w.translate(_DIAC_TO_ASCII)
            ascii_freq[af][w] += sum(lf.values())
        for af, wf in ascii_freq.items():
            self._ascii_index[af] = sorted(wf.items(), key=lambda x: -x[1])

        for w, ms in msd_d.items():
            self._msd_index[w] = {m: l for m, (l, f) in ms.items()}
        for w, ps in pos_d.items():
            for pos, lf in ps.items():
                self._pos_index[w][pos] = sorted(lf.items(), key=lambda x: -x[1])
        for w, lf in word_d.items():
            self._word_index[w] = sorted(lf.items(), key=lambda x: -x[1])

    def _correct(self, lemma):
        if not lemma: return lemma
        lemma = LEMMA_CORRECTIONS.get(lemma, lemma)
        return EKAVIZATION.get(lemma, lemma)

    def lemmatize(self, word, msd=""):
        w = word.lower()
        pos = msd[0] if msd else ""
        if w in _HTETI_FORMS and pos == 'V': return 'hteti'
        # Layer 0: puni modul za restauraciju dijakritika (Sekcija 4.3)
        if w not in self._word_index and not (_DIAC_CHARS & set(word)):
            restored = self._v6.restore(word, msd)
            if restored != word:
                w = restored.lower()
        # Layer 1: MSD exact
        if msd and w in self._msd_index:
            lemma = self._msd_index[w].get(msd)
            if lemma: return self._correct(lemma)
        # Layer 2: POS lookup
        if pos and w in self._pos_index and pos in self._pos_index[w]:
            cands = self._pos_index[w][pos]
            if msd.startswith(('Ap', 'Ag')):
                for l, f in cands:
                    if not l.endswith(('ti', 'ći', 'ci')): return self._correct(l)
            for l, f in cands:
                if l == w: return w
            return self._correct(cands[0][0])
        # MSD prefix relaxation
        if msd and w in self._msd_index:
            for plen in range(len(msd)-1, 0, -1):
                for fm, l in self._msd_index[w].items():
                    if fm.startswith(msd[:plen]): return self._correct(l)
        # Frequency
        if w in self._word_index:
            for l, f in self._word_index[w]:
                if l == w: return w
            return self._correct(self._word_index[w][0][0])
        # Suffix heuristic
        if pos == 'N' and len(w) > 3:
            g = msd[2] if len(msd) > 2 else ''
            n = msd[3] if len(msd) > 3 else ''
            c = msd[4] if len(msd) > 4 else ''
            if g == 'm' and n == 's' and c == 'g' and w.endswith('a'):
                guess = w[:-1]
                if guess in self._word_index: return guess
            if g == 'f' and n == 's' and c == 'g' and w.endswith('e'):
                guess = w[:-1] + 'a'
                if guess in self._word_index: return guess
        # Identity
        if word[0].isupper() and len(word) > 1 and word[1:].islower(): return word
        return w


def load_reldi(path):
    """Load ReLDI CoNLL-UP → [[(word, lemma, xpos), ...], ...]"""
    sentences, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current: sentences.append(current); current = []
                continue
            if line.startswith('#'): continue
            p = line.split('\t')
            if len(p) < 5 or '-' in p[0] or '.' in p[0]: continue
            current.append((p[1], p[2], p[4]))  # word, lemma, xpos
    if current: sentences.append(current)
    return sentences


def strip_word(w):
    """Strip diacritics from a word."""
    return w.translate(_DIAC_TO_ASCII)


def bootstrap_ci(correct_a, correct_b, n, n_boot=10000, alpha=0.05):
    """Bootstrap CI for difference in accuracy between two systems."""
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        acc_a = correct_a[idx].mean()
        acc_b = correct_b[idx].mean()
        diffs.append(acc_a - acc_b)
    diffs = sorted(diffs)
    lo = diffs[int(n_boot * alpha / 2)]
    hi = diffs[int(n_boot * (1 - alpha / 2))]
    return np.mean(diffs), lo, hi


def main():
    base = Path(__file__).parent.parent.parent
    srlex = str(base / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
    reldi_path = str(base / 'data' / 'ReLDI-NormTagNER-sr' / 'reldi-normtagner-sr.conllup')
    ud_test = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'

    print("=" * 70, flush=True)
    print("EXPERIMENT: LEMMATIZATION ON STRIPPED (NO DIACRITICS) TEXT", flush=True)
    print("=" * 70, flush=True)

    # Load lemmatizer
    print("\n1. Loading lemmatizer...", flush=True)
    lem = Lemmatizer(srlex)

    # Load ReLDI
    print("\n2. Loading ReLDI corpus...", flush=True)
    reldi = load_reldi(reldi_path)
    n_reldi = sum(len(s) for s in reldi)
    print(f"   {len(reldi)} sentences, {n_reldi:,} tokens", flush=True)

    # ================================================================
    # EXPERIMENT 1: Our system on STRIPPED ReLDI (gold MSD)
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: OUR SYSTEM ON STRIPPED ReLDI TEXT (gold MSD)", flush=True)
    print("=" * 70, flush=True)

    # a) Normal (original text, gold MSD)
    correct_normal = 0
    total = 0
    for sent in reldi:
        for w, gl, msd in sent:
            total += 1
            pred = lem.lemmatize(w, msd)
            if pred and gl and pred.lower() == gl.lower():
                correct_normal += 1
    acc_normal = round(100 * correct_normal / total, 2)
    print(f"   Normal text (gold MSD):  {acc_normal}% ({correct_normal}/{total})", flush=True)

    # b) Stripped (no diacritics, gold MSD)
    correct_stripped = 0
    total2 = 0
    for sent in reldi:
        for w, gl, msd in sent:
            total2 += 1
            w_stripped = strip_word(w)
            pred = lem.lemmatize(w_stripped, msd)
            if pred and gl and pred.lower() == gl.lower():
                correct_stripped += 1
    acc_stripped = round(100 * correct_stripped / total2, 2)
    improvement = round(acc_stripped - acc_normal, 2)
    print(f"   Stripped text (gold MSD): {acc_stripped}% ({correct_stripped}/{total2})", flush=True)

    # "Normal" ReLDI text is itself noisy (some words already lack
    # diacritics), so the informative comparison is original input vs
    # artificially stripped input, for both systems.

    # ================================================================
    # EXPERIMENT 2: CLASSLA ON STRIPPED ReLDI TEXT
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: CLASSLA-STANZA ON STRIPPED ReLDI TEXT", flush=True)
    print("=" * 70, flush=True)

    # Check if CLASSLA env exists
    classla_python = '/tmp/classla_env/bin/python3'
    classla_available = Path(classla_python).exists()

    if not classla_available:
        print("   CLASSLA env not found, skipping", flush=True)
        classla_acc_stripped = None
    else:
        # Write stripped sentences to temp file for CLASSLA
        import subprocess, tempfile

        # Prepare stripped text
        stripped_sents = []
        for sent in reldi:
            stripped_sents.append([(strip_word(w), gl, msd) for w, gl, msd in sent])

        # Write as CoNLL-U for pre-tokenized input
        tmp_conllu = '/tmp/reldi_stripped.conllu'
        with open(tmp_conllu, 'w', encoding='utf-8') as f:
            for i, sent in enumerate(stripped_sents):
                f.write(f'# sent_id = {i}\n')
                f.write(f'# text = {" ".join(w for w, _, _ in sent)}\n')
                for j, (w, gl, msd) in enumerate(sent, 1):
                    f.write(f'{j}\t{w}\t_\t_\t_\t_\t_\t_\t_\t_\n')
                f.write('\n')

        # CLASSLA script
        classla_script = '''
import sys
import classla

nlp = classla.Pipeline("sr", processors="tokenize,pos,lemma",
                        tokenize_pretokenized=True,
                        use_gpu=False)

# Read pre-tokenized input
sentences = []
current = []
with open("/tmp/reldi_stripped.conllu") as f:
    for line in f:
        line = line.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\\t")
        if len(parts) >= 2:
            current.append(parts[1])
if current:
    sentences.append(current)

# Process
doc = nlp([[w for w in sent] for sent in sentences])

# Output lemmas
for sent in doc.sentences:
    lemmas = [w.lemma for w in sent.words]
    print("\\t".join(lemmas))
'''

        tmp_script = '/tmp/classla_stripped_eval.py'
        with open(tmp_script, 'w') as f:
            f.write(classla_script)

        print("   Running CLASSLA on stripped ReLDI text...", flush=True)
        t0 = time.time()
        result = subprocess.run(
            [classla_python, tmp_script],
            capture_output=True, text=True, timeout=600
        )
        classla_time = time.time() - t0
        print(f"   CLASSLA finished in {classla_time:.1f}s", flush=True)

        if result.returncode != 0:
            print(f"   CLASSLA error: {result.stderr[:500]}", flush=True)
            classla_acc_stripped = None
        else:
            # Parse CLASSLA output
            classla_lemmas = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    classla_lemmas.append(line.strip().split('\t'))

            # Evaluate
            cl_correct = 0
            cl_total = 0
            our_correct_arr = []
            cl_correct_arr = []

            for i, (sent_gold, sent_stripped) in enumerate(zip(reldi, stripped_sents)):
                if i >= len(classla_lemmas):
                    break
                cl_lems = classla_lemmas[i]
                for j, (w_orig, gl, msd) in enumerate(sent_gold):
                    cl_total += 1
                    w_stripped = strip_word(w_orig)

                    # Our prediction on stripped text
                    our_pred = lem.lemmatize(w_stripped, msd)
                    our_ok = 1 if (our_pred and gl and our_pred.lower() == gl.lower()) else 0
                    our_correct_arr.append(our_ok)

                    # CLASSLA prediction
                    if j < len(cl_lems):
                        cl_pred = cl_lems[j]
                        cl_ok = 1 if (cl_pred and gl and cl_pred.lower() == gl.lower()) else 0
                    else:
                        cl_ok = 0
                    cl_correct_arr.append(cl_ok)

            our_correct_arr = np.array(our_correct_arr)
            cl_correct_arr = np.array(cl_correct_arr)

            our_acc = round(100 * our_correct_arr.mean(), 2)
            cl_acc = round(100 * cl_correct_arr.mean(), 2)

            print(f"\n   Results on STRIPPED ReLDI text:", flush=True)
            print(f"   Our system (gold MSD):  {our_acc}%", flush=True)
            print(f"   CLASSLA-Stanza:         {cl_acc}%", flush=True)
            print(f"   Difference:             {our_acc - cl_acc:+.2f} pp", flush=True)

            # McNemar test
            a = int(((our_correct_arr == 1) & (cl_correct_arr == 1)).sum())
            b = int(((our_correct_arr == 1) & (cl_correct_arr == 0)).sum())
            c = int(((our_correct_arr == 0) & (cl_correct_arr == 1)).sum())
            d = int(((our_correct_arr == 0) & (cl_correct_arr == 0)).sum())
            print(f"\n   McNemar contingency:", flush=True)
            print(f"     Both correct: {a}", flush=True)
            print(f"     Only ours correct: {b}", flush=True)
            print(f"     Only CLASSLA correct: {c}", flush=True)
            print(f"     Both wrong: {d}", flush=True)

            if (b + c) > 0:
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                # p-value from chi2 with 1 df
                from scipy import stats
                p_val = 1 - stats.chi2.cdf(chi2, 1)
                print(f"     Chi2 = {chi2:.2f}, p = {p_val:.6f}", flush=True)

            # Bootstrap CI
            mean_diff, ci_lo, ci_hi = bootstrap_ci(our_correct_arr, cl_correct_arr, len(our_correct_arr))
            print(f"\n   Bootstrap 95% CI for difference (ours - CLASSLA):", flush=True)
            print(f"     Mean: {100*mean_diff:+.2f} pp", flush=True)
            print(f"     95% CI: [{100*ci_lo:+.2f}, {100*ci_hi:+.2f}] pp", flush=True)

    # ================================================================
    # EXPERIMENT 3: BOOTSTRAP CI FOR UD TEST SPLIT (standard eval)
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: BOOTSTRAP CI FOR UD TEST SPLIT COMPARISON", flush=True)
    print("=" * 70, flush=True)

    # Load UD test
    ud_sents = []
    current = []
    with open(ud_test) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current: ud_sents.append(current); current = []
                continue
            if line.startswith('#'): continue
            p = line.split('\t')
            if len(p) < 6 or '-' in p[0] or '.' in p[0]: continue
            current.append((p[1], p[2], p[4]))
    if current: ud_sents.append(current)

    # Our predictions (gold MSD)
    our_ud = []
    for sent in ud_sents:
        for w, gl, msd in sent:
            pred = lem.lemmatize(w, msd)
            our_ud.append(1 if (pred and gl and pred.lower() == gl.lower()) else 0)
    our_ud = np.array(our_ud)

    # We need CLASSLA predictions for bootstrap - load from previous eval if available
    # For now, just compute our bootstrap CI
    our_acc_ud = round(100 * our_ud.mean(), 2)
    print(f"   Our accuracy (gold MSD): {our_acc_ud}%", flush=True)

    # Bootstrap CI for our accuracy
    rng = np.random.RandomState(42)
    accs = []
    for _ in range(10000):
        idx = rng.randint(0, len(our_ud), size=len(our_ud))
        accs.append(our_ud[idx].mean())
    accs = sorted(accs)
    lo = round(100 * accs[250], 2)
    hi = round(100 * accs[9749], 2)
    print(f"   Bootstrap 95% CI: [{lo}%, {hi}%]", flush=True)

    # ================================================================
    # EXPERIMENT 4: COARSE POS vs FULL MSD
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 4: COARSE POS vs FULL MSD ON UD TEST", flush=True)
    print("=" * 70, flush=True)

    # Full MSD (standard)
    correct_msd = 0
    # Coarse POS only (first char)
    correct_pos = 0
    # No POS at all
    correct_none = 0
    total_ud = 0

    for sent in ud_sents:
        for w, gl, msd in sent:
            total_ud += 1
            # Full MSD
            pred_msd = lem.lemmatize(w, msd)
            if pred_msd and gl and pred_msd.lower() == gl.lower():
                correct_msd += 1
            # Coarse POS only
            coarse = msd[0] if msd else ""
            pred_pos = lem.lemmatize(w, coarse)
            if pred_pos and gl and pred_pos.lower() == gl.lower():
                correct_pos += 1
            # No POS
            pred_none = lem.lemmatize(w, "")
            if pred_none and gl and pred_none.lower() == gl.lower():
                correct_none += 1

    acc_msd = round(100 * correct_msd / total_ud, 2)
    acc_pos = round(100 * correct_pos / total_ud, 2)
    acc_none = round(100 * correct_none / total_ud, 2)

    print(f"   Full MSD tag:    {acc_msd}% ({correct_msd}/{total_ud})", flush=True)
    print(f"   Coarse POS only: {acc_pos}% ({correct_pos}/{total_ud})", flush=True)
    print(f"   No POS/MSD:      {acc_none}% ({correct_none}/{total_ud})", flush=True)
    print(f"   MSD vs POS gain: {acc_msd - acc_pos:+.2f} pp", flush=True)
    print(f"   POS vs None gain: {acc_pos - acc_none:+.2f} pp", flush=True)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY OF ALL EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)
    print(f"  Exp 1: Our system on stripped ReLDI: {acc_stripped}%", flush=True)
    print(f"  Exp 1: Our system on normal ReLDI:   {acc_normal}%", flush=True)
    print(f"  Exp 3: Our system on UD test (CI):   {our_acc_ud}% [{lo}%, {hi}%]", flush=True)
    print(f"  Exp 4: Full MSD={acc_msd}%, Coarse POS={acc_pos}%, None={acc_none}%", flush=True)

    # Save
    results = {
        'reldi_normal': acc_normal,
        'reldi_stripped_ours': acc_stripped,
        'ud_test_accuracy': our_acc_ud,
        'ud_test_ci_95': [lo, hi],
        'coarse_pos_vs_msd': {
            'full_msd': acc_msd,
            'coarse_pos': acc_pos,
            'no_pos': acc_none,
        }
    }
    out = Path(__file__).parent.parent / 'results' / 'experiments_v5.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}", flush=True)


if __name__ == '__main__':
    main()
