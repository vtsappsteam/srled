#!/usr/bin/env python3
"""
Expand supplementary dictionary from gold-annotated corpora.

Extracts (word, lemma, MSD) triples from SETimes.SR 2.0 and ReLDI
that are NOT in srLex, creating a supplementary lookup table.
Then fine-tunes POS tagger on combined data and evaluates.

"""

import gzip
import json
import pickle
import time
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path(__file__).parent.parent.parent
SRLEX = str(BASE / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
RELDI = str(BASE / 'data' / 'ReLDI-NormTagNER-sr' / 'reldi-normtagner-sr.conllup')
SETIMES2_TRAIN = '/tmp/SETimes_SR_2.0/train.conllu'
SETIMES2_DEV = '/tmp/SETimes_SR_2.0/dev.conllu'
SETIMES2_TEST = '/tmp/SETimes_SR_2.0/test.conllu'
V5_MODEL = BASE / 'NLTK Treniranje' / 'models_v5' / 'perceptron-tagger-ud-clean.pickle'
V7_MODEL = BASE / 'NLTK Treniranje' / 'models_v7' / 'perceptron-tagger-expanded.pickle'
SUPP_DICT_PATH = Path(__file__).parent.parent / 'data' / 'expanded_supplement.json'

# Lemmatizer constants
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


def load_conllu(path):
    """Load CoNLL-U → [[(word, lemma, xpos), ...], ...]"""
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
                current.append((parts[1], parts[2], parts[4]))
    if current: sentences.append(current)
    return sentences


def load_reldi_splits(path):
    """Load ReLDI with train/test split."""
    train, test = [], []
    current = []
    split = 'other'
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    if 'train' in split: train.append(current)
                    elif 'test' in split: test.append(current)
                    current = []
                continue
            if line.startswith('# contained_in_datasets'):
                if 'train' in line: split = 'train'
                elif 'test' in line: split = 'test'
                else: split = 'other'
                continue
            if line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) >= 5 and '-' not in parts[0] and '.' not in parts[0]:
                current.append((parts[1], parts[2], parts[4]))
    if current:
        if 'train' in split: train.append(current)
        elif 'test' in split: test.append(current)
    return train, test


def load_srlex_words(path):
    """Load all word forms from srLex."""
    words = set()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                words.add(parts[0].lower())
    return words


class Lemmatizer:
    """Same as pos_lemmatizer_v2 but with optional extra dictionary."""

    def __init__(self, srlex_path, extra_dict=None):
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

        # Add extra dictionary entries
        if extra_dict:
            for w, entries in extra_dict.items():
                w = w.lower()
                for lemma, msd, freq in entries:
                    lemma = lemma.lower()
                    pos = msd[0] if msd else '?'
                    ex = msd_d[w].get(msd, (None, 0))
                    if freq > ex[1]:
                        msd_d[w][msd] = (lemma, freq)
                    pos_d[w][pos][lemma] += freq
                    word_d[w][lemma] += freq

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
        if w not in self._word_index and not (_DIAC_CHARS & set(word)):
            af = w.translate(_DIAC_TO_ASCII)
            cands = self._ascii_index.get(af if af != w else w, [])
            if cands:
                diac = [(c, f) for c, f in cands if c != af]
                if diac:
                    if pos:
                        for c, f in diac:
                            if pos in self._pos_index.get(c, {}):
                                w = c; break
                        else: w = diac[0][0]
                    else: w = diac[0][0]
        if msd and w in self._msd_index:
            lemma = self._msd_index[w].get(msd)
            if lemma: return self._correct(lemma)
        if pos and w in self._pos_index and pos in self._pos_index[w]:
            cands = self._pos_index[w][pos]
            if msd.startswith(('Ap', 'Ag')):
                for l, f in cands:
                    if not l.endswith(('ti', 'ći', 'ci')): return self._correct(l)
            for l, f in cands:
                if l == w: return w
            return self._correct(cands[0][0])
        if msd and w in self._msd_index:
            for plen in range(len(msd)-1, 0, -1):
                for fm, l in self._msd_index[w].items():
                    if fm.startswith(msd[:plen]): return self._correct(l)
        if w in self._word_index:
            for l, f in self._word_index[w]:
                if l == w: return w
            return self._correct(self._word_index[w][0][0])
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
        if word[0].isupper() and len(word) > 1 and word[1:].islower(): return word
        return w


def evaluate(lem, test_sents, tagger=None):
    correct_gold = correct_pred = total = 0
    if tagger:
        words_by_sent = [[w for w, l, m in s] for s in test_sents]
        pred_tags = tagger.tag_sents(words_by_sent)
    for si, sent in enumerate(test_sents):
        for ti, (w, gl, msd) in enumerate(sent):
            total += 1
            pred_g = lem.lemmatize(w, msd)
            if pred_g and gl and pred_g.lower() == gl.lower(): correct_gold += 1
            if tagger:
                pred_p = lem.lemmatize(w, pred_tags[si][ti][1])
                if pred_p and gl and pred_p.lower() == gl.lower(): correct_pred += 1
    return {
        'gold': round(100 * correct_gold / total, 2),
        'pred': round(100 * correct_pred / total, 2) if tagger else None,
        'total': total,
    }


def main():
    print("=" * 60)
    print("DICTIONARY EXPANSION + FINE-TUNING")
    print("=" * 60)

    # 1. Load srLex vocabulary
    print("\n1. Loading srLex vocabulary...", flush=True)
    srlex_words = load_srlex_words(SRLEX)
    print(f"   srLex words: {len(srlex_words):,}")

    # 2. Load training corpora
    print("\n2. Loading training corpora...", flush=True)
    setimes2_train = load_conllu(SETIMES2_TRAIN)
    setimes2_dev = load_conllu(SETIMES2_DEV)
    reldi_train, reldi_test = load_reldi_splits(RELDI)

    n_set = sum(len(s) for s in setimes2_train) + sum(len(s) for s in setimes2_dev)
    n_rel = sum(len(s) for s in reldi_train)
    print(f"   SETimes 2.0 train+dev: {len(setimes2_train)+len(setimes2_dev)} sent, {n_set:,} tokens")
    print(f"   ReLDI train: {len(reldi_train)} sent, {n_rel:,} tokens")
    print(f"   ReLDI test: {len(reldi_test)} sent, {sum(len(s) for s in reldi_test):,} tokens")

    # 3. Extract new word-lemma pairs NOT in srLex
    print("\n3. Extracting OOV word-lemma pairs...", flush=True)
    extra_dict = defaultdict(list)  # word -> [(lemma, msd, freq)]
    new_pairs = Counter()

    all_train_data = []
    for sent in setimes2_train + setimes2_dev + reldi_train:
        for w, lemma, msd in sent:
            w_lower = w.lower()
            if w_lower not in srlex_words and lemma != '_' and len(w) > 1:
                key = (w_lower, lemma.lower(), msd)
                new_pairs[key] += 1

    # Build extra dict
    for (w, lemma, msd), freq in new_pairs.items():
        if freq >= 1:  # Include even single occurrences from gold data
            extra_dict[w].append((lemma, msd, freq))

    unique_words = len(extra_dict)
    total_entries = sum(len(v) for v in extra_dict.values())
    print(f"   New OOV words: {unique_words:,}")
    print(f"   New word-lemma-msd entries: {total_entries:,}")

    # Show examples
    print(f"   Examples:")
    for (w, l, m), f in new_pairs.most_common(10):
        print(f"     {w:<20} -> {l:<20} [{m}] (freq={f})")

    # 4. Save expanded dictionary
    print(f"\n4. Saving expanded dictionary...", flush=True)
    save_dict = {w: [(l, m, f) for l, m, f in entries]
                 for w, entries in extra_dict.items()}
    with open(SUPP_DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=0)
    print(f"   Saved: {SUPP_DICT_PATH}")
    print(f"   Size: {SUPP_DICT_PATH.stat().st_size / 1024:.0f} KB")

    # 5. Evaluate: baseline (srLex only) vs expanded (srLex + extra)
    print(f"\n5. Evaluating baseline vs expanded...", flush=True)

    # Load tagger
    with open(V5_MODEL, 'rb') as f:
        tagger_v5 = pickle.load(f)

    # Baseline lemmatizer
    lem_base = Lemmatizer(SRLEX)
    # Expanded lemmatizer
    lem_expanded = Lemmatizer(SRLEX, extra_dict=dict(extra_dict))

    # UD test
    ud_test = load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    base_ud = evaluate(lem_base, ud_test, tagger_v5)
    exp_ud = evaluate(lem_expanded, ud_test, tagger_v5)

    # ReLDI test
    base_reldi = evaluate(lem_base, reldi_test, tagger_v5)
    exp_reldi = evaluate(lem_expanded, reldi_test, tagger_v5)

    print(f"\n   {'Dataset':<20} {'Metric':<15} {'Baseline':>10} {'Expanded':>10} {'Delta':>8}")
    print(f"   {'-'*65}")
    print(f"   {'UD test':<20} {'gold MSD':<15} {base_ud['gold']:>9}% {exp_ud['gold']:>9}% {exp_ud['gold']-base_ud['gold']:>+7.2f}")
    print(f"   {'UD test':<20} {'pred MSD':<15} {base_ud['pred']:>9}% {exp_ud['pred']:>9}% {exp_ud['pred']-base_ud['pred']:>+7.2f}")
    print(f"   {'ReLDI test':<20} {'gold MSD':<15} {base_reldi['gold']:>9}% {exp_reldi['gold']:>9}% {exp_reldi['gold']-base_reldi['gold']:>+7.2f}")
    print(f"   {'ReLDI test':<20} {'pred MSD':<15} {base_reldi['pred']:>9}% {exp_reldi['pred']:>9}% {exp_reldi['pred']-base_reldi['pred']:>+7.2f}")

    # 6. Fine-tune tagger on combined data (UD train + SETimes2 train + ReLDI train)
    print(f"\n6. Fine-tuning v7 tagger on combined data...", flush=True)

    with open(V5_MODEL, 'rb') as f:
        tagger_v7 = pickle.load(f)

    # Prepare combined training data
    ud_train = load_conllu(str(Path('/tmp/UD_Serbian-SET/sr_set-ud-train.conllu')))
    combined_tagged = []
    for sent in setimes2_train + setimes2_dev:
        combined_tagged.append([(w, m) for w, l, m in sent])
    for sent in reldi_train:
        combined_tagged.append([(w, m) for w, l, m in sent])

    print(f"   Combined train: {len(combined_tagged)} sentences")
    t0 = time.time()
    tagger_v7.train(combined_tagged, nr_iter=3)
    print(f"   Fine-tuned in {time.time()-t0:.1f}s")

    V7_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(V7_MODEL, 'wb') as f:
        pickle.dump(tagger_v7, f)
    print(f"   Saved: {V7_MODEL}")

    # 7. Final evaluation: expanded dict + v7 tagger
    print(f"\n7. Final evaluation (expanded dict + v7 tagger)...", flush=True)
    final_ud = evaluate(lem_expanded, ud_test, tagger_v7)
    final_reldi = evaluate(lem_expanded, reldi_test, tagger_v7)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"\n   {'Config':<35} {'UD test':>10} {'ReLDI test':>12}")
    print(f"   {'-'*58}")
    print(f"   {'Baseline (srLex + v5)':<35} {base_ud['pred']:>9}% {base_reldi['pred']:>11}%")
    print(f"   {'+ expanded dict':<35} {exp_ud['pred']:>9}% {exp_reldi['pred']:>11}%")
    print(f"   {'+ expanded dict + v7 tagger':<35} {final_ud['pred']:>9}% {final_reldi['pred']:>11}%")
    print(f"\n   Improvement on ReLDI: {final_reldi['pred'] - base_reldi['pred']:+.2f} pp")
    print(f"   Regression on UD:     {final_ud['pred'] - base_ud['pred']:+.2f} pp")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
