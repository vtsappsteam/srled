"""
Vectorized averaged-perceptron tagger with a compact, memory-mapped model format.

Produces predictions identical to nltk.tag.perceptron.PerceptronTagger:
the feature extraction and tie-breaking semantics are replicated exactly,
only the scoring is reimplemented with numpy over a CSR weight matrix.

Model format (directory):
  features.marisa   marisa_trie.Trie over feature strings (feature -> row id)
  row_ptr.npy       int64 [n_features + 1]  CSR row pointers
  col_idx.npy       int32 [nnz]             class indices
  w_val.npy         float64 [nnz]           weights
  meta.pickle       {'classes': [...], 'tagdict': {...}}

"""

import pickle
from pathlib import Path

import marisa_trie
import numpy as np

START = ["-START-", "-START2-"]
END = ["-END-", "-END2-"]


def _normalize(word):
    if "-" in word and word[0] != "-":
        return "!HYPHEN"
    if word.isdigit() and len(word) == 4:
        return "!YEAR"
    if word and word[0].isdigit():
        return "!DIGITS"
    return word.lower()


def _get_feature_keys(i, word, context, prev, prev2):
    i += 2  # len(START)
    c_i = context[i]
    c_m1 = context[i - 1]
    c_p1 = context[i + 1]
    return (
        "bias",
        "i suffix " + word[-3:],
        "i pref1 " + (word[0] if word else ""),
        "i-1 tag " + prev,
        "i-2 tag " + prev2,
        "i tag+i-2 tag " + prev + " " + prev2,
        "i word " + c_i,
        "i-1 tag+i word " + prev + " " + c_i,
        "i-1 word " + c_m1,
        "i-1 suffix " + c_m1[-3:],
        "i-2 word " + context[i - 2],
        "i+1 word " + c_p1,
        "i+1 suffix " + c_p1[-3:],
        "i+2 word " + context[i + 2],
    )


class FastPerceptronTagger:
    """Drop-in replacement for PerceptronTagger.tag() with numpy scoring."""

    def __init__(self, features_trie, row_ptr, col_idx, w_val, classes, tagdict):
        self._trie = features_trie
        self._row_ptr = row_ptr
        self._col_idx = col_idx
        self._w_val = w_val
        # Classes sorted so that on score ties the alphabetically largest
        # label wins, matching NLTK's max(classes, key=lambda l: (score, l)).
        self._classes = classes
        self._n_classes = len(classes)
        self.tagdict = tagdict

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        trie = marisa_trie.Trie()
        trie.load(str(model_dir / 'features.marisa'))
        row_ptr = np.load(model_dir / 'row_ptr.npy', mmap_mode='r')
        col_idx = np.load(model_dir / 'col_idx.npy', mmap_mode='r')
        w_val = np.load(model_dir / 'w_val.npy', mmap_mode='r')
        with open(model_dir / 'meta.pickle', 'rb') as f:
            meta = pickle.load(f)
        return cls(trie, row_ptr, col_idx, w_val,
                   meta['classes'], meta['tagdict'])

    def tag(self, tokens):
        trie = self._trie
        row_ptr = self._row_ptr
        col_idx = self._col_idx
        w_val = self._w_val
        classes = self._classes
        n_classes = self._n_classes
        tagdict = self.tagdict

        prev, prev2 = START
        output = []
        context = START + [_normalize(w) for w in tokens] + END

        for i, word in enumerate(tokens):
            tag = tagdict.get(word)
            if not tag:
                keys = _get_feature_keys(i, word, context, prev, prev2)
                cols = []
                vals = []
                for k in keys:
                    rid = trie.get(k)
                    if rid is None:
                        continue
                    s, e = row_ptr[rid], row_ptr[rid + 1]
                    if s != e:
                        cols.append(col_idx[s:e])
                        vals.append(w_val[s:e])
                if not cols:
                    # No active features: all scores zero, NLTK returns
                    # the alphabetically largest class.
                    tag = classes[-1]
                else:
                    scores = np.bincount(np.concatenate(cols),
                                         weights=np.concatenate(vals),
                                         minlength=n_classes)
                    # Classes are sorted, so among tied scores the highest
                    # index is the alphabetically largest label - the same
                    # tie-break as NLTK's max(classes, key=(score, label)).
                    ties = np.flatnonzero(scores == scores.max())
                    tag = classes[ties[-1]]
            output.append((word, tag))
            prev2 = prev
            prev = tag
        return output


def build_from_nltk(nltk_pickle_path, out_dir, prune_below=0.0):
    """Convert an NLTK PerceptronTagger pickle to the compact format.

    prune_below: drop (feature, class) weights with |w| < prune_below,
    then drop features left with no weights. 0.0 keeps everything
    except empty/zero entries, which never influence predictions.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(nltk_pickle_path, 'rb') as f:
        tagger = pickle.load(f)

    classes = sorted(tagger.model.classes)
    cls2idx = {c: j for j, c in enumerate(classes)}

    # Collect surviving features
    kept = []
    for feat, cw in tagger.model.weights.items():
        entries = [(cls2idx[c], w) for c, w in cw.items()
                   if w != 0 and abs(w) >= prune_below]
        if entries:
            kept.append((feat, entries))

    trie = marisa_trie.Trie(feat for feat, _ in kept)

    # CSR ordered by trie row id
    by_rid = [None] * len(trie)
    for feat, entries in kept:
        by_rid[trie[feat]] = entries

    nnz = sum(len(e) for e in by_rid)
    row_ptr = np.zeros(len(by_rid) + 1, dtype=np.int64)
    col_idx = np.zeros(nnz, dtype=np.int32)
    # float64 matches NLTK's Python-float weights exactly, so score sums
    # differ from NLTK only by addition order
    w_val = np.zeros(nnz, dtype=np.float64)
    pos = 0
    for rid, entries in enumerate(by_rid):
        for c, w in entries:
            col_idx[pos] = c
            w_val[pos] = w
            pos += 1
        row_ptr[rid + 1] = pos

    trie.save(str(out_dir / 'features.marisa'))
    np.save(out_dir / 'row_ptr.npy', row_ptr)
    np.save(out_dir / 'col_idx.npy', col_idx)
    np.save(out_dir / 'w_val.npy', w_val)
    with open(out_dir / 'meta.pickle', 'wb') as f:
        pickle.dump({'classes': classes, 'tagdict': dict(tagger.tagdict)}, f)

    return {'features': len(kept), 'nnz': nnz, 'classes': len(classes)}
