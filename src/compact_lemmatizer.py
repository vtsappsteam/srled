"""
Memory-mapped compact lemmatizer backend.

Stores the srLex + expanded-dictionary indexes in marisa tries with
struct-packed per-word records instead of Python hash tables. Produces
predictions identical to the in-memory lemmatizer (the cascade logic and
all orderings are preserved), trading some lookup speed for a ~10x
smaller resident footprint and near-instant memory-mapped loading.

Model format (directory):
  words.marisa      Trie over word forms (word -> id, id -> word)
  entries.marisa    BytesTrie word -> packed record (see _pack_word)
  ascii.marisa      BytesTrie ascii form -> packed (word_id, freq) list
  lemmas.marisa     Trie over lemma strings (lemma -> id)
  meta.pickle       {'msds': [...], 'lemma_corrections': {...}}

"""

import pickle
import struct
from pathlib import Path

import marisa_trie

EKAVIZATION = {
    'također': 'takođe', 'usprkos': 'uprkos', 'unatoč': 'uprkos',
    'željeti': 'želeti', 'vidjeti': 'videti', 'htjeti': 'hteti',
    'razumjeti': 'razumeti', 'smjeti': 'smeti', 'voljeti': 'voleti',
    'živjeti': 'živeti', 'trpjeti': 'trpeti', 'letjeti': 'leteti',
    'gorjeti': 'goreti', 'sjesti': 'sesti',
    'doživiti': 'doživeti', 'voliti': 'voleti', 'resiti': 'rešiti',
}
# Fallback correction table for model directories built before the
# corrections were embedded in meta.pickle (the current builder stores the
# full deployed table, data/lemma_corrections_v2.csv, in the model itself).
LEMMA_CORRECTIONS = {
    'premer': 'premijer', 'mišlenje': 'mišljenje', 'skopje': 'skoplje',
    'k': 'ka', 'tko': 'ko', 'netko': 'neko', 'nitko': 'niko',
    'cel': 'ceo',
    'efekt': 'efekat', 'projekt': 'projekat', 'objekt': 'objekat',
    'ambient': 'ambijent',
    'ekonomist': 'ekonomista', 'terorist': 'terorista',
    'gardist': 'gardista', 'alijas': 'alijansa',
    'nauk': 'nauka',
}
_HTETI_FORMS = {'neće', 'neću', 'nećemo', 'nećete'}
_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


def _normalize_pnpa(tag):
    """Pn*/Pa* -> P* (the lexicon has no Pn/Pa MSDs)."""
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag


class CompactLemmatizer:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self._words = marisa_trie.Trie()
        self._words.load(str(model_dir / 'words.marisa'))
        self._entries = marisa_trie.BytesTrie()
        self._entries.load(str(model_dir / 'entries.marisa'))
        self._ascii = marisa_trie.BytesTrie()
        self._ascii.load(str(model_dir / 'ascii.marisa'))
        self._lemmas = marisa_trie.Trie()
        self._lemmas.load(str(model_dir / 'lemmas.marisa'))
        with open(model_dir / 'meta.pickle', 'rb') as f:
            meta = pickle.load(f)
        self._msds = meta['msds']
        self._msd2id = {m: i for i, m in enumerate(self._msds)}
        # Correction table travels with the model; fall back to the module
        # constant for directories built by older versions.
        self._corrections = meta.get('lemma_corrections', LEMMA_CORRECTIONS)
        # Layer 0: kompaktni v6 modul (deli logiku sa referentnim)
        self._v6 = None
        if (model_dir / 'v6_ascii.marisa').exists():
            import sys as _sys
            _src = str(Path(__file__).parent)
            if _src not in _sys.path:
                _sys.path.insert(0, _src)
            from compact_v6 import CompactV6Restorer
            self._v6 = CompactV6Restorer(model_dir)

    def _lemma(self, lid):
        return self._lemmas.restore_key(lid)

    def _record(self, word):
        vals = self._entries.get(word)
        return vals[0] if vals else None

    @staticmethod
    def _parse(blob):
        """Unpack a word record into (msd_pairs, pos_lists, word_list)."""
        off = 0
        n_msd = struct.unpack_from('<H', blob, off)[0]; off += 2
        msd_pairs = []
        for _ in range(n_msd):
            mid, lid = struct.unpack_from('<HI', blob, off); off += 6
            msd_pairs.append((mid, lid))
        n_pos = struct.unpack_from('<B', blob, off)[0]; off += 1
        pos_lists = {}
        for _ in range(n_pos):
            pc, n = struct.unpack_from('<BH', blob, off); off += 3
            lst = []
            for _ in range(n):
                lid, fr = struct.unpack_from('<II', blob, off); off += 8
                lst.append((lid, fr))
            pos_lists[chr(pc)] = lst
        n_word = struct.unpack_from('<H', blob, off)[0]; off += 2
        word_list = []
        for _ in range(n_word):
            lid, fr = struct.unpack_from('<II', blob, off); off += 8
            word_list.append((lid, fr))
        return msd_pairs, pos_lists, word_list

    def _correct(self, lemma):
        if not lemma:
            return lemma
        lemma = self._corrections.get(lemma, lemma)
        return EKAVIZATION.get(lemma, lemma)

    def lemmatize(self, word, msd=""):
        msd = _normalize_pnpa(msd)
        w = word.lower()
        pos = msd[0] if msd else ""

        if w in _HTETI_FORMS and pos == 'V':
            return 'hteti'

        rec = self._record(w)

        # Layer 0: restauracija dijakritika punim v6 modulom
        if rec is None and self._v6 is not None and not (_DIAC_CHARS & set(word)):
            restored = self._v6.restore(word, msd)
            if restored != word:
                w = restored.lower()
                rec = self._record(w)

        if rec is not None:
            msd_pairs, pos_lists, word_list = self._parse(rec)

            # Layer 1: MSD exact
            if msd:
                mid = self._msd2id.get(msd)
                if mid is not None:
                    for m, lid in msd_pairs:
                        if m == mid:
                            return self._correct(self._lemma(lid))

            # Layer 2: POS lookup
            if pos and pos in pos_lists:
                cands = pos_lists[pos]
                if msd.startswith(('Ap', 'Ag')):
                    for lid, fr in cands:
                        lm = self._lemma(lid)
                        if not lm.endswith(('ti', 'ći', 'ci')):
                            return self._correct(lm)
                for lid, fr in cands:
                    if self._lemma(lid) == w:
                        return w
                return self._correct(self._lemma(cands[0][0]))

            # MSD prefix relaxation
            if msd:
                for plen in range(len(msd) - 1, 0, -1):
                    pref = msd[:plen]
                    for m, lid in msd_pairs:
                        if self._msds[m].startswith(pref):
                            return self._correct(self._lemma(lid))

            # Frequency fallback
            if word_list:
                for lid, fr in word_list:
                    if self._lemma(lid) == w:
                        return w
                return self._correct(self._lemma(word_list[0][0]))

        # Morphological guesser for OOV nouns
        if pos == 'N' and len(w) > 3:
            g = msd[2] if len(msd) > 2 else ''
            n_ = msd[3] if len(msd) > 3 else ''
            c = msd[4] if len(msd) > 4 else ''
            if g == 'm' and n_ == 's' and c == 'g' and w.endswith('a'):
                guess = w[:-1]
                if self._record(guess) is not None:
                    return guess
            if g == 'f' and n_ == 's' and c == 'g' and w.endswith('e'):
                guess = w[:-1] + 'a'
                if self._record(guess) is not None:
                    return guess

        # Identity fallback
        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            return word
        return w


def build_from_lemmatizer(lem, out_dir, lemma_corrections=None):
    """Build the compact model from a loaded AblationLemmatizer, preserving
    entry ordering exactly so that predictions are identical.

    lemma_corrections: the correction table to embed in meta.pickle
    (defaults to the module fallback table)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    words = set(lem._msd_index) | set(lem._pos_index) | set(lem._word_index)
    words_trie = marisa_trie.Trie(words)

    lemmas = set()
    for d in lem._msd_index.values():
        # None lemmas come from zero-frequency srLex entries; the reference
        # lemmatizer skips them with `if lemma:`, so omitting the pair here
        # preserves identical fall-through behaviour.
        lemmas.update(v for v in d.values() if v is not None)
    for pd in lem._pos_index.values():
        for lst in pd.values():
            lemmas.update(l for l, f in lst)
    for lst in lem._word_index.values():
        lemmas.update(l for l, f in lst)
    lemmas_trie = marisa_trie.Trie(lemmas)

    msds = sorted({m for d in lem._msd_index.values() for m in d})
    msd2id = {m: i for i, m in enumerate(msds)}

    def pack_word(w):
        parts = []
        md = [(m, l) for m, l in lem._msd_index.get(w, {}).items() if l is not None]
        parts.append(struct.pack('<H', len(md)))
        for m, l in md:
            parts.append(struct.pack('<HI', msd2id[m], lemmas_trie[l]))
        pd = lem._pos_index.get(w, {})
        parts.append(struct.pack('<B', len(pd)))
        for pc, lst in pd.items():
            parts.append(struct.pack('<BH', ord(pc), len(lst)))
            for l, fr in lst:
                parts.append(struct.pack('<II', lemmas_trie[l], min(fr, 0xFFFFFFFF)))
        wl = lem._word_index.get(w, [])
        parts.append(struct.pack('<H', min(len(wl), 0xFFFF)))
        for l, fr in wl[:0xFFFF]:
            parts.append(struct.pack('<II', lemmas_trie[l], min(fr, 0xFFFFFFFF)))
        return b''.join(parts)

    entries_trie = marisa_trie.BytesTrie((w, pack_word(w)) for w in words)

    def pack_ascii(lst):
        parts = [struct.pack('<H', min(len(lst), 0xFFFF))]
        for cw, fr in lst[:0xFFFF]:
            parts.append(struct.pack('<II', words_trie[cw], min(fr, 0xFFFFFFFF)))
        return b''.join(parts)

    ascii_trie = marisa_trie.BytesTrie(
        (a, pack_ascii(lst)) for a, lst in lem._ascii_index.items() if lst)

    words_trie.save(str(out_dir / 'words.marisa'))
    entries_trie.save(str(out_dir / 'entries.marisa'))
    ascii_trie.save(str(out_dir / 'ascii.marisa'))
    lemmas_trie.save(str(out_dir / 'lemmas.marisa'))
    with open(out_dir / 'meta.pickle', 'wb') as f:
        pickle.dump({'msds': msds,
                     'lemma_corrections': dict(lemma_corrections
                                               if lemma_corrections is not None
                                               else LEMMA_CORRECTIONS)}, f)

    return {'words': len(words), 'lemmas': len(lemmas), 'msds': len(msds)}
