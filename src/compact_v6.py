"""
Compact (memory-mapped) backend za v6 modul restauracije dijakritika.

CompactCandidateGenerator nasleđuje CandidateGenerator i overriduje SAMO
pristup skladištu (marisa trie umesto Python dict-ova); sva logika
odlučivanja (POSDisambiguator: Np hijerarhija, dominance guard, č/ć,
OOV lanac) je DOSLOVNO ISTI kod kao u referentnoj implementaciji, čime
je identičnost izlaza obezbeđena po konstrukciji.

Model fajlovi (u compact_lexicon direktorijumu):
  v6_words.marisa   Trie nad rečima kandidatima (originalno slovo)
  v6_ascii.marisa   BytesTrie ascii_key -> packed [(word_id, msd_id, freq)]
                    u ISTOM redosledu kao referentni indeks
  v6_meta.pickle    {'msds': [...]}
  srwac_supplement_v2.json, srwac_augment_np.json  (kopije, mali JSON-i)

"""
import json
import pickle
import shutil
import struct
import sys
from pathlib import Path

import marisa_trie

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator
from diacritics import needs_restoration, strip_diacritics


class CompactCandidateGenerator(CandidateGenerator):
    """Isti interfejs kao CandidateGenerator, skladište u marisa trie."""

    def __init__(self, model_dir):
        # NE pozivamo super().__init__ (ne učitavamo srLex u RAM)
        model_dir = Path(model_dir)
        self._v6words = marisa_trie.Trie()
        self._v6words.load(str(model_dir / 'v6_words.marisa'))
        self._v6ascii = marisa_trie.BytesTrie()
        self._v6ascii.load(str(model_dir / 'v6_ascii.marisa'))
        with open(model_dir / 'v6_meta.pickle', 'rb') as f:
            self._msds = pickle.load(f)['msds']
        self._supplement = {}
        self._augment_np = {}
        supp = model_dir / 'srwac_supplement_v2.json'
        aug = model_dir / 'srwac_augment_np.json'
        if supp.exists():
            with open(supp, 'r', encoding='utf-8') as f:
                self._supplement = {k: (v['word'], v['freq'])
                                    for k, v in json.load(f).items()}
        if aug.exists():
            with open(aug, 'r', encoding='utf-8') as f:
                self._augment_np = {k: (v['word'], v['freq'])
                                    for k, v in json.load(f).items()}
        # word_set interfejs za edit-distance heuristike: trie podržava `in`
        self._word_set = self._v6words
        self._loaded = True

    def get_candidates(self, word):
        ascii_key = strip_diacritics(word).lower()
        vals = self._v6ascii.get(ascii_key)
        if not vals:
            return []
        blob = vals[0]
        n = struct.unpack_from('<H', blob, 0)[0]
        out = []
        off = 2
        for _ in range(n):
            wid, mid, fr = struct.unpack_from('<IHI', blob, off)
            off += 10
            out.append((self._v6words.restore_key(wid), None,
                        self._msds[mid], fr))
        return out


class CompactV6Restorer:
    """Kompaktni ekvivalent V6Restorer-a: ista restore() semantika."""

    def __init__(self, model_dir):
        self.cg = CompactCandidateGenerator(model_dir)
        self._disamb = POSDisambiguator(self.cg, tagger=None)

    def restore(self, word, msd=''):
        if not needs_restoration(word):
            return word
        return self._disamb._restore_word_pos(word, msd)


def build_v6_tables(cg, out_dir, data_dir=None):
    """
    Pakuje CandidateGenerator._index u v6_* trie fajlove, čuvajući
    redosled kandidata TAČNO kao u referentnom indeksu. Kopira i male
    JSON tabele u model direktorijum.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # samo ključevi koje restore() uopšte može da upita
    keys = [k for k in cg._index if needs_restoration(k)]

    words = set()
    msds = set()
    for k in keys:
        for w, _l, m, _f in cg._index[k]:
            words.add(w)
            msds.add(m)
    words_trie = marisa_trie.Trie(set(cg._word_set) | words)
    msds = sorted(msds)
    msd2id = {m: i for i, m in enumerate(msds)}

    def pack(entries):
        parts = [struct.pack('<H', min(len(entries), 0xFFFF))]
        for w, _l, m, fr in entries[:0xFFFF]:
            parts.append(struct.pack('<IHI', words_trie[w], msd2id[m],
                                     min(fr, 0xFFFFFFFF)))
        return b''.join(parts)

    ascii_trie = marisa_trie.BytesTrie((k, pack(cg._index[k])) for k in keys)

    words_trie.save(str(out_dir / 'v6_words.marisa'))
    ascii_trie.save(str(out_dir / 'v6_ascii.marisa'))
    with open(out_dir / 'v6_meta.pickle', 'wb') as f:
        pickle.dump({'msds': msds}, f)

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    for name in ('srwac_supplement_v2.json', 'srwac_augment_np.json'):
        src = Path(data_dir) / name
        if src.exists():
            shutil.copy(src, out_dir / name)

    return {'ascii_keys': len(keys), 'words': len(words_trie), 'msds': len(msds)}
