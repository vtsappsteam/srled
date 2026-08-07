"""
Kandidat-generator za vraćanje dijakritika.

Koristi srLex 1.3 morfološki rečnik za generisanje mogućih
dijakritičkih oblika za svaku reč bez dijakritika.

Struktura indeksa:
    ascii_lower_form → [
        (original_word, lemma, msd, frequency),
        ...
    ]

Primer:
    "sto" → [
        ("što", "što", "Cs", 1055237),
        ("sto", "sto", "Mdc", 66183),
    ]

    "reci" → [
        ("reći", "reći", "Vmn", 93085),
        ("reči", "reč", "Ncfsg", 25305),
        ("reci", "reći", "Vmm2s", 9484),
    ]

"""

import gzip
import os
import json
import time
from collections import defaultdict
from diacritics import strip_diacritics, has_diacritics, DIACRITIC_CHARS


class CandidateGenerator:
    """
    Generiše kandidate za dijakritičku restauraciju koristeći srLex.

    Za svaku reč bez dijakritika, vraća listu mogućih oblika sa
    dijakritikama, zajedno sa POS tagom i frekvencijom iz korpusa.
    """

    def __init__(self, srlex_path=None, supplement_path=None, augment_np_path=None,
                 extra_dict_path=None):
        self._index = {}          # ascii_lower → [(word, lemma, msd, freq)]
        self._word_set = set()    # set svih reči iz srLex (sa dijakritikama)
        self._supplement = {}     # ascii_lower → (word, freq) iz srWaC dopune
        self._augment_np = {}     # ascii_lower → (Word, freq) za vlastita imena
        self._loaded = False

        if srlex_path:
            self.load(srlex_path)
        if extra_dict_path:
            self.load_extra_dict(extra_dict_path)
        if supplement_path:
            self.load_supplement(supplement_path)
        if augment_np_path:
            self.load_augment_np(augment_np_path)

    def load_extra_dict(self, extra_dict_path):
        """
        Dodaje expanded supplement (gold-korpusne word→[(lemma,msd,freq)]
        odrednice van srLex-a) u glavni indeks, da modul za restauraciju
        pokriva isti rečnik kao lematizator.
        """
        import json
        with open(extra_dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        added = 0
        for word, entries in data.items():
            ascii_key = strip_diacritics(word).lower()
            for lemma, msd, freq in entries:
                self._index.setdefault(ascii_key, []).append((word, lemma, msd, freq))
                added += 1
            self._word_set.add(word)
        # održi sortiranost po frekvenciji (opadajuće)
        for key in {strip_diacritics(w).lower() for w in data}:
            if key in self._index:
                self._index[key].sort(key=lambda x: -x[3])
        print(f"  Extra dict: +{added:,} odrednica u indeks")

    def load(self, srlex_path):
        """
        Učitava srLex i gradi indeks za brzu pretragu.

        Format srLex:
            word  lemma  MSD  features  UPOS  UD_features  freq  rel_freq
        """
        print(f"Učitavam srLex: {srlex_path}")
        start = time.time()

        index = defaultdict(list)
        word_set = set()

        open_func = gzip.open if srlex_path.endswith('.gz') else open

        with open_func(srlex_path, 'rt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 7:
                    continue

                word = parts[0]
                lemma = parts[1]
                msd = parts[2]

                try:
                    freq = int(parts[6])
                except (ValueError, IndexError):
                    freq = 0

                word_set.add(word)

                # Indeksiraj po ASCII lowercase obliku
                ascii_key = strip_diacritics(word).lower()
                index[ascii_key].append((word, lemma, msd, freq))

        # Sortiraj kandidate po frekvenciji (opadajuće)
        for key in index:
            index[key].sort(key=lambda x: -x[3])

        self._index = dict(index)
        self._word_set = word_set
        self._loaded = True

        elapsed = time.time() - start
        print(f"  Učitano {len(self._word_set):,} oblika, "
              f"{len(self._index):,} ASCII ključeva za {elapsed:.1f}s")

        # Statistike
        ambiguous = sum(1 for v in self._index.values()
                       if len(set(w for w, _, _, _ in v)) > 1)
        print(f"  Višeznačnih ASCII oblika: {ambiguous:,}")

    def get_candidates(self, word):
        """
        Vraća listu kandidata za datu reč bez dijakritika.

        Args:
            word: reč bez dijakritika (npr. "sto", "reci", "nas")

        Returns:
            Lista tuple-ova (original_word, lemma, msd, frequency)
            sortirana po frekvenciji opadajuće.
            Prazna lista ako reč nije pronađena u rečniku.
        """
        ascii_key = strip_diacritics(word).lower()
        candidates = self._index.get(ascii_key, [])

        # Filtriramo da kandidati imaju isti casing pattern kao ulaz
        # (ako je ulaz "Sto", vraćamo "Što" i "Sto", ne "što" i "sto")
        if word and word[0].isupper():
            # Ulaz počinje velikim slovom
            result = []
            for orig, lemma, msd, freq in candidates:
                # Prihvati i uppercase i lowercase varijante
                result.append((orig, lemma, msd, freq))
            return result

        return candidates

    def get_unique_words(self, word):
        """
        Vraća samo jedinstvene reči (bez duplikata po MSD tagu).

        Args:
            word: reč bez dijakritika

        Returns:
            Dict: {original_word: max_frequency}
        """
        candidates = self.get_candidates(word)
        word_freqs = {}
        for orig, lemma, msd, freq in candidates:
            if orig not in word_freqs or freq > word_freqs[orig]:
                word_freqs[orig] = freq
        return word_freqs

    def get_candidates_by_pos(self, word, pos_tag):
        """
        Vraća kandidate filtrirane po POS tagu.

        Ovo je ključna funkcija za POS-aware disambiguation:
        ako tager kaže da je reč imenica (N), vraćamo samo
        kandidate koji su imenice u srLex-u.

        Args:
            word: reč bez dijakritika
            pos_tag: POS tag (prvo slovo MSD taga, npr. 'N', 'V', 'A')

        Returns:
            Lista kandidata koji se slažu sa POS tagom,
            sortirana po frekvenciji.
        """
        candidates = self.get_candidates(word)
        pos_upper = pos_tag[0].upper() if pos_tag else ''

        filtered = [
            (orig, lemma, msd, freq) for orig, lemma, msd, freq in candidates
            if msd and msd[0].upper() == pos_upper
        ]

        return filtered

    def get_candidates_by_msd(self, word, msd_tag):
        """
        Vraća kandidate filtrirane po punom MSD tagu.

        Još preciznija filtracija -- koristi ceo MSD tag
        (npr. Ncmsn za imenicu muški rod jednina nominativ).

        Args:
            word: reč bez dijakritika
            msd_tag: puni MSD tag

        Returns:
            Lista kandidata sa tačnim MSD tagom.
        """
        candidates = self.get_candidates(word)

        # Exact match
        exact = [
            (orig, lemma, msd, freq) for orig, lemma, msd, freq in candidates
            if msd.lower() == msd_tag.lower()
        ]
        if exact:
            return exact

        # Prefix match (ako je MSD tag skraćen)
        prefix = [
            (orig, lemma, msd, freq) for orig, lemma, msd, freq in candidates
            if msd.lower().startswith(msd_tag.lower()) or
               msd_tag.lower().startswith(msd.lower())
        ]
        return prefix

    def is_known_word(self, word):
        """Proverava da li reč (sa dijakritikama) postoji u srLex-u."""
        return word in self._word_set

    def load_supplement(self, supplement_path):
        """Učitava dopunski rečnik iz srWaC-a za OOV reči."""
        import json
        print(f"Učitavam dopunski rečnik: {supplement_path}")
        with open(supplement_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._supplement = {k: (v['word'], v['freq']) for k, v in data.items()}
        print(f"  {len(self._supplement):,} dopunskih oblika")

    def get_supplement(self, word):
        """Vraća dopunski kandidat za OOV reč (iz srWaC-a)."""
        ascii_key = strip_diacritics(word).lower()
        if ascii_key in self._supplement:
            return self._supplement[ascii_key]
        return None

    def load_augment_np(self, augment_np_path):
        """Učitava srWaC tabelu za vlastita imena (velika početna slova)."""
        import json
        print(f"Učitavam Np augmentaciju: {augment_np_path}")
        with open(augment_np_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._augment_np = {k: (v['word'], v['freq']) for k, v in data.items()}
        print(f"  {len(self._augment_np):,} Np oblika")

    def get_augment_np(self, word):
        """
        Vraća dijakritizovanu varijantu vlastitog imena iz srWaC-a,
        ili None. Poziva se samo kad tager kaže Np i reč je kapitalizovana;
        tabela sadrži samo ključeve gde kapitalizovana dijakritizovana
        varijanta dominira nad ASCII varijantom u srWaC-u.
        """
        ascii_key = strip_diacritics(word).lower()
        if ascii_key in self._augment_np:
            return self._augment_np[ascii_key]
        return None

    def most_frequent(self, word):
        """
        Vraća najfrekventniji kandidat (baseline pristup).

        Ovo je najjednostavniji pristup -- uvek biraj
        najčešći oblik iz korpusa.
        """
        candidates = self.get_unique_words(word)
        if not candidates:
            # Probaj dopunski rečnik
            supp = self.get_supplement(word)
            if supp:
                return supp[0]
            return word  # vrati original ako nije ni u dopunskom
        return max(candidates, key=candidates.get)

    def stats(self):
        """Vraća statistike o indeksu."""
        if not self._loaded:
            return {}

        total_keys = len(self._index)
        total_words = len(self._word_set)

        # Klasifikacija po višeznačnosti
        unique_per_key = {}
        for key, candidates in self._index.items():
            unique_words = set(w for w, _, _, _ in candidates)
            unique_per_key[key] = len(unique_words)

        no_change = sum(1 for k, v in self._index.items()
                       if len(set(w for w, _, _, _ in v)) == 1
                       and list(set(w for w, _, _, _ in v))[0] == k)
        unambiguous = sum(1 for v in unique_per_key.values() if v == 1) - no_change
        ambiguous = sum(1 for v in unique_per_key.values() if v > 1)

        return {
            'total_words': total_words,
            'total_ascii_keys': total_keys,
            'no_diacritics_needed': no_change,
            'unambiguous': unambiguous,
            'ambiguous': ambiguous,
            'ambiguous_pct': round(100 * ambiguous / total_keys, 2),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    import sys

    srlex_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'POS-Aware-Stemmer', 'data', 'srLex_v1.3.gz'
    )

    gen = CandidateGenerator(srlex_path)

    # Statistike
    stats = gen.stats()
    print(f"\nStatistike:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test primeri
    test_words = [
        'sto', 'vise', 'nas', 'reci', 'cesto', 'zeleo',
        'zivim', 'skola', 'djak', 'srce', 'kuce',
    ]

    print(f"\nPrimeri:")
    for word in test_words:
        unique = gen.get_unique_words(word)
        if len(unique) > 1:
            cands = ', '.join(f'{w}({f:,})' for w, f in
                            sorted(unique.items(), key=lambda x: -x[1]))
            print(f"  {word:15s} → VIŠEZNAČNO: {cands}")
        elif unique:
            w, f = list(unique.items())[0]
            print(f"  {word:15s} → {w} ({f:,})")
        else:
            print(f"  {word:15s} → NEPOZNATO (nije u srLex-u)")

    # Test POS filtriranja
    print(f"\nPOS filtriranje za 'nas':")
    for pos in ['P', 'A', 'N']:
        cands = gen.get_candidates_by_pos('nas', pos)
        if cands:
            words = set(w for w, _, _, _ in cands)
            print(f"  POS={pos}: {words}")

    print(f"\nPOS filtriranje za 'sto':")
    for pos in ['C', 'M', 'N']:
        cands = gen.get_candidates_by_pos('sto', pos)
        if cands:
            words = set(w for w, _, _, _ in cands)
            print(f"  POS={pos}: {words}")
