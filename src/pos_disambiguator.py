"""
POS-Aware Disambiguator za vraćanje dijakritika.

Ključna inovacija ovog sistema: koristi POS tag dodeljen tekstu
BEZ dijakritika da suzi listu kandidata iz srLex rečnika.

Arhitektura:
    1. Tokenizuj tekst bez dijakritika
    2. POS-taguj tokenizovani tekst (tager radi i bez dijakritika)
    3. Za svaki token, generiši kandidate iz srLex-a
    4. Filtriraj kandidate po POS/MSD tagu
    5. Ako ostane jedan kandidat → rešeno
    6. Ako ostane više → koristi frekvenciju / n-gram / neural

"""

import os
import sys
import re
import time
from collections import defaultdict

from diacritics import strip_diacritics, needs_restoration, has_diacritics
from candidate_generator import CandidateGenerator
from oov_handler import apply_heuristics, generate_diacritic_variants
from ngram_model import NgramModel


class POSDisambiguator:
    """
    Vraća dijakritike u tekst koristeći POS tagove i srLex rečnik.

    Metode restauracije (od najjednostavnije do najsloženije):
        1. frequency_baseline  -- uvek biraj najčešći oblik
        2. pos_aware           -- koristi POS tag za filtriranje
        3. msd_aware           -- koristi puni MSD tag (precizniji)
    """

    def __init__(self, candidate_gen, tagger=None, ngram_model=None):
        """
        Args:
            candidate_gen: CandidateGenerator instanca (srLex indeks)
            tagger: POS tager koji prihvata listu reči i vraća [(word, tag)]
                    Ako je None, koristi se samo frequency baseline.
            ngram_model: NgramModel instanca za kontekstualno bodovanje
        """
        self.cg = candidate_gen
        self.tagger = tagger
        self.ngram = ngram_model

        # Statistike
        self._stats = defaultdict(int)

    def restore_text(self, text, method='pos_aware'):
        """
        Vraća dijakritike u tekst.

        Args:
            text: tekst bez dijakritika
            method: 'frequency', 'pos_aware', ili 'msd_aware'

        Returns:
            Tekst sa vraćenim dijakritikama
        """
        # Tokenizuj -- čuvaj i razmake/interpunkciju
        tokens = self._tokenize(text)

        if method == 'frequency':
            restored = self._restore_frequency(tokens)
        elif method == 'pos_aware':
            restored = self._restore_pos_aware(tokens)
        elif method == 'msd_aware':
            restored = self._restore_pos_aware_full(tokens, use_msd=True)
        elif method == 'pos_ngram':
            restored = self._restore_pos_ngram(tokens)
        else:
            raise ValueError(f"Nepoznat metod: {method}")

        return ''.join(restored)

    def restore_tokens(self, words, method='pos_aware'):
        """
        Vraća dijakritike za listu reči (bez razmaka).

        Args:
            words: lista reči bez dijakritika
            method: metod restauracije

        Returns:
            Lista reči sa vraćenim dijakritikama
        """
        if method == 'frequency':
            return [self._restore_word_frequency(w) for w in words]
        elif method in ('pos_aware', 'msd_aware'):
            return self._restore_words_pos(words, method)
        else:
            raise ValueError(f"Nepoznat metod: {method}")

    # ================================================================
    # METOD 1: Frequency Baseline
    # ================================================================

    def _restore_frequency(self, tokens):
        """Najjednostavniji pristup: uvek biraj najfrekventniji oblik."""
        result = []
        for token, is_word in tokens:
            if is_word and needs_restoration(token):
                restored = self._restore_word_frequency(token)
                result.append(restored)
            else:
                result.append(token)
        return result

    def _restore_word_frequency(self, word):
        """Vrati najfrekventniju varijantu reči iz srLex-a."""
        self._stats['total_words'] += 1

        if not needs_restoration(word):
            self._stats['no_restoration_needed'] += 1
            return word

        self._stats['needs_restoration'] += 1

        unique = self.cg.get_unique_words(word)
        if not unique:
            self._stats['not_in_dictionary'] += 1
            # OOV: probaj dopunski rečnik (srWaC) prvo
            supp = self.cg.get_supplement(word)
            if supp:
                self._stats['oov_supplement_resolved'] += 1
                return self._match_case(word, supp[0])

            # OOV fallback: primeni heuristike (dž pravila + sufiksna pravila)
            word_set = self.cg._word_set if hasattr(self.cg, '_word_set') else None
            restored = apply_heuristics(word, word_set=word_set)
            if restored != word:
                self._stats['oov_heuristic_resolved'] += 1
            return restored

        if len(unique) == 1:
            self._stats['unambiguous'] += 1
            return list(unique.keys())[0]

        self._stats['ambiguous_frequency_resolved'] += 1
        # Biraj najfrekventniji
        best = max(unique, key=unique.get)

        # Čuvaj casing
        return self._match_case(word, best)

    # ================================================================
    # METOD 2: POS-Aware
    # ================================================================

    def _restore_pos_aware(self, tokens):
        """Koristi POS tag za disambiguation."""
        return self._restore_pos_aware_full(tokens, use_msd=False)

    def _restore_pos_aware_full(self, tokens, use_msd=False):
        """Koristi POS ili MSD tag za disambiguation."""
        # Izvuci samo reči za POS tagovanje
        word_tokens = [(i, token) for i, (token, is_word) in enumerate(tokens)
                       if is_word]

        if not word_tokens or self.tagger is None:
            return self._restore_frequency(tokens)

        # POS taguj
        words = [token for _, token in word_tokens]
        tagged = self.tagger.tag(words)

        # Napravi mapu
        pos_map = {}
        word_idx_map = {}
        for j, ((idx, _), (_, tag)) in enumerate(zip(word_tokens, tagged)):
            pos_map[idx] = tag
            word_idx_map[idx] = j

        # Restauriraj sa POS/MSD informacijom + n-gram kontekst za sz parove
        result = []
        for i, (token, is_word) in enumerate(tokens):
            if is_word and needs_restoration(token):
                tag = pos_map.get(i, '')
                j = word_idx_map.get(i, -1)
                prev_word = words[j-1] if j > 0 else '<S>'
                next_word = words[j+1] if j < len(words)-1 else None

                if use_msd:
                    restored = self._restore_word_msd(token, tag)
                else:
                    restored = self._restore_word_pos(token, tag,
                                                      prev_word=prev_word,
                                                      next_word=next_word)
                result.append(restored)
            else:
                result.append(token)

        return result

    def _restore_words_pos(self, words, method='pos_aware'):
        """Restauriraj listu reči sa POS tagovima."""
        if self.tagger is None:
            return [self._restore_word_frequency(w) for w in words]

        tagged = self.tagger.tag(words)
        result = []

        for word, (_, tag) in zip(words, tagged):
            if needs_restoration(word):
                if method == 'msd_aware':
                    restored = self._restore_word_msd(word, tag)
                else:
                    restored = self._restore_word_pos(word, tag)
                result.append(restored)
            else:
                result.append(word)

        return result

    # Frekventni š/ž parovi gde n-gram kontekst bolje razrešava od POS-a
    _SZ_NGRAM_WORDS = {'sto', 'vise', 'sirom', 'posto', 'kaze', 'trazi',
                        'slazu', 'pokazu', 'brze', 'bas', 'mars', 'tesko'}

    def _restore_word_pos(self, word, pos_tag, prev_word=None, next_word=None):
        """
        Vrati dijakritike koristeći POS tag za filtriranje.

        Logika:
        1. Generiši sve kandidate iz srLex-a
        2. Filtriraj po POS tagu (prvo slovo MSD-a)
        3. Ako ostane jedan → vrati ga
        4. Ako ostane više → č/ć razrešavanje, pa n-gram za š/ž, pa frequency
        5. Ako nema nijednog sa tim POS-om → n-gram ili frequency fallback
        """
        self._stats['total_words'] += 1

        if not needs_restoration(word):
            self._stats['no_restoration_needed'] += 1
            return word

        self._stats['needs_restoration'] += 1

        # Svi kandidati
        all_candidates = self.cg.get_unique_words(word)
        if not all_candidates:
            self._stats['not_in_dictionary'] += 1
            return self._restore_oov(word, pos_tag)

        # Vlastita imena (tager kaže Np, reč kapitalizovana) imaju
        # hijerarhiju dokaza; mora PRE unambiguous prečice, jer je
        # pogrešan common-noun kandidat često jedini (bus, sirak):
        #   1. srLex Np kandidat sa freq > 0 → normalan tok niže.
        #   2. srWaC Np tabela → korpusni dokaz; jači od freq-0
        #      srLex odrednica (Žože/Npmsg-0 gubi od Žoze/251).
        #   3. srLex Np kandidat sa freq 0 → kurirana odrednica je
        #      jedini dokaz (retka prezimena: Grdešić, Ćakuli).
        #   4. Ništa od toga → reč se NE dira: nepoznato ime ne sme
        #      da se koerzira u zajedničku imenicu (Pasi ≠ paši).
        if pos_tag and pos_tag[:2] == 'Np' and word[:1].isupper():
            cands = self.cg.get_candidates(word)
            np_cands = [(w, freq) for w, _, msd, freq in cands
                        if msd.startswith('Np')]
            if not any(freq > 0 for _, freq in np_cands):
                aug = self.cg.get_augment_np(word)
                if aug:
                    self._stats['np_augment_resolved'] += 1
                    return aug[0]
                # freq-0 Np odrednica je upotrebljiva samo kad je JEDINO
                # čitanje forme (Grdešić); ako postoji i common čitanje
                # (crna/Črna), pusti normalan tok gde guard štiti.
                if np_cands and all(msd.startswith('Np') for _, _, msd, _ in cands):
                    self._stats['np_zero_freq_resolved'] += 1
                    return self._match_case(word, np_cands[0][0])
                if not np_cands:
                    self._stats['np_left_untouched'] += 1
                    return word

        if len(all_candidates) == 1:
            self._stats['unambiguous'] += 1
            return self._match_case(word, list(all_candidates.keys())[0])

        # POS filtriranje
        if pos_tag:
            pos_char = pos_tag[0].upper()
            pos_filtered = self.cg.get_candidates_by_pos(word, pos_char)

            # Ako POS=V (glagol), uključi i Q (rečca) kandidate
            # jer "neće" je V u srWaC ali Q u srLex-u
            if pos_char == 'V':
                q_candidates = self.cg.get_candidates_by_pos(word, 'Q')
                if q_candidates:
                    pos_filtered = pos_filtered + q_candidates

            if pos_filtered:
                # Agregiraj po reči
                pos_words = {}
                for orig, lemma, msd, freq in pos_filtered:
                    if orig not in pos_words or freq > pos_words[orig]:
                        pos_words[orig] = freq

                unique_pos_words = set(pos_words.keys())

                if len(unique_pos_words) == 1:
                    resolved_word = list(unique_pos_words)[0]
                    # Za frekventne š/ž parove, proveri n-gram
                    # jer POS tager može da pogreši na ovim rečima
                    if (self.ngram and word.lower() in self._SZ_NGRAM_WORDS
                            and prev_word and len(all_candidates) > 1):
                        ngram_best = self.ngram.best_candidate(
                            list(all_candidates.keys()), prev_word, next_word
                        )
                        if ngram_best and ngram_best != resolved_word:
                            # N-gram se ne slaže sa POS-om -- proveri ko je jači
                            # Koristi n-gram samo ako je mnogo siguraniji
                            scores = self.ngram.score_candidates(
                                list(all_candidates.keys()), prev_word, next_word
                            )
                            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                            if len(sorted_scores) >= 2:
                                diff = sorted_scores[0][1] - sorted_scores[1][1]
                                if diff > 1.0:  # n-gram je siguran
                                    self._stats['sz_ngram_override'] += 1
                                    return self._match_case(word, sorted_scores[0][0])
                    self._stats['pos_resolved'] += 1
                    resolved_word = self._freq_guard(resolved_word, all_candidates)
                    return self._match_case(word, resolved_word)
                else:
                    # Više kandidata sa istim POS-om
                    # Probaj č/ć razrešavanje za poznate obrasce
                    cc_resolved = self._resolve_cc(word, pos_words)
                    if cc_resolved:
                        self._stats['cc_resolved'] += 1
                        return self._match_case(word, cc_resolved)
                    # Fallback na najfrekventniji
                    self._stats['pos_narrowed'] += 1
                    best = max(pos_words, key=pos_words.get)
                    best = self._freq_guard(best, all_candidates)
                    return self._match_case(word, best)

        # Fallback: za frekventne š/ž parove probaj n-gram kontekst
        if self.ngram and word.lower() in self._SZ_NGRAM_WORDS and prev_word:
            best = self.ngram.best_candidate(
                list(all_candidates.keys()), prev_word, next_word
            )
            if best:
                self._stats['sz_ngram_resolved'] += 1
                return self._match_case(word, best)

        # Fallback: frequency
        self._stats['pos_fallback'] += 1
        best = max(all_candidates, key=all_candidates.get)
        return self._match_case(word, best)

    # Odnos frekvencija iznad kog se ne veruje POS filtriranju:
    # ako je POS-izabrani kandidat FREQ_GUARD_RATIO puta ređi od globalno
    # najfrekventnijeg, tag je verovatnije pogrešan nego da je u pitanju
    # stvarno ređa varijanta (npr. tager na ogoljenom tekstu označi "sto"
    # kao broj, pa POS filter izabere "sto" umesto 16x češćeg "što";
    # ili freq-0 odrednica "Črna" pobedi "crna" sa freq 13,656).
    FREQ_GUARD_RATIO = 10

    def _freq_guard(self, chosen, all_candidates):
        """Frequency dominance guard preko POS-filtriranog izbora."""
        global_best = max(all_candidates, key=all_candidates.get)
        if chosen == global_best:
            return chosen
        chosen_freq = max(all_candidates.get(chosen, 0), 1)
        if all_candidates[global_best] >= self.FREQ_GUARD_RATIO * chosen_freq:
            self._stats['freq_guard_override'] += 1
            return global_best
        return chosen

    def _restore_oov(self, word, pos_tag=None):
        """
        OOV lanac za reč koje nema u srLex-u:
        1. srWaC dopunski rečnik (dominantna dijakritizovana varijanta)
        2. srWaC Np tabela za vlastita imena (kad tager kaže Np)
        3. Rastavljanje na delove oko crtice (64-godisnji → 64-godišnji)
        4. Heuristike: č/ć obrasci, sufiksna pravila, dž/đ digrafi
        """
        # Kratki all-caps tokeni su akronimi (VRS, OEBS): lowercase lookup
        # u dopunskom rečniku bi ih pogrešno dijakritizovao (VRS → VRŠ)
        if word.isupper() and 2 <= len(word) <= 5:
            self._stats['oov_acronym_untouched'] += 1
            return word

        supp = self.cg.get_supplement(word)
        if supp:
            self._stats['oov_supplement_resolved'] += 1
            return self._match_case(word, supp[0])

        if pos_tag and pos_tag[:2] == 'Np' and word[:1].isupper():
            aug = self.cg.get_augment_np(word)
            if aug:
                self._stats['np_augment_resolved'] += 1
                return aug[0]

        if '-' in word and len(word) > 3:
            parts = word.split('-')
            restored_parts = []
            changed = False
            for p in parts:
                if p and needs_restoration(p):
                    cands = self.cg.get_unique_words(p)
                    if cands:
                        best = max(cands, key=cands.get)
                        restored_parts.append(self._match_case(p, best))
                        changed = changed or best.lower() != p.lower()
                        continue
                    supp_p = self.cg.get_supplement(p)
                    if supp_p:
                        restored_parts.append(self._match_case(p, supp_p[0]))
                        changed = True
                        continue
                    heur = apply_heuristics(p)
                    changed = changed or heur != p
                    restored_parts.append(heur)
                    continue
                restored_parts.append(p)
            if changed:
                self._stats['oov_hyphen_resolved'] += 1
                return '-'.join(restored_parts)

        word_set = getattr(self.cg, '_word_set', None)
        restored = apply_heuristics(word, word_set=word_set)
        if restored != word:
            self._stats['oov_heuristic_resolved'] += 1
        return restored

    # ================================================================
    # METOD 3: MSD-Aware (puni MSD tag)
    # ================================================================

    def _restore_word_msd(self, word, msd_tag):
        """
        Koristi puni MSD tag za preciznije filtriranje.

        Primer: za "reci" sa MSD="Vmn" (infinitiv), bira "reći"
                za "reci" sa MSD="Ncfsg" (genitiv), bira "reči"
        """
        self._stats['total_words'] += 1

        if not needs_restoration(word):
            self._stats['no_restoration_needed'] += 1
            return word

        self._stats['needs_restoration'] += 1

        all_candidates = self.cg.get_unique_words(word)
        if not all_candidates:
            self._stats['not_in_dictionary'] += 1
            return word

        if len(all_candidates) == 1:
            self._stats['unambiguous'] += 1
            return self._match_case(word, list(all_candidates.keys())[0])

        if msd_tag:
            # Probaj exact MSD match
            msd_filtered = self.cg.get_candidates_by_msd(word, msd_tag)
            if msd_filtered:
                msd_words = {}
                for orig, lemma, msd, freq in msd_filtered:
                    if orig not in msd_words or freq > msd_words[orig]:
                        msd_words[orig] = freq

                if len(set(msd_words.keys())) == 1:
                    self._stats['msd_resolved'] += 1
                    return self._match_case(word, list(msd_words.keys())[0])
                else:
                    self._stats['msd_narrowed'] += 1
                    best = max(msd_words, key=msd_words.get)
                    return self._match_case(word, best)

            # Fallback na POS
            return self._restore_word_pos(word, msd_tag)

        # Fallback: frequency
        self._stats['pos_fallback'] += 1
        best = max(all_candidates, key=all_candidates.get)
        return self._match_case(word, best)

    # ================================================================
    # METOD 4: POS + N-gram (hibridni)
    # ================================================================

    def _restore_pos_ngram(self, tokens):
        """
        Kombinuje POS filtriranje sa n-gram kontekstom.

        Korak 1: POS taguj tekst
        Korak 2: Za svaku reč, generiši kandidate iz srLex-a
        Korak 3: Filtriraj po POS tagu
        Korak 4: Ako ostane >1 kandidat, koristi n-gram za finalni izbor
        """
        word_tokens = [(i, token) for i, (token, is_word) in enumerate(tokens)
                       if is_word]

        if not word_tokens or self.tagger is None:
            return self._restore_frequency(tokens)

        # POS taguj
        words = [token for _, token in word_tokens]
        tagged = self.tagger.tag(words)

        # Napravi mapu
        pos_map = {}
        word_idx_map = {}  # pozicija u tokens → pozicija u words listi
        for j, ((idx, _), (_, tag)) in enumerate(zip(word_tokens, tagged)):
            pos_map[idx] = tag
            word_idx_map[idx] = j

        # Restauriraj
        result = []
        for i, (token, is_word) in enumerate(tokens):
            if is_word and needs_restoration(token):
                pos_tag = pos_map.get(i, '')
                j = word_idx_map.get(i, -1)

                # Prethodna i sledeća reč za n-gram kontekst
                prev_word = words[j-1] if j > 0 else '<S>'
                next_word = words[j+1] if j < len(words)-1 else None

                restored = self._restore_word_pos_ngram(
                    token, pos_tag, prev_word, next_word
                )
                result.append(restored)
            else:
                result.append(token)

        return result

    def _restore_word_pos_ngram(self, word, pos_tag, prev_word, next_word):
        """
        POS filtriranje + n-gram SAMO kao tiebreaker.

        Logika:
        1. Ako jednoznačno u srLex-u → vrati odmah
        2. Ako POS sužava na 1 kandidat → vrati odmah (POS pobedio)
        3. Ako POS sužava na >1 kandidat → n-gram bira među njima
        4. Ako POS nema kandidata → n-gram bira među SVIM kandidatima
        5. Fallback: frequency

        N-gram NIKADA ne overrideuje POS odluku -- samo bira unutar
        POS-filtriranog skupa.
        """
        self._stats['total_words'] += 1

        if not needs_restoration(word):
            self._stats['no_restoration_needed'] += 1
            return word

        self._stats['needs_restoration'] += 1

        all_candidates = self.cg.get_unique_words(word)
        if not all_candidates:
            self._stats['not_in_dictionary'] += 1
            return apply_heuristics(word)

        if len(all_candidates) == 1:
            self._stats['unambiguous'] += 1
            return self._match_case(word, list(all_candidates.keys())[0])

        # Korak 1: POS filtriranje
        pos_candidates = None
        if pos_tag:
            pos_char = pos_tag[0].upper()
            pos_filtered = self.cg.get_candidates_by_pos(word, pos_char)
            if pos_filtered:
                pos_words = {}
                for orig, lemma, msd, freq in pos_filtered:
                    if orig not in pos_words or freq > pos_words[orig]:
                        pos_words[orig] = freq

                if len(pos_words) == 1:
                    # POS je razrešio potpuno -- gotovo
                    self._stats['pos_resolved'] += 1
                    return self._match_case(word, list(pos_words.keys())[0])

                # POS sužava ali ostaje >1 kandidat
                pos_candidates = pos_words

        # Korak 2: N-gram tiebreaker
        # Koristi SAMO ako POS je suzio ali ostalo >1 kandidat
        # Ako POS nije pomogao uopste, koristi frequency (sigurnije)
        if not pos_candidates:
            # POS nije imao kandidata -- frequency fallback
            self._stats['pos_fallback'] += 1
            best = max(all_candidates, key=all_candidates.get)
            return self._match_case(word, best)

        remaining = pos_candidates

        if len(remaining) > 1 and self.ngram:
            best = self.ngram.best_candidate(
                list(remaining.keys()), prev_word, next_word
            )
            if best:
                if pos_candidates:
                    self._stats['pos_narrowed_ngram_resolved'] += 1
                else:
                    self._stats['ngram_resolved'] += 1
                return self._match_case(word, best)

        # Fallback: frequency iz preostalih kandidata
        self._stats['pos_fallback'] += 1
        best = max(remaining, key=remaining.get)
        return self._match_case(word, best)

    # ================================================================
    # č/ć RAZREŠAVANJE
    # ================================================================

    # Poznati obrasci gde je ć ispravno (negacija, futur, infinitiv na -ći)
    _CC_PREFER_C_ACUTE = {
        'neće', 'neću', 'nećeš', 'nećemo', 'nećete',
        'biće', 'dobiće', 'moraće', 'imaće', 'radiće',
        'videće', 'trebaće', 'moći', 'doći', 'otići',
        'proći', 'izaći', 'ući', 'naći', 'poći', 'reći',
    }

    def _resolve_cc(self, word, candidates):
        """
        Razrešava č/ć konflikt kad POS ne pomaže.
        Vraća kandidata ili None.
        """
        lower = word.lower()

        # Proveri da li je neki kandidat u listi poznatih ć oblika
        for cand in candidates:
            if cand.lower() in self._CC_PREFER_C_ACUTE:
                return cand

        # Obrazac: reči na -će/-ću/-ćeš/-ćemo/-ćete su skoro uvek futur
        # (neće, biće, dobiće, trebaće...)
        for cand in candidates:
            cl = cand.lower()
            if cl.endswith(('će', 'ću', 'ćeš', 'ćemo', 'ćete')):
                # Proveri da li postoji i č varijanta
                has_c = any(c.lower().endswith(('če', 'ču', 'češ', 'čemo', 'čete'))
                           for c in candidates if c != cand)
                if has_c:
                    return cand  # Preferiraj ć za futurske oblike

        return None

    # ================================================================
    # POMOĆNE FUNKCIJE
    # ================================================================

    def _tokenize(self, text):
        """
        Tokenizuje tekst čuvajući razmake i interpunkciju.

        Vraća listu (token, is_word) gde is_word=True za reči,
        False za razmake/interpunkciju.
        """
        tokens = []
        pattern = re.compile(r'(\w+|[^\w])', re.UNICODE)

        for match in pattern.finditer(text):
            token = match.group()
            is_word = bool(re.match(r'\w+$', token, re.UNICODE))
            tokens.append((token, is_word))

        return tokens

    def _match_case(self, original, replacement):
        """
        Prilagodi casing replacement-a da odgovara originalu.

        "Sto" + "što" → "Što"
        "STO" + "što" → "ŠTO"
        "sto" + "što" → "što"
        "SETimes" + "setimes" → "SETimes" (čuvaj original ako je mešan)
        """
        if not original or not replacement:
            return replacement

        if original.isupper() and len(original) > 1:
            return replacement.upper()
        elif original.islower():
            return replacement.lower()
        elif original[0].isupper() and original[1:].islower():
            # Samo prvo slovo veliko (Sto → Što)
            return replacement[0].upper() + replacement[1:].lower()
        else:
            # Mešani casing (SETimes, BiH, SAD) -- čuvaj original
            return original

    def get_stats(self):
        """Vraća statistike restauracije."""
        return dict(self._stats)

    def reset_stats(self):
        """Resetuje statistike."""
        self._stats = defaultdict(int)


# ================================================================
# TEST
# ================================================================

if __name__ == '__main__':
    srlex_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'POS-Aware-Stemmer', 'data', 'srLex_v1.3.gz'
    )

    print("Učitavam kandidat-generator...")
    cg = CandidateGenerator(srlex_path)

    # Bez tagera (samo frequency baseline)
    disamb = POSDisambiguator(cg, tagger=None)

    test_sentences = [
        "Zivim u Nisu i radim na fakultetu",
        "Sto je to sto treba da uradimo",
        "Nas profesor je rekao da cemo dobiti vise bodova",
        "Zelim da ti kazem nesto vazno",
        "Cesto idem u skolu peske",
        "Reci mi sta zelis",
        "On je zeleo da dodje kod nas",
        "Djak je bio dobar u skoli",
    ]

    print("\n" + "=" * 60)
    print("FREQUENCY BASELINE (bez POS tagera)")
    print("=" * 60)

    for sent in test_sentences:
        restored = disamb.restore_text(sent, method='frequency')
        print(f"  IN:  {sent}")
        print(f"  OUT: {restored}")
        print()

    stats = disamb.get_stats()
    print("Statistike:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
