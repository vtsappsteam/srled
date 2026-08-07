#!/usr/bin/env python3
"""
8 Gramatičkih pravila za korekciju srWaC anotacije.

Ova pravila ispravljaju poznate sistematske greške u originalnoj
MULTEXT-East V5 anotaciji srWaC korpusa.

Izvor: ICEST-2023-3734 (Vukotić)

Autor: Nikola Vukotić
Datum: April 2026

PRAVILA:
1. Ako je "da" ispred glagola → particle (Q)
2. Ako je "ko" ispred glagola → pronoun (Pi)
3. Ako je "da" + "li" zajedno → question particles (Qq)
4. Ako je "sve" ispred verbalne ili opšte zamenice → general pronoun (Pg)
5. Ako je "sve" ispred priloga → general adverb (Rgp)
6. "svi", "sva" → adjective pronoun (Pi)
7. "koji", "koje", "koja" → adjective pronoun (Pi - indefinite/interrogative)
8. Ako je "da" ispred zareza → affirmative particle (Qr)
"""

from typing import List, Tuple
import re


# ==============================================================================
# MULTEXT-EAST V5 TAGOVI
# ==============================================================================

# Particle tags
TAG_PARTICLE_GENERAL = 'Q'           # General particle
TAG_PARTICLE_AFFIRMATIVE = 'Qr'      # Affirmative particle (da kao potvrda)
TAG_PARTICLE_NEGATIVE = 'Qz'         # Negative particle (ne)
TAG_PARTICLE_INTERROGATIVE = 'Qq'    # Interrogative particle (li, zar)

# Pronoun tags (indefinite/interrogative - "pridevske" zamenice)
TAG_PRONOUN_INDEFINITE = 'Pi'        # Indefinite pronoun (koji, ko, šta, neki)
TAG_PRONOUN_GENERAL = 'Pg'           # General pronoun (sve, sav)

# Adverb tags
TAG_ADVERB_GENERAL = 'Rgp'           # General adverb positive degree

# Conjunction (pogrešno tagirano u originalu)
TAG_CONJUNCTION = 'C'


def is_verb(msd: str) -> bool:
    """Proveri da li je MSD tag za glagol."""
    return msd.startswith('V') if msd else False


def is_adverb(msd: str) -> bool:
    """Proveri da li je MSD tag za prilog."""
    return msd.startswith('R') if msd else False


def is_pronoun(msd: str) -> bool:
    """Proveri da li je MSD tag za zamenicu."""
    return msd.startswith('P') if msd else False


def is_general_pronoun(msd: str) -> bool:
    """Proveri da li je MSD tag za opštu zamenicu (Pg)."""
    return msd.startswith('Pg') if msd else False


def is_verbal_pronoun(msd: str) -> bool:
    """Proveri da li je MSD tag za verbalnu zamenicu (osobne, povratne)."""
    return msd.startswith(('Pp', 'Px')) if msd else False


def is_conjunction(msd: str) -> bool:
    """Proveri da li je MSD tag za veznik."""
    return msd.startswith('C') if msd else False


def is_punctuation(token: str) -> bool:
    """Proveri da li je token interpunkcija."""
    return token in {',', '.', '!', '?', ';', ':', '-', '(', ')', '"', "'"}


def is_comma(token: str) -> bool:
    """Proveri da li je token zarez."""
    return token == ','


# ==============================================================================
# GLAVNA FUNKCIJA - PRIMENA PRAVILA NA REČENICU
# ==============================================================================

def apply_rules_to_sentence(tokens: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Primeni svih 8 gramatičkih pravila na rečenicu.

    Args:
        tokens: Lista (token, lemma, msd) za jednu rečenicu

    Returns:
        Lista (token, lemma, msd) sa ispravljenim tagovima
    """
    if not tokens:
        return tokens

    result = list(tokens)  # Napravi kopiju
    n = len(result)

    for i in range(n):
        token, lemma, msd = result[i]
        token_lower = token.lower()

        # Dobavi sledeći token ako postoji
        next_token = result[i + 1][0].lower() if i + 1 < n else None
        next_msd = result[i + 1][2] if i + 1 < n else None

        # ======================================================================
        # PRAVILO 3: "da" + "li" → question particles (Qq)
        # Mora biti pre pravila 1 jer ima prioritet
        # ======================================================================
        if token_lower == 'da' and next_token == 'li':
            result[i] = (token, lemma, TAG_PARTICLE_INTERROGATIVE)
            continue

        if token_lower == 'li' and i > 0 and result[i - 1][0].lower() == 'da':
            result[i] = (token, lemma, TAG_PARTICLE_INTERROGATIVE)
            continue

        # ======================================================================
        # PRAVILO 8: "da" ispred zareza → affirmative particle (Qr)
        # ======================================================================
        if token_lower == 'da' and is_conjunction(msd) and next_token and is_comma(next_token):
            result[i] = (token, lemma, TAG_PARTICLE_AFFIRMATIVE)
            continue

        # ======================================================================
        # PRAVILO 1: "da" ispred glagola → particle (Q)
        # ======================================================================
        if token_lower == 'da' and is_conjunction(msd) and next_msd and is_verb(next_msd):
            result[i] = (token, lemma, TAG_PARTICLE_GENERAL)
            continue

        # ======================================================================
        # PRAVILO 2: "ko" ispred glagola → pronoun (Pi)
        # ======================================================================
        if token_lower == 'ko' and is_conjunction(msd) and next_msd and is_verb(next_msd):
            # Ispravi na upitnu zamenicu - zadržaj gramatičke karakteristike ako postoje
            new_msd = 'Pi3nsn'  # interrogative, 3rd person, neuter, singular, nominative
            result[i] = (token, lemma, new_msd)
            continue

        # ======================================================================
        # PRAVILO 4: "sve" ispred verbalne ili opšte zamenice → general pronoun (Pg)
        # ======================================================================
        if token_lower == 'sve' and next_msd and (is_verbal_pronoun(next_msd) or is_general_pronoun(next_msd)):
            # Ispravi na opštu zamenicu
            new_msd = 'Pg-nsa'  # general, neuter, singular, accusative
            result[i] = (token, lemma, new_msd)
            continue

        # ======================================================================
        # PRAVILO 5: "sve" ispred priloga → general adverb (Rgp)
        # ======================================================================
        if token_lower == 'sve' and next_msd and is_adverb(next_msd):
            result[i] = (token, lemma, TAG_ADVERB_GENERAL)
            continue

        # ======================================================================
        # PRAVILO 6: "svi", "sva" → adjective pronoun (Pi)
        # ======================================================================
        if token_lower in {'svi', 'sva'}:
            if token_lower == 'svi':
                new_msd = 'Pi-mpn'  # indefinite, masculine, plural, nominative
            else:  # sva
                new_msd = 'Pi-fsn'  # indefinite, feminine, singular, nominative (ili Pi-npn za neutrum plural)
            result[i] = (token, lemma, new_msd)
            continue

        # ======================================================================
        # PRAVILO 7: "koji", "koje", "koja" → adjective pronoun (Pi)
        # ======================================================================
        if token_lower in {'koji', 'koje', 'koja'} and not msd.startswith('Pi'):
            # Ispravi na indefinite/interrogative pronoun
            if token_lower == 'koji':
                new_msd = 'Pi-msn'  # masculine singular nominative
            elif token_lower == 'koja':
                new_msd = 'Pi-fsn'  # feminine singular nominative
            else:  # koje
                new_msd = 'Pi-nsn'  # neuter singular nominative (ili plural)
            result[i] = (token, lemma, new_msd)
            continue

    return result


# ==============================================================================
# WRAPPER ZA POJEDINAČNE TOKENE (backward compatibility)
# ==============================================================================

def apply_all_rules(token: str, lemma: str, msd: str) -> Tuple[str, str, str]:
    """
    Primeni pravila na pojedinačni token (bez konteksta).

    NAPOMENA: Ova funkcija ne može da primeni kontekst-zavisna pravila (1-5, 8).
    Koristi apply_rules_to_sentence() za punu funkcionalnost.

    Args:
        token: Originalni token (reč)
        lemma: Lema (osnovni oblik)
        msd: Morfosintaktički deskriptor (MULTEXT-East V5)

    Returns:
        Tuple (token, lemma, msd) - možda izmenjeni
    """
    token_lower = token.lower()

    # Pravilo 6: "svi", "sva" → adjective pronoun
    if token_lower in {'svi', 'sva'}:
        if token_lower == 'svi':
            msd = 'Pi-mpn'
        else:
            msd = 'Pi-fsn'

    # Pravilo 7: "koji", "koje", "koja" → adjective pronoun
    if token_lower in {'koji', 'koje', 'koja'} and not msd.startswith('Pi'):
        if token_lower == 'koji':
            msd = 'Pi-msn'
        elif token_lower == 'koja':
            msd = 'Pi-fsn'
        else:
            msd = 'Pi-nsn'

    return token, lemma, msd


# ==============================================================================
# STATISTIKA PRIMENJENIH PRAVILA
# ==============================================================================

class RuleStatistics:
    """Prati statistiku primenjenih pravila."""

    def __init__(self):
        self.counts = {
            'rule_1_da_before_verb': 0,
            'rule_2_ko_before_verb': 0,
            'rule_3_da_li_question': 0,
            'rule_4_sve_before_pronoun': 0,
            'rule_5_sve_before_adverb': 0,
            'rule_6_svi_sva': 0,
            'rule_7_koji_koje_koja': 0,
            'rule_8_da_before_comma': 0,
        }

    def increment(self, rule_name: str):
        if rule_name in self.counts:
            self.counts[rule_name] += 1

    def report(self) -> str:
        lines = ["Statistika primenjenih pravila:", "-" * 40]
        total = 0
        for rule, count in self.counts.items():
            lines.append(f"  {rule}: {count:,}")
            total += count
        lines.append("-" * 40)
        lines.append(f"  UKUPNO: {total:,}")
        return '\n'.join(lines)


# ==============================================================================
# TESTIRANJE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTIRANJE GRAMATIČKIH PRAVILA")
    print("=" * 70)

    # Test primeri - rečenice
    test_sentences = [
        # Pravilo 1: "da" ispred glagola
        [("Hoću", "hteti", "Vmip1s"), ("da", "da", "Cs"), ("radim", "raditi", "Vmip1s")],

        # Pravilo 2: "ko" ispred glagola
        [("Ko", "ko", "Cs"), ("radi", "raditi", "Vmip3s"), ("?", "?", "Z")],

        # Pravilo 3: "da li"
        [("Da", "da", "Cs"), ("li", "li", "Q"), ("ideš", "ići", "Vmip2s"), ("?", "?", "Z")],

        # Pravilo 5: "sve" ispred priloga
        [("Sve", "sve", "Agpnsa"), ("brže", "brzo", "Rgc"), ("trči", "trčati", "Vmip3s")],

        # Pravilo 6: "svi"
        [("Svi", "sav", "Agpmpn"), ("su", "biti", "Var3p"), ("došli", "doći", "Vmp-pm")],

        # Pravilo 7: "koji"
        [("Čovek", "čovek", "Ncmsn"), ("koji", "koji", "Cs"), ("radi", "raditi", "Vmip3s")],

        # Pravilo 8: "da" ispred zareza
        [("Da", "da", "Cs"), (",", ",", "Z"), ("da", "da", "Cs"), (".", ".", "Z")],
    ]

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"  Input:  {[(t, m) for t, l, m in sentence]}")

        corrected = apply_rules_to_sentence(sentence)

        print(f"  Output: {[(t, m) for t, l, m in corrected]}")

        # Prikaži izmene
        changes = []
        for (t1, l1, m1), (t2, l2, m2) in zip(sentence, corrected):
            if m1 != m2:
                changes.append(f"'{t1}': {m1} → {m2}")

        if changes:
            print(f"  Izmene: {', '.join(changes)}")
        else:
            print(f"  Izmene: (nema)")

    print("\n" + "=" * 70)
    print("TESTIRANJE ZAVRŠENO")
    print("=" * 70)
