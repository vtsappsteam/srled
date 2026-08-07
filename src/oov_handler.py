"""
OOV (Out-of-Vocabulary) handler za vraćanje dijakritika.

Za reči koje nisu u srLex-u, koristi:
1. Pravila za digraf dž (dz → dž osim na granici prefiksa)
2. Edit distance prema srLex-u (za reči bliske poznatim oblicima)
3. Character-level heuristike

"""

import re
from diacritics import DIACRITIC_CHARS, strip_diacritics


# ============================================================
# 1. Pravila za digraf dž
# ============================================================

# Prefiksi koji završavaju na "d" ispred korena na "z"
# U tim slučajevima "dz" je PRAVO dz (d+z), NE dž
DZ_PREFIX_EXCEPTIONS = {
    'nadz': 'nadz',    # nadzor, nadzirati, nadzornik
    'podz': 'podz',    # podzemni, podzakonski, podzemlje
    'odz': 'odz',      # odziv, odzvanja, odzvoniti
    'predz': 'predz',  # predznak, predznanje
}

# Poznate reči sa pravim "dz" (ne dž)
DZ_EXCEPTIONS = {
    'nadzor', 'nadzora', 'nadzoru', 'nadzorom', 'nadzori', 'nadzore',
    'nadzornik', 'nadzornika', 'nadzorniku', 'nadzornici',
    'nadzirati', 'nadzirem', 'nadzire', 'nadziru',
    'podzemni', 'podzemna', 'podzemne', 'podzemno', 'podzemnih',
    'podzemnu', 'podzemnom', 'podzemlje', 'podzemnon',
    'podzakonski', 'podzakonskih', 'podzakonskim',
    'odziv', 'odziva', 'odzivu', 'odzivom', 'odzivi',
    'odzvanja', 'odzvanjaju', 'odzvanjao',
    'odzvoniti', 'odzvonilo', 'odzvoni',
    'predznak', 'predznaka', 'predznaku', 'predznakom',
    'predznanje', 'predznanja', 'predznanju',
}


# ============================================================
# 1b. Pravila za digraf dj (dj → đ za OOV reči)
# ============================================================

# Počeci reči kod kojih je "dj" najčešće PRAVO d+j (ijekavski oblici
# i prefiksi na granici morfeme), izvedeno iz srLex statistike:
# odj- (odjek), podj- (podjednako), nedj- (nedjelja), nadj- (nadjačati),
# udj- (udjenuti), razdj- (razdjel), predj- (predjelo), snabdj-, pridj-, zdj-
DJ_PREFIX_EXCEPTIONS = (
    'odj', 'podj', 'nedj', 'nadj', 'udj', 'razdj', 'predj',
    'snabdj', 'pridj', 'zdj', 'vidj', 'sudj', 'gdj',
)


def apply_dj_rules(word):
    """
    Primenjuje dj → đ konverziju za OOV reči.

    U srLex-u 33,294 oblika sadrži đ (ASCII: dj) prema 8,657 oblika sa
    pravim d+j, pri čemu su pravi d+j oblici koncentrisani na poznate
    ijekavske/prefiksalne početke. Za OOV reči van tih obrazaca,
    "dj" je najverovatnije đ (npr. strana imena: Djokovic → Đokovic).
    """
    lower = word.lower()
    for prefix in DJ_PREFIX_EXCEPTIONS:
        if lower.startswith(prefix):
            return word
    result = word.replace('dj', 'đ')
    result = result.replace('Dj', 'Đ')
    result = result.replace('DJ', 'Đ')
    return result


def apply_dz_rules(word):
    """
    Primenjuje pravila za dz → dž konverziju.

    Pravilo: "dz" je skoro uvek "dž" (82% slučajeva u srLex-u),
    OSIM na granici prefiksa (nad-zor, pod-zemni, od-zvanja, pred-znak).

    Args:
        word: reč bez dijakritika koja sadrži "dz"

    Returns:
        Reč sa primenjenom dž konverzijom (ili originalna ako je izuzetak)
    """
    lower = word.lower()

    # Proveri da li je u listi izuzetaka
    if lower in DZ_EXCEPTIONS:
        return word  # zadrži originalno "dz"

    # Proveri prefiksna pravila
    for prefix in DZ_PREFIX_EXCEPTIONS:
        if lower.startswith(prefix):
            return word  # zadrži "dz"

    # Default: zameni "dz" sa "dž"
    result = word
    result = result.replace('dz', 'dž')
    result = result.replace('Dz', 'Dž')
    result = result.replace('DZ', 'DŽ')

    return result


# ============================================================
# 2. Character-level heuristike
# ============================================================

# Česti obrasci gde je zamena skoro sigurna
SAFE_PATTERNS = [
    # š obrasci
    (r'(?<=[aeiou])s(?=t[aeiou])', 'š'),     # asto→ašto, ista→išta
    (r'^sk(?=[aeiou])', 'šk'),                 # ska→ška, sko→ško
    (r'(?<=[kptrn])s(?=[kt])', 'š'),           # nst→nšt

    # č obrasci
    (r'(?<=[aeiou])c(?=[aeiou])', None),       # ambiguous, skip

    # ž obrasci
    (r'(?<=[aeiou])z(?=[aeiou])', None),       # ambiguous, skip
]


def apply_heuristics(word, word_set=None):
    """
    Primenjuje heuristike za OOV reči.

    Redosled:
    1. č/ć pravila za poznate obrasce (ne- + će/ću/ćeš/ćemo/ćete)
    2. Sufiksna pravila za OOV reči (-scenje→-šćenje, -senje→-šenje, itd.)
    3. dz → dž pravilo
    4. Edit distance lookup u rečniku

    Args:
        word: reč bez dijakritika
        word_set: set poznatih reči (opciono, za edit distance)

    Returns:
        Reč sa primenjenim heuristikama
    """
    result = word
    lower = word.lower()

    # 1. č/ć pravila za negaciju pomoćnog glagola
    # ne + će/ću/ćeš/ćemo/ćete → uvek ć (ne č)
    CC_NEGATION = {
        'nece': 'neće', 'necu': 'neću', 'neces': 'nećeš',
        'necemo': 'nećemo', 'necete': 'nećete',
    }
    if lower in CC_NEGATION:
        replacement = CC_NEGATION[lower]
        return _match_heuristic_case(word, replacement)

    # Slično za složene oblike: dobiće, moraće, biće, imaće, itd.
    # Obrazac: *ce na kraju gde je ć (futur), ne č
    CC_FUTURE_SUFFIXES = {
        'bice': 'biće', 'dobice': 'dobiće', 'morace': 'moraće',
        'imace': 'imaće', 'radice': 'radiće', 'videce': 'videće',
        'trebace': 'trebaće', 'moci': 'moći', 'doci': 'doći',
        'otici': 'otići', 'proci': 'proći', 'izaci': 'izaći',
        'uci': 'ući', 'naci': 'naći', 'poci': 'poći',
        'reci': 'reći',  # infinitiv "reći" je češći od "reči" i "reci"
    }
    if lower in CC_FUTURE_SUFFIXES:
        replacement = CC_FUTURE_SUFFIXES[lower]
        return _match_heuristic_case(word, replacement)

    # 2. Sufiksna pravila za OOV reči
    SUFFIX_RULES = [
        ('scenje', 'šćenje'),     # koriscenje → korišćenje
        ('scenja', 'šćenja'),
        ('scenju', 'šćenju'),
        ('scenjem', 'šćenjem'),
        ('senje', 'šenje'),       # hapsenje → hapšenje
        ('senja', 'šenja'),
        ('senju', 'šenju'),
        ('senjem', 'šenjem'),
        ('cionise', 'cioniše'),   # funkcionise → funkcioniše
        ('cionisu', 'cionišu'),
    ]
    for ascii_suf, diac_suf in SUFFIX_RULES:
        if lower.endswith(ascii_suf):
            prefix = word[:len(word)-len(ascii_suf)]
            result = prefix + diac_suf
            return result

    # 3. dz → dž pravilo
    if 'dz' in lower:
        result = apply_dz_rules(result)
        if result != word:
            return result

    # 3b. dj → đ pravilo (posle dz, jer "dz" ima prioritet u npr. "dzudo")
    if 'dj' in lower:
        result = apply_dj_rules(result)
        if result != word:
            return result

    # 4. Edit distance lookup
    if word_set:
        found = find_closest_in_dict(word, word_set, max_distance=3)
        if found:
            return found

    return result


def _match_heuristic_case(original, replacement):
    """Prilagodi casing."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    elif original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


# ============================================================
# 3. Edit distance lookup
# ============================================================

def find_closest_in_dict(word, word_set, max_distance=3):
    """
    Pronalazi najbližu reč u rečniku po dijakritičkom edit distance-u.

    Zamene su ograničene na dijakritičke parove:
        c → č, ć
        s → š
        z → ž

    Strategija: počni sa 1 zamenom, pa 2, pa 3.
    Na svakom nivou, ako tačno 1 kandidat postoji, vrati ga.
    Ako ih ima više, vrati None (nejednoznačno).

    Args:
        word: reč bez dijakritika
        word_set: set poznatih reči (sa dijakritikama)
        max_distance: maksimalni broj zamena

    Returns:
        Najbliža reč ili None
    """
    for dist in range(1, max_distance + 1):
        candidates = generate_diacritic_variants(word, dist)
        found = [c for c in candidates if c in word_set]

        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            # Više kandidata na istom distance-u -- nejednoznačno
            # Ali ako svi imaju iste dijakritike na različitim pozicijama
            # (npr. "koriscenje" → samo "korišćenje"), vrati
            unique_lower = set(f.lower() for f in found)
            if len(unique_lower) == 1:
                return found[0]
            return None

    return None


def generate_diacritic_variants(word, max_changes=1):
    """
    Generiše sve moguće varijante reči sa do max_changes
    dijakritičkih zamena.

    Primer: "cas" → ["čas", "ćas", "caš"]
    """
    replacements = {
        'c': ['č', 'ć'],
        'C': ['Č', 'Ć'],
        's': ['š'],
        'S': ['Š'],
        'z': ['ž'],
        'Z': ['Ž'],
    }

    variants = set()
    chars = list(word)

    # Pronađi pozicije gde je moguća zamena
    positions = []
    for i, ch in enumerate(chars):
        if ch in replacements:
            positions.append(i)

    # Generiši varijante sa 1 zamenom
    for pos in positions:
        ch = chars[pos]
        for repl in replacements[ch]:
            variant = chars.copy()
            variant[pos] = repl
            variants.add(''.join(variant))

    # Generiši varijante sa 2 zamene (ako max_changes > 1)
    if max_changes > 1:
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                ch1 = chars[pos1]
                ch2 = chars[pos2]
                for repl1 in replacements[ch1]:
                    for repl2 in replacements[ch2]:
                        variant = chars.copy()
                        variant[pos1] = repl1
                        variant[pos2] = repl2
                        variants.add(''.join(variant))

    return list(variants)


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    # Test dž pravila
    test_words = [
        'dzep', 'budzet', 'udzbenika', 'dzez', 'menadzer',
        'nadzor', 'podzemni', 'odziv', 'predznak', 'odzvanja',
        'tinejdzer', 'dzungla', 'Dzoni', 'otadzbina',
    ]

    print("Test dž pravila:")
    print(f"{'Ulaz':<20} {'Izlaz':<20} {'Ispravno?'}")
    print("-" * 50)

    expected = {
        'dzep': 'džep', 'budzet': 'budžet', 'udzbenika': 'udžbenika',
        'dzez': 'džez', 'menadzer': 'menadžer',
        'nadzor': 'nadzor', 'podzemni': 'podzemni',
        'odziv': 'odziv', 'predznak': 'predznak', 'odzvanja': 'odzvanja',
        'tinejdzer': 'tinejdžer', 'dzungla': 'džungla',
        'Dzoni': 'Džoni', 'otadzbina': 'otadžbina',
    }

    correct = 0
    for word in test_words:
        result = apply_dz_rules(word)
        exp = expected.get(word, word)
        ok = '✓' if result == exp else '✗'
        if result == exp:
            correct += 1
        print(f"  {word:<20} {result:<20} {ok} (trebalo: {exp})")

    print(f"\nTačnost: {correct}/{len(test_words)}")

    # Test variant generation
    print(f"\nVarijante za 'cas': {generate_diacritic_variants('cas')}")
    print(f"Varijante za 'reci': {generate_diacritic_variants('reci')}")
    print(f"Varijante za 'zivim': {generate_diacritic_variants('zivim', max_changes=2)}")
