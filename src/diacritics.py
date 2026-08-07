"""
Serbian Latin diacritical character definitions and utilities.

Serbian Latin script (Vuk's alphabet) has 30 graphemes, five of which
carry diacritical marks:
    c with caron  (IPA: /ts/)    c with acute  (IPA: /tc/)
    z with caron  (IPA: /z/)     s with caron  (IPA: /s/)
    d with stroke (IPA: /dz/)

When diacritics are omitted, the following ambiguities arise:
    both c-caron and c-acute collapse to "c"
    s-caron collapses to "s"
    z-caron collapses to "z"
    d-stroke collapses to "dj" or "d"
"""

DIACRITICS_TO_ASCII = {
    '\u010d': 'c',  '\u010c': 'C',   # c-caron
    '\u0107': 'c',  '\u0106': 'C',   # c-acute
    '\u017e': 'z',  '\u017d': 'Z',   # z-caron
    '\u0161': 's',  '\u0160': 'S',   # s-caron
    '\u0111': 'dj', '\u0110': 'Dj',  # d-stroke
}

ASCII_TO_DIACRITICS = {
    'c': ['c', '\u010d', '\u0107'],
    'C': ['C', '\u010c', '\u0106'],
    's': ['s', '\u0161'],
    'S': ['S', '\u0160'],
    'z': ['z', '\u017e'],
    'Z': ['Z', '\u017d'],
}

AMBIGUOUS_CHARS = set('cCsSzZ')
DIACRITIC_CHARS = set('\u010d\u0107\u017e\u0161\u0111\u010c\u0106\u017d\u0160\u0110')


def strip_diacritics(text):
    """Remove diacritical marks from text."""
    return ''.join(DIACRITICS_TO_ASCII.get(ch, ch) for ch in text)


def has_diacritics(text):
    """Check whether text contains any diacritical characters."""
    return any(ch in DIACRITIC_CHARS for ch in text)


def needs_restoration(word):
    """Check whether a word could potentially contain diacritics."""
    if any(ch in AMBIGUOUS_CHARS for ch in word):
        return True
    lower = word.lower()
    return 'dj' in lower


def normalize_for_lookup(word):
    """Normalize word for dictionary lookup: lowercase + strip diacritics."""
    return strip_diacritics(word.lower())
