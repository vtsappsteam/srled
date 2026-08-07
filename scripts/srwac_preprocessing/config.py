#!/usr/bin/env python3
"""
Konfiguracija za preprocesiranje srWaC korpusa.

Autor: Nikola Vukotić
Datum: April 2026
"""

from pathlib import Path

# ==============================================================================
# PUTANJE
# ==============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"

# Kreiraj foldere ako ne postoje
for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MULTEXT-EAST V5 KONFIGURACIJA
# ==============================================================================

# POS kategorije
POS_CATEGORIES = {
    'N': 'Noun',
    'V': 'Verb',
    'A': 'Adjective',
    'P': 'Pronoun',
    'R': 'Adverb',
    'S': 'Adposition',
    'C': 'Conjunction',
    'M': 'Numeral',
    'Q': 'Particle',
    'I': 'Interjection',
    'Y': 'Abbreviation',
    'X': 'Residual',
    'Z': 'Punctuation',
}

# Atribut pozicije po POS kategoriji (MULTEXT-East V5 za srpski)
# KRITIČNO: Ove pozicije su verifikovane iz zvanične specifikacije
ATTRIBUTE_POSITIONS = {
    'N': {  # Noun
        'Type': 1,
        'Gender': 2,
        'Number': 3,
        'Case': 4,
        'Animate': 5,
    },
    'V': {  # Verb
        'Type': 1,
        'VForm': 2,
        'Tense': 3,
        'Person': 4,
        'Number': 5,
        'Gender': 6,
        'Voice': 7,
        'Negative': 8,
        'Clitic': 9,
        'Aspect': 10,
    },
    'A': {  # Adjective - PAŽNJA: Case je na poziciji 5, NE 4!
        'Type': 1,
        'Degree': 2,
        'Gender': 3,
        'Number': 4,
        'Case': 5,
        'Definiteness': 6,
        'Animate': 7,
    },
    'P': {  # Pronoun - Case je na poziciji 5
        'Type': 1,
        'Person': 2,
        'Gender': 3,
        'Number': 4,
        'Case': 5,
        'Owner_Number': 6,
        'Owner_Gender': 7,
        'Animate': 8,
    },
    'M': {  # Numeral - Case je na poziciji 4
        'Type': 1,
        'Gender': 2,
        'Number': 3,
        'Case': 4,
        'Form': 5,
        'Animate': 6,
    },
    'R': {  # Adverb
        'Type': 1,
        'Degree': 2,
    },
    'S': {  # Adposition
        'Type': 1,
        'Formation': 2,
        'Case': 3,
    },
    'C': {  # Conjunction
        'Type': 1,
        'Formation': 2,
    },
}

# Mapiranje Case vrednosti
CASE_VALUES = {
    'n': 'Nominative',
    'g': 'Genitive',
    'd': 'Dative',
    'a': 'Accusative',
    'v': 'Vocative',
    'l': 'Locative',
    'i': 'Instrumental',
}

# ==============================================================================
# PROCESIRANJE
# ==============================================================================

# Chunk size za procesiranje velikih fajlova
CHUNK_SIZE = 500000

# Encoding
ENCODING = 'utf-8'

# Da li čuvati ćirilicu ili konvertovati u latinicu
CONVERT_CYRILLIC_TO_LATIN = True
