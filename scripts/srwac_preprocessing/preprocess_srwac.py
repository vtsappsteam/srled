#!/usr/bin/env python3
"""
Glavna skripta za preprocesiranje srWaC korpusa.

Ova skripta:
1. Parsira XML fajlove srWaC korpusa
2. Primenjuje 7 gramatičkih pravila za korekciju
3. Čuva rezultate u CSV i TXT formatu

Autor: Nikola Vukotić
Datum: April 2026

Korišćenje:
    python preprocess_srwac.py --input data/raw/srWaC1.1.01.xml --output data/processed/
    python preprocess_srwac.py --input data/raw/ --output data/processed/ --all
"""

import argparse
import csv
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple, Generator
import sys

# Lokalni importi
from config import (
    RAW_DIR, PROCESSED_DIR, LOG_DIR,
    CHUNK_SIZE, ENCODING, CONVERT_CYRILLIC_TO_LATIN
)
from grammar_rules import apply_rules_to_sentence


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def setup_logging(log_file: Path = None):
    """Podesi logging."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"preprocess_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==============================================================================
# ĆIRILICA -> LATINICA KONVERZIJA
# ==============================================================================
CYRILLIC_TO_LATIN = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E',
    'Ж': 'Ž', 'З': 'Z', 'И': 'I', 'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj',
    'М': 'M', 'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S',
    'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č',
    'Џ': 'Dž', 'Ш': 'Š',
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e',
    'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj',
    'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's',
    'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č',
    'џ': 'dž', 'ш': 'š',
}


def cyrillic_to_latin(text: str) -> str:
    """Konvertuj ćirilicu u latinicu."""
    result = []
    for char in text:
        if char in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[char])
        else:
            result.append(char)
    return ''.join(result)


# ==============================================================================
# XML PARSIRANJE
# ==============================================================================
def parse_xml_file(xml_path: Path, logger) -> Generator[Tuple[str, str, str, int], None, None]:
    """
    Parsiraj XML fajl i vrati tokene.

    Yields:
        Tuple (token, lemma, msd, sentence_id)
    """
    logger.info(f"Parsiram: {xml_path}")

    sentence_id = 0

    try:
        context = ET.iterparse(str(xml_path), events=('end',))

        for event, elem in context:
            if elem.tag == 's':
                sentence_id += 1

                if elem.text:
                    # Prikupi sve tokene u rečenici
                    sentence_tokens = []
                    lines = elem.text.strip().split('\n')

                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            token, lemma, base_form, msd = parts[0], parts[1], parts[2], parts[3]

                            # Konvertuj ćirilicu ako je potrebno
                            if CONVERT_CYRILLIC_TO_LATIN:
                                token = cyrillic_to_latin(token)
                                lemma = cyrillic_to_latin(lemma)

                            sentence_tokens.append((token, lemma, msd))

                    # Primeni gramatička pravila na celu rečenicu
                    corrected_tokens = apply_rules_to_sentence(sentence_tokens)

                    # Vrati ispravljene tokene
                    for token, lemma, msd in corrected_tokens:
                        yield token, lemma, msd, sentence_id

                # Oslobodi memoriju
                elem.clear()

    except ET.ParseError as e:
        logger.error(f"XML greška: {e}")
        raise


def count_sentences(xml_path: Path) -> int:
    """Prebroj rečenice u XML fajlu za progress bar."""
    count = 0
    try:
        context = ET.iterparse(str(xml_path), events=('end',))
        for event, elem in context:
            if elem.tag == 's':
                count += 1
            elem.clear()
    except:
        pass
    return count


# ==============================================================================
# OUTPUT FUNKCIJE
# ==============================================================================
def save_to_csv(tokens: List[Tuple[str, str, str, int]], output_path: Path, logger):
    """Sačuvaj tokene u CSV format."""
    logger.info(f"Čuvam CSV: {output_path}")

    with open(output_path, 'w', newline='', encoding=ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(['Token', 'Lemma', 'MSD', 'Sentence_ID'])
        writer.writerows(tokens)

    logger.info(f"Sačuvano {len(tokens)} tokena u {output_path}")


def save_to_txt(tokens: List[Tuple[str, str, str, int]], output_path: Path, logger):
    """Sačuvaj tokene u TXT format (NLTK kompatibilno)."""
    logger.info(f"Čuvam TXT: {output_path}")

    current_sentence = []
    current_sentence_id = None

    with open(output_path, 'w', encoding=ENCODING) as f:
        for token, lemma, msd, sentence_id in tokens:
            if current_sentence_id is None:
                current_sentence_id = sentence_id

            if sentence_id != current_sentence_id:
                # Nova rečenica - zapiši prethodnu
                f.write(' '.join(current_sentence) + '\n')
                current_sentence = []
                current_sentence_id = sentence_id

            current_sentence.append(f"{token}/{msd}")

        # Zapiši poslednju rečenicu
        if current_sentence:
            f.write(' '.join(current_sentence) + '\n')

    logger.info(f"Sačuvano u {output_path}")


# ==============================================================================
# GLAVNA FUNKCIJA
# ==============================================================================
def process_file(input_path: Path, output_dir: Path, logger):
    """Procesiraj jedan XML fajl."""
    logger.info(f"=" * 70)
    logger.info(f"Procesiranje: {input_path.name}")
    logger.info(f"=" * 70)

    # Output putanje
    base_name = input_path.stem
    csv_output = output_dir / f"{base_name}_processed.csv"
    txt_output = output_dir / f"{base_name}_processed.txt"

    # Parsiraj i prikupi tokene
    tokens = []
    total_sentences = count_sentences(input_path)

    with tqdm(total=total_sentences, desc="Procesiranje") as pbar:
        last_sentence_id = 0
        for token, lemma, msd, sentence_id in parse_xml_file(input_path, logger):
            tokens.append((token, lemma, msd, sentence_id))
            if sentence_id != last_sentence_id:
                pbar.update(1)
                last_sentence_id = sentence_id

    # Sačuvaj rezultate
    save_to_csv(tokens, csv_output, logger)
    save_to_txt(tokens, txt_output, logger)

    # Statistika
    logger.info(f"Statistika:")
    logger.info(f"  - Ukupno tokena: {len(tokens):,}")
    logger.info(f"  - Ukupno rečenica: {sentence_id:,}")

    return len(tokens), sentence_id


def main():
    parser = argparse.ArgumentParser(description='Preprocesiranje srWaC korpusa')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input XML fajl ili folder')
    parser.add_argument('--output', '-o', type=str, default=str(PROCESSED_DIR),
                        help='Output folder')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Procesiraj sve XML fajlove u folderu')

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PREPROCESIRANJE srWaC KORPUSA")
    logger.info("=" * 70)

    total_tokens = 0
    total_sentences = 0

    if args.all and input_path.is_dir():
        # Procesiraj sve XML fajlove
        xml_files = sorted(input_path.glob('*.xml'))
        logger.info(f"Pronađeno {len(xml_files)} XML fajlova")

        for xml_file in xml_files:
            tokens, sentences = process_file(xml_file, output_dir, logger)
            total_tokens += tokens
            total_sentences += sentences

    elif input_path.is_file():
        # Procesiraj jedan fajl
        total_tokens, total_sentences = process_file(input_path, output_dir, logger)

    else:
        logger.error(f"Invalid input: {input_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("ZAVRŠENO")
    logger.info(f"  - Ukupno tokena: {total_tokens:,}")
    logger.info(f"  - Ukupno rečenica: {total_sentences:,}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
