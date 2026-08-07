#!/usr/bin/env python3
"""
Klasifikacija zamenica na imenske (nominal) i pridevske (adjectival).

Modifikuje MSD tag: P + n/a + podvrsta + ostali atributi
  Pp1msn  →  Pnp1msn   (lična zamenica, imenska)
  Pd-msn  →  Pad-msn   (pokazna, pridevska)
  Px--sa  →  Pnx--sa   (sebe/se, imenska)
  Px-msan →  Pax-msan  (svoj, pridevska)

Ulaz: srWaC CSV (Token,Lemma,MSD,Sentence_ID)
Izlaz: Modifikovani CSV sa novim MSD tagovima za zamenice

Autor: Nikola Vukotić
Datum: Jun 2026
"""

import csv
import os
import sys
import time

# ============================================================================
# KLASIFIKACIJA PO LEMI
# ============================================================================

# Imeničke zamenice - stoje samostalno kao imenička fraza
# Izvor: srpska normativna gramatika + srLex 1.3
IMENSKE_LEME = frozenset({
    # Lične (Pp)
    'ja', 'ti', 'on', 'ona', 'ono', 'mi', 'vi', 'oni', 'one',

    # Povratna lična (Px) - samo "sebe/se", NE "svoj"
    'sebe', 'se',

    # Upitne imeničke (Pq)
    'ko', 'tko', 'šta', 'što', 'zašto',

    # Neodređene imeničke (Pi)
    'neko', 'netko', 'nešto', 'ponešto',

    # Odrične imeničke (Pi)
    'niko', 'nitko', 'ništa',

    # Opšte imeničke (Pi)
    'svako', 'svatko', 'svašta', 'štošta', 'koješta',

    # Ikoje imeničke (Pi)
    'iko', 'itko', 'išta', 'kogod', 'štagod',
})

# Pridevske leme - SVE ostale zamenice su pridevske
# Uključuje: pokazne (Pd), prisvojne (Ps), povratno-prisvojnu (svoj),
# upitno-odnosne pridevske (koji, čiji, kakav, koliki),
# neodređene pridevske (neki, nekakav...), odrične (nikakav...),
# opšte pridevske (svaki, svačiji...), sav, oboje

# Px specijalan slučaj: "sebe/se" = imenska, "svoj" i oblici = pridevska
PX_PRIDEVSKE_LEME = frozenset({
    'svoj', 'svoja', 'svoje', 'svog', 'svoga', 'svojeg', 'svojega',
    'svom', 'svome', 'svomu', 'svojem', 'svojemu', 'svojih', 'svojim',
    'svojima', 'svojoj', 'svojom', 'svoju',
})


def classify_pronoun(msd: str, lemma: str) -> str:
    """Klasifikuj zamenicu i modifikuj MSD tag.

    Args:
        msd: Originalni MSD tag (npr. 'Pp1msn')
        lemma: Lema reči (npr. 'ja')

    Returns:
        Modifikovani MSD tag (npr. 'Pnp1msn')
    """
    if not msd or not msd.startswith('P') or len(msd) < 2:
        return msd

    lemma_lower = lemma.lower().strip()
    subtype = msd[1]  # p, x, d, i, s, q, r, g

    # Pp (lične) - uvek imeničke
    if subtype == 'p':
        return 'Pn' + msd[1:]

    # Px - zavisi od leme: sebe/se = imenska, svoj = pridevska
    if subtype == 'x':
        if lemma_lower in IMENSKE_LEME or lemma_lower in ('se', 'sebe', 'sebi', 'sobom'):
            return 'Pn' + msd[1:]
        else:
            return 'Pa' + msd[1:]

    # Pd (pokazne) - uvek pridevske
    if subtype == 'd':
        return 'Pa' + msd[1:]

    # Ps (prisvojne) - uvek pridevske
    if subtype == 's':
        return 'Pa' + msd[1:]

    # Pi (neodređene/odrične/opšte) - zavisi od leme
    if subtype == 'i':
        if lemma_lower in IMENSKE_LEME:
            return 'Pn' + msd[1:]
        else:
            return 'Pa' + msd[1:]

    # Pq (upitne) - zavisi od leme
    if subtype == 'q':
        if lemma_lower in IMENSKE_LEME:
            return 'Pn' + msd[1:]
        else:
            return 'Pa' + msd[1:]

    # Pr (odnosne), Pg (opšte) - pridevske po defaultu
    if subtype in ('r', 'g'):
        return 'Pa' + msd[1:]

    # Nepoznat tip - ne diraj
    return msd


# ============================================================================
# OBRADA FAJLA
# ============================================================================

def process_file(input_path: str, output_path: str):
    """Obradi srWaC CSV i klasifikuj zamenice.

    Čita red po red (ne koristi pandas) za memorijsku efikasnost
    i da izbegne NA/NaN probleme.
    """
    total = 0
    modified = 0
    nominal = 0
    adjectival = 0

    print(f"Ulaz:  {input_path}")
    print(f"Izlaz: {output_path}")
    print()

    t0 = time.time()

    with open(input_path, 'r', encoding='utf-8', newline='') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # Zaglavlje
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            total += 1

            if total % 5_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  Obrađeno {total:,} redova ({elapsed:.0f}s)")

            if len(row) < 3:
                writer.writerow(row)
                continue

            token, lemma, msd = row[0], row[1], row[2]

            # Samo zamenice (P tag)
            if msd.startswith('P'):
                new_msd = classify_pronoun(msd, lemma)
                if new_msd != msd:
                    modified += 1
                    if new_msd[1] == 'n':
                        nominal += 1
                    elif new_msd[1] == 'a':
                        adjectival += 1
                    row[2] = new_msd

            writer.writerow(row)

    elapsed = time.time() - t0
    print(f"\nZavršeno za {elapsed:.1f}s")
    print(f"  Ukupno redova:       {total:,}")
    print(f"  Modifikovanih:       {modified:,}")
    print(f"  Imeničke (Pn):       {nominal:,}")
    print(f"  Pridevske (Pa):      {adjectival:,}")
    print(f"  Nemodifikovane:      {total - modified:,}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  KLASIFIKACIJA ZAMENICA: IMENSKE vs PRIDEVSKE")
    print("  Format: P + n/a + podvrsta + atributi")
    print("=" * 60)
    print()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "data", "processed", "srWaC1.1.01_processed.csv")
    output_file = os.path.join(base_dir, "data", "processed", "srWaC1.1.01_pronouns_classified.csv")

    if not os.path.exists(input_file):
        print(f"GREŠKA: Fajl ne postoji: {input_file}")
        sys.exit(1)

    process_file(input_file, output_file)

    print(f"\nNovi fajl: {output_file}")
    print("Gotovo!")


if __name__ == '__main__':
    main()
