#!/usr/bin/env python3
"""Table 7: MSD granularity on UD test with the FULL system configuration
(expanded supplement + integrated v6 Layer 0)."""
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'src'))
import evaluate_three_testsets as ev3

OUT = str(BASE / 'results' / 'granularity_full_system.json')


def main():
    lem = ev3.Lemmatizer(ev3.SRLEX, ev3.EXTRA)
    sents = ev3.load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    res = {}
    for mode, mk in [('full', lambda m: m),
                     ('coarse', lambda m: m[0] if m else ''),
                     ('none', lambda m: '')]:
        tot = ok = 0
        for s in sents:
            for w, gl, msd in s:
                tot += 1
                p = lem.lemmatize(w, mk(msd))
                if p and gl and p.lower() == gl.lower():
                    ok += 1
        res[mode] = round(100 * ok / tot, 2)
        print(mode, res[mode], flush=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
