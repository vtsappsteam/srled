"""
V6Restorer: jedinstveni Layer 0 za lematizator.

Tanak omotač oko CandidateGenerator + POSDisambiguator koji lematizatoru
daje jednu metodu restore(word, msd). Poziva se SAMO za reči koje su OOV
za lematizator i ne sadrže dijakritike; unutra radi puni v6 tok:
srLex(+expanded) kandidati → Np hijerarhija → POS filter + dominance
guard → č/ć obrasci → OOV lanac (srWaC tabele, hyphen, dž/đ, sufiksi).

"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator
from diacritics import needs_restoration

_DATA = os.path.join(_SRC, '..', 'data')


class V6Restorer:
    def __init__(self, srlex_path, extra_dict_path=None,
                 supplement_path=None, augment_np_path=None):
        if supplement_path is None:
            supplement_path = os.path.join(_DATA, 'srwac_supplement_v2.json')
        if augment_np_path is None:
            augment_np_path = os.path.join(_DATA, 'srwac_augment_np.json')
        self.cg = CandidateGenerator(
            srlex_path,
            supplement_path=supplement_path,
            augment_np_path=augment_np_path,
            extra_dict_path=extra_dict_path)
        self._disamb = POSDisambiguator(self.cg, tagger=None)

    def restore(self, word, msd=''):
        """Vrati (moguće) restauriranu reč; original ako nema dokaza."""
        if not needs_restoration(word):
            return word
        return self._disamb._restore_word_pos(word, msd)
