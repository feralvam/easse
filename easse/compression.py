from typing import List
from collections import Counter
import numpy as np

import easse.utils.preprocessing as utils_prep
from easse.sari import compute_precision_recall_f1


def corpus_f1_token(sys_sents: List[str], refs_sents: List[List[str]], lowercase: bool = True, tokenizer: str = '13a'):
    def find_correct_tokens(sys_tokens, ref_tokens):
        return list((Counter(sys_tokens) & Counter(ref_tokens)).elements())

    sys_sents = [utils_prep.normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[utils_prep.normalize(sent, lowercase, tokenizer) for sent in ref_sents] for ref_sents in refs_sents]

    f1_token_scores = []
    for sys_sent, *ref_sents in zip(sys_sents, *refs_sents):
        sys_tokens = sys_sent.split()
        sys_total = len(sys_tokens)

        candidate_f1_token_scores = []
        for ref_sent in ref_sents:
            ref_tokens = ref_sent.split()
            ref_total = len(ref_tokens)

            correct_tokens = len(find_correct_tokens(sys_tokens, ref_tokens))
            _, _, f1 = compute_precision_recall_f1(correct_tokens, sys_total, ref_total)
            candidate_f1_token_scores.append(f1)

        f1_token_scores.append(np.max(candidate_f1_token_scores))

    return 100.0 * np.mean(f1_token_scores)
