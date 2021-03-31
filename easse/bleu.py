"""
Implements the evaluation metrics based on BLEU score
"""

import numpy as np
from typing import List

import sacrebleu

import easse.utils.preprocessing as utils_prep


def corpus_bleu(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    smooth_method: str = 'exp',
    smooth_value: float = None,
    force: bool = True,
    lowercase: bool = False,
    tokenizer: str = '13a',
    use_effective_order: bool = False,
):

    sys_sents = [utils_prep.normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[utils_prep.normalize(sent, lowercase, tokenizer) for sent in ref_sents] for ref_sents in refs_sents]

    return sacrebleu.corpus_bleu(
        sys_sents,
        refs_sents,
        smooth_method,
        smooth_value,
        force,
        lowercase=False,
        tokenize='none',
        use_effective_order=use_effective_order,
    ).score


def sentence_bleu(
    sys_sent: str,
    ref_sents: List[str],
    smooth_method: str = 'floor',
    smooth_value: float = None,
    lowercase: bool = False,
    tokenizer: str = '13a',
    use_effective_order: bool = True,
):

    return corpus_bleu(
        [sys_sent],
        [[ref] for ref in ref_sents],
        smooth_method,
        smooth_value,
        force=True,
        lowercase=lowercase,
        tokenizer=tokenizer,
        use_effective_order=use_effective_order,
    )


def corpus_averaged_sentence_bleu(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    smooth_method: str = 'floor',
    smooth_value: float = None,
    lowercase: bool = False,
    tokenizer: str = '13a',
    use_effective_order: bool = True,
):

    scores = []
    for sys_sent, *ref_sents in zip(sys_sents, *refs_sents):
        scores.append(
            sentence_bleu(
                sys_sent,
                ref_sents,
                smooth_method,
                smooth_value,
                lowercase=lowercase,
                tokenizer=tokenizer,
                use_effective_order=use_effective_order,
            )
        )
    return np.mean(scores)
