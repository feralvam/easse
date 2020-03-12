"""
Implements the evaluation metrics use in the split-and-rephrase task.
"""

import numpy as np

import sacrebleu


def corpus_macro_avg_sent_bleu(sys_sents, refs_sents, lowercase: bool = False, tokenizer: str = '13a'):
    scores = []
    for sys_sent, *ref_sents in zip(sys_sents, *refs_sents):
        scores.append(sacrebleu.corpus_bleu(sys_sent, [[x] for x in ref_sents],
                                            smooth_method='floor', use_effective_order=True, force=True,
                                            tokenize=tokenizer, lowercase=lowercase).score)
    return np.mean(scores)
