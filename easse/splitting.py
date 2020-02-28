"""
Implements the evaluation metrics use in the split-and-rephrase task.

It is mostly the same as https://github.com/google-research/google-research/blob/master/wiki_split_bleu_eval/score_lib.py
with a few modifications:
- BLEU scores are computed using sacreBLEU instead of NLTK

"""

import itertools
import numpy as np

import sacrebleu


def corpus_macro_avg_sent_bleu(sys_sents, refs_sents, lowercase: bool = False, tokenizer: str = '13a'):
    scores = []
    for sys_sent, *ref_sents in zip(sys_sents, *refs_sents):
        scores.append(sacrebleu.corpus_bleu(sys_sent, [[x] for x in ref_sents],
                                            smooth_method='floor', use_effective_order=True, force=True,
                                            tokenize=tokenizer, lowercase=lowercase).score)
    return np.mean(scores)


def num_tokens(s):
    return len(s.split())


def sys_length_statistics(data):
    """Updates results with simple length-based statistics.

  parcels / input_sentence - (S/C metric in paper) macro averaged num
  tokens per parcel (Tokens/S in paper)

  Example of an item in data: ['parcel1 here .', 'parcel 2 here .']

  Args:
    data: list of parcel lists.

  Returns:
    dictionary of results
  """

    results = {}

    # Average number of parcels per decomposed instance.
    parcel_counts = [len(instance) for instance in data]
    results['sys_lengths.simple_per_complex'] = np.mean(parcel_counts)

    # Token counts.
    token_counts = []
    for instance in data:
        token_counts.append([num_tokens(parcel) for parcel in instance])

    # Macro averaged number of tokens per parcel.
    results['sys_lengths.tokens_per_simple'] = np.mean(
        [np.mean(counts) for counts in token_counts])

    # Micro averaged number of tokens per parcel.
    total_tokens = np.sum(list(itertools.chain.from_iterable(token_counts)))
    total_parcels = np.sum(parcel_counts)
    results['sys_lengths.tokens_per_simple_micro'] = total_tokens / total_parcels

    return results


def ref_length_statistics(data):
    """Updates results with simple length-based statistics over multi-ref data.

  Example of an item in data: [['parcel1 here .', 'parcel 2 here .'], [alt..]]

  Args:
    data: list of list of parcel lists.

  Returns:
    dictionary of results
  """

    results = {}

    # Macro-average number of parcels per decomposed instance.
    parcel_counts = []
    for instance in data:
        parcel_counts.append([len(analysis) for analysis in instance])

    results['ref_lengths.simple_per_complex'] = np.mean(
        [np.mean(counts) for counts in parcel_counts])

    # Token counts.
    token_counts = []
    for instance in data:
        instance_counts = []
        for analysis in instance:
            instance_counts.append([num_tokens(parcel) for parcel in analysis])
        token_counts.append(instance_counts)

    # Macro averaged number of tokens per parcel.
    token_means_per_analysis = []
    for instance in token_counts:
        token_means_per_analysis.append(
            [np.mean(analysis_counts) for analysis_counts in instance])

    results['ref_lengths.tokens_per_simple'] = np.mean(
        [np.mean(counts) for counts in token_means_per_analysis])

    return results
