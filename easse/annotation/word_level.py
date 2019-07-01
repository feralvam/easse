"""
This module contains functions for annotating simplification operations at the sentencelevel.
"""
import time


from typing import List
from operator import itemgetter

import multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from easse.aligner.corenlp_utils import syntactic_parse_texts, posTag
from easse.aligner.aligner import align

# =============================================================================
# Constants
# =============================================================================

SIMOP_LABELS = ['B-A', 'B-D', 'B-M', 'B-R', 'I-A', 'I-D', 'I-M', 'I-R']
ORIG_OPS_LABELS = ['D', 'M', 'R', 'C']

# =============================================================================
# Internal Functions
# =============================================================================


def _have_same_postag(src_index, ref_index, src_parse, ref_parse):
    """Check if two tokens in two parse trees have the same part-of-speech tag, given their indexes."""
    src_postags = posTag(src_parse)
    ref_postags = posTag(ref_parse)

    src_token_info = src_postags[src_index - 1]
    assert src_token_info[1] == src_index
    src_postag = src_token_info[3]

    ref_token_info = ref_postags[ref_index - 1]
    assert ref_token_info[1] == ref_index
    ref_postag = ref_token_info[3]

    return src_postag == ref_postag


def _improve_replace(src_annots, ref_annots, src_parse, ref_parse):
    """
    Apply a simple heuristic based on simplification operations labels, sentence positions and part-of-speech tags
    to improve the annotation of replacements.
    """

    for ref_token in ref_annots:
        # check that the token has been labeled as 'add'
        if ref_token['label'] == 'B-A':
            # find a token in source, with the same index and labeled as 'delete'
            same_index_in_src = [src_token for src_token in src_annots if
                                 src_token['index'] == ref_token['index'] and src_token['label'] == 'B-D']
            if same_index_in_src:  # it is not an empty list
                src_token = same_index_in_src[0]  # there is always one at the most
                # check that both tokens have the same part of speech tag
                same_postag = _have_same_postag(src_token['index'], ref_token['index'], src_parse, ref_parse)
                if same_postag:
                    src_token['label'] = 'B-R'  # the token in the source must be labeled as replace
                    ref_token['label'] = 'O'  # the token in the reference now has no label
                    src_token['replace'] = [ref_token['word']]  # include the information of the replacement


def _label_move(src_annots, ref_annots, aligns):
    """
    Annotate 'move' checking if the relative index of a token in source changes in reference, considering  
    preceding deletions, additions and multi-token replacements.
    """

    shift_left = 0
    for src_token in src_annots:
        # check if the token has been labeled to be deleted or as part of a replace or rewrite
        if src_token['label'] in ['B-D', 'I-D', 'I-R']:
            shift_left += 1
        else:
            # get its position in the reference (using the word alignments)
            ref_index_list = [ref_index for src_index, ref_index in aligns if src_index == src_token['index']]
            if ref_index_list:
                ref_index = ref_index_list[0]
            else:
                ref_index = src_token['index']

            # count the number 'add' to the reference up until the new position of the source token
            shift_right = 0
            for ref_token in ref_annots:
                if ref_token['index'] < ref_index:
                    if ref_token['label'] in ['B-A', 'I-A']:
                        shift_right += 1
                else:
                    break

            # check if the token needs to be moved
            if (src_token['index'] - shift_left + shift_right) != ref_index:
                if src_token['label'] == 'O':  # the token is kept and moved
                    new_label = 'B-M'
                else:
                    new_label = src_token['label']  # in any other case, the label stays the same
                src_token['label'] = new_label
                src_token['move'] = ref_index


def _label_delete_replace(src, ref, aligns):
    """Annotate deletions and replacements in the source sentence."""
    src_annots = []
    for token_index, token_word in enumerate(src, start=1):
        src_token = {'index': token_index, 'word': token_word, 'label': ''}
        # get the indexes of all the words in the reference to which the current token in source is aligned to
        aligns_list = [ref_index for src_index, ref_index in aligns if src_index == token_index]
        # check if the token is aligned
        if aligns_list:
            # check if it has been aligned to only one token and if they are exactly the same
            if len(aligns_list) == 1 and token_word.lower() == ref[aligns_list[0] - 1].lower():
                # it is a 'keep'
                src_token['label'] = 'O'
            else:
                # it is a 'replacement'
                src_token['label'] = 'B-R'
                src_token['replace'] = []
                # recover all the tokens in reference for which this token in source is replaced
                aligns_list.sort()
                for ref_index in aligns_list:
                    src_token['replace'].append(ref[ref_index - 1])
        else:
            # label as delete
            src_token['label'] = 'B-D'

        src_annots.append(src_token)

    src_annots = sorted(src_annots, key=itemgetter('index'))

    return src_annots


def _label_add_replace(ref, aligns, src_annots):
    """Annotate additions in the reference sentence. Improve replacements annotation in the source sentence."""

    ref_annots = []
    for token_index, token_word in enumerate(ref, start=1):
        ref_token = {'index': token_index, 'word': token_word, 'label': ''}
        # get the indexes of all the tokens in the source to which the current token in reference is aligned
        aligns_list = [src_index for src_index, ref_index in aligns if ref_index == token_index]
        # check if the token is aligned
        if aligns_list:
            # it is the replacement of some word(s) in the source
            ref_token['label'] = 'O'  # the token in the reference has no label
            if len(aligns_list) > 1:
                # it is the replacement of a phrase in source, so the source token annotations should be changed
                aligns_list.sort()
                for i in range(1, len(aligns_list)):  # token 0 already has 'B-R' because of label_delete_replace
                    src_index = aligns_list[i]
                    src_token = [src_token for src_token in src_annots if src_token['index'] == src_index][0]
                    src_token['label'] = 'I-R'
                    src_token['replace'] = []  # token with 'B-R' has all the replacement tokens
        else:
            # label as 'add'
            ref_token['label'] = 'B-A'

        ref_annots.append(ref_token)

    ref_annots = sorted(ref_annots, key=itemgetter('index'))

    return ref_annots


# =============================================================================
# Main Functions
# =============================================================================


def annotate_sentence(src, ref, aligns, src_parse, ref_parse):
    """Annotate all the simplification operations in the sentence pair src-ref."""
    # token-level delete, add and replace
    src_annots = _label_delete_replace(src, ref, aligns)
    ref_annots = _label_add_replace(ref, aligns, src_annots)

    # simple heuristic to improve token-level replacements
    _improve_replace(src_annots, ref_annots, src_parse, ref_parse)

    # token-level move
    _label_move(src_annots, ref_annots, aligns)

    return src_annots, ref_annots


def _calculate_coverage(gold_labels, auto_labels, labels):
    coverage_per_label = []
    for label in labels:
        num = auto_labels.count(label)
        total = gold_labels.count(label)
        coverage = 1.0
        if total > 0:
            coverage = 1.0 * num / total
        coverage_per_label.append(coverage)

    return np.asarray(coverage_per_label)


def _from_annots_to_labels(sent_annots, labels_to_include=SIMOP_LABELS, default_label='C'):
    labels = []
    for token in sent_annots:
        label = token['label'].split('-')[-1]
        if label not in labels_to_include:
            label = default_label
        labels.append(label)

    return labels


def analyse_operations_sentence(orig_sent, sys_sent, ref_sents, orig_parse, sys_parse, ref_parses):
    word_aligns_orig_sys = align(orig_parse, sys_parse)[0]
    orig_annots, sys_annots = annotate_sentence(orig_sent.split(), sys_sent.split(),
                                                word_aligns_orig_sys, orig_parse, sys_parse)

    orig_auto_labels = _from_annots_to_labels(orig_annots, ORIG_OPS_LABELS, 'C')

    curr_sent_scores = []
    for ref_sent, ref_parse in zip(ref_sents, ref_parses):
        word_aligns_orig_ref = align(orig_parse, ref_parse)[0]

        orig_annots, ref_annots = annotate_sentence(orig_sent.split(), ref_sent.split(),
                                                    word_aligns_orig_ref, orig_parse, ref_parse)

        orig_silver_labels = _from_annots_to_labels(orig_annots, ORIG_OPS_LABELS, 'C')

        f1_per_label = f1_score(orig_silver_labels, orig_auto_labels, labels=ORIG_OPS_LABELS, average=None)

        curr_sent_scores.append(f1_per_label)

    return np.amax(curr_sent_scores, axis=0)


def corpus_analyse_operations(orig_sentences: List[str], sys_sentences: List[str], refs_sentences: List[List[str]],
                              as_str=False, verbose=False):
    orig_parses = syntactic_parse_texts(orig_sentences, verbose=verbose)
    sys_parses = syntactic_parse_texts(sys_sentences, verbose=verbose)
    refs_parses = [syntactic_parse_texts(ref_sents, verbose=verbose) for ref_sents in refs_sentences]

    time_start = time.time()

    all_parsers = [orig_parses] + [sys_parses] + refs_parses
    all_parsers = zip(*all_parsers)

    all_sentences = [orig_sentences] + [sys_sentences] + refs_sentences
    all_sentences = zip(*all_sentences)

    if verbose:
        print("Analysing word-level operations")

    pool = mp.Pool()
    corpus_scores = []
    for sentences, parsers in tqdm(zip(all_sentences, all_parsers), disable=(not verbose)):
        orig_sent, sys_sent, *ref_sents = sentences
        orig_parse, sys_parse, *ref_parses = parsers

        # sent_scores = pool.apply(analyse_operations_sentence,
        #                          args=(orig_sent, sys_sent, ref_sents, orig_parse, sys_parse, ref_parses))

        sent_scores = analyse_operations_sentence(orig_sent, sys_sent, ref_sents, orig_parse, sys_parse, ref_parses)

        corpus_scores.append(sent_scores)

    label_scores = np.mean(corpus_scores, axis=0)

    assert len(label_scores) == len(ORIG_OPS_LABELS)
    score_per_label = list(zip(ORIG_OPS_LABELS, label_scores))

    if as_str:
        score_per_label = " ".join([f"{label}={100.* score:.2f}" for label, score in score_per_label])

    time_end = time.time() - time_start
    print(f"Time: {time_end}")

    return score_per_label
