import pytest
import numpy as np

from easse.compression import corpus_f1_token


def test_corpus_macro_avg_f1_token_single_reference():
    ref_sents = ["A mother are accused of committing one local robbery"]
    sys_sent = "A mother and her teenage son are accused of committing one robbery"

    f1_token = corpus_f1_token([sys_sent], [ref_sents])

    correct = ['A', 'mother', 'are', 'accused', 'of', 'committing', 'one', 'robbery']

    precision = len(correct) / len(sys_sent.split())
    recall = len(correct) / len(ref_sents[0].split())
    f1 = 2 * precision * recall / (precision + recall)
    f1 = 100.0 * f1

    assert f1_token == pytest.approx(f1)


def test_corpus_macro_avg_f1_token_multiple_references():
    ref_sents = ["A mother are accused of committing one local robbery", "A mother accused of committing robbery"]
    sys_sent = "A mother and her teenage son are accused of committing one robbery"

    f1_token = corpus_f1_token([sys_sent], [ref_sents])

    correct_ref1 = ['A', 'mother', 'are', 'accused', 'of', 'committing', 'one', 'robbery']
    precision_ref1 = len(correct_ref1) / len(sys_sent.split())
    recall_ref1 = len(correct_ref1) / len(ref_sents[0].split())
    f1_ref1 = 2 * precision_ref1 * recall_ref1 / (precision_ref1 + recall_ref1)

    correct_ref2 = ['A', 'mother', 'accused', 'of', 'committing', 'robbery']
    precision_ref2 = len(correct_ref2) / len(sys_sent.split())
    recall_ref2 = len(correct_ref2) / len(ref_sents[1].split())
    f1_ref2 = 2 * precision_ref2 * recall_ref2 / (precision_ref2 + recall_ref2)

    f1 = 100.0 * np.max([f1_ref1, f1_ref2])

    assert f1_token == pytest.approx(f1)
