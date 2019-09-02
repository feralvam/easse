from collections import Counter
from functools import lru_cache
from typing import List

from ts.text import get_ngrams
from ts.utils.helpers import yield_lines_in_parallel, safe_division
from ts.preprocess import normalize_sentence

from easse.utils.preprocessing import normalize


# Helper functions
def f1_measure(precision, recall):
    return 2 * safe_division(precision * recall, precision + recall)


def bi(val):
    # Binary indicator
    if val > 0:
        return 1
    return 0


def average(values):
    return sum(values) / len(values)


class Stats:
    def __init__(self,
                 get_precision_numerator,
                 get_precision_denominator,
                 get_recall_numerator=None,
                 get_recall_denominator=None):
        self.get_precision_numerator = get_precision_numerator
        self.get_precision_denominator = get_precision_denominator
        self.get_recall_numerator = get_recall_numerator
        self.get_recall_denominator = get_recall_denominator
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0

    def __add__(self, other):
        if other == 0:
            return self
        self.precision_numerator += other.precision_numerator
        self.precision_denominator += other.precision_denominator
        self.recall_numerator += other.recall_numerator
        self.recall_denominator += other.recall_denominator
        return self

    def add(self, I, O, R, r):
        self.precision_numerator += self.get_precision_numerator(I, O, R, r)
        self.precision_denominator += self.get_precision_denominator(I, O, R, r)
        # Wei Xu's paper advises to not take recall into account
        if self.get_recall_numerator is not None:
            self.recall_numerator += self.get_recall_numerator(I, O, R, r)
            self.recall_denominator += self.get_recall_denominator(I, O, R, r)

    def get_precision(self):
        return safe_division(self.precision_numerator, self.precision_denominator)

    def get_recall(self):
        return safe_division(self.recall_numerator, self.recall_denominator)

    def get_f1_measure(self):
        return f1_measure(self.get_precision(), self.get_recall())

    def __str__(self):
        return (f'{self.__class__.__name__}('
                f'precision={self.precision_numerator}/{self.precision_denominator}, '
                f'recall={self.recall_numerator}/{self.recall_denominator})')


class SARIScorer:
    '''SARI from [Xu et al. 2016] Optimizing Statistical Machine Translation for Text Simplification'''
    def __init__(self, version='joshuastar'):
        self.set_formulas(version)
        # TODO : Do it by ngrams
        self.ngram_sizes = [1, 2, 3, 4]
        self.ngram_addition_stats = {}
        self.ngram_keep_stats = {}
        self.ngram_deletion_stats = {}
        for n in self.ngram_sizes:
            self.ngram_addition_stats[n] = Stats(self.addition_precision_numerator, self.addition_precision_denominator,
                                                 self.addition_recall_numerator, self.addition_recall_denominator)
            self.ngram_keep_stats[n] = Stats(self.keep_precision_numerator, self.keep_precision_denominator,
                                             self.keep_recall_numerator, self.keep_recall_denominator)
            self.ngram_deletion_stats[n] = Stats(self.deletion_precision_numerator, self.deletion_precision_denominator,
                                                 self.deletion_recall_numerator, self.deletion_recall_denominator)

    @lru_cache(maxsize=10**5)
    def _cached_add(self, orig_sent, sys_sent, ref_sents):
        for n in self.ngram_sizes:
            source_counter = Counter(get_ngrams(orig_sent, n))
            pred_counter = Counter(get_ngrams(sys_sent, n))
            refs_counter = Counter([ngram for ref_sent in ref_sents for ngram in get_ngrams(ref_sent, n)])
            nb_refs = len(ref_sents)
            ngrams = set(list(source_counter.keys()) + list(pred_counter.keys()) + list(refs_counter.keys()))
            for ngram in ngrams:
                I = source_counter[ngram]  # noqa: E741
                O = pred_counter[ngram]  # noqa: E741
                R = refs_counter[ngram]
                r = nb_refs
                self.ngram_addition_stats[n].add(I, O, R, r)
                self.ngram_keep_stats[n].add(I, O, R, r)
                self.ngram_deletion_stats[n].add(I, O, R, r)


    def add(self, orig_sent, sys_sent, ref_sents):
        assert type(ref_sents) == list
        self._cached_add(orig_sent, sys_sent, tuple(ref_sents))

    def get_addition_score(self):
        return average([self.ngram_addition_stats[n].get_f1_measure() for n in self.ngram_sizes])

    def get_keep_score(self):
        return average([self.ngram_keep_stats[n].get_f1_measure() for n in self.ngram_sizes])

    def get_deletion_score(self):
        if self.deletion_recall_numerator is not None:
            return average([self.ngram_deletion_stats[n].get_f1_measure() for n in self.ngram_sizes])
        else:
            # Don't take recall into account as advised in Wei Xu's paper
            return average([self.ngram_deletion_stats[n].get_precision() for n in self.ngram_sizes])

    def score(self):
        return average([self.get_addition_score(), self.get_keep_score(), self.get_deletion_score()])

    def set_formulas(self, version):
        method_name = f'set_{version}_formulas'
        assert hasattr(self, method_name), f'SARI version "{version}" incorrect. No method called "{method_name}()"'
        getattr(self, method_name)()

    # In all the following the following variables represent:
    # I -> #g(I): Number of occurences of the current ngram in the input / source sentence
    # O -> #g(O): Number of occurences of the current ngram in the output / predicted sentence
    # R -> #g(R): Number of occurences of the current ngram in all the reference sentences
    # r: Number of reference sentences
    def set_cocoxu_formulas(self):
        # These formulas match the Cocoxu implementation from:
        # https://github.com/cocoxu/simplification/blob/master/SARI.py
        # #Criticism Mismatch between what is said in the paper and what is implemented here.
        # Addition
        self.addition_precision_numerator = lambda I, O, R, r: min(bi(bi(O) - bi(I)), bi(R))
        self.addition_precision_denominator = lambda I, O, R, r: bi(bi(O) - bi(I))
        self.addition_recall_numerator = lambda I, O, R, r: self.addition_precision_numerator(I, O, R, r)
        self.addition_recall_denominator = lambda I, O, R, r: bi(bi(R) - bi(I))
        # Keeps
        self.keep_precision_numerator = lambda I, O, R, r: min(bi(I), bi(O), safe_division((R / r), min(I, O)))
        self.keep_precision_denominator = lambda I, O, R, r: min(bi(I), bi(O))
        self.keep_recall_numerator = lambda I, O, R, r: safe_division(min(I, O, R / r), min(I, R / r))
        self.keep_recall_denominator = lambda I, O, R, r: min(bi(I), bi(R / r))
        # Deletions
        # This formula is a complete non-sense, it is not even close to what is written in the paper
        self.deletion_precision_numerator = lambda I, O, R, r: safe_division(max(0,
                                                                                 max(0, r * I - r * O) - R),
                                                                             max(0, r * I - r * O))  # noqa: E501
        self.deletion_precision_denominator = lambda I, O, R, r: bi(max(0, I - O))

    def set_joshua_formulas(self):
        # Joshua toolkit SARI.java implementation from:
        # https://github.com/apache/incubator-joshua/blob/master/src/main/java/org/apache/joshua/metrics/SARI.java
        # #Criticism This implementation has a bug, they multiply keepCandCorrectNgram
        # (which is a double) by 1,000,000 in order to store it as an int and
        # then divide it by 1,000,000 again to retrieve the double value (very
        # dirty). The problem is that keepCandCorrectNgram is summed over all
        # sentences after being multiplied by 1,000,000, and therefore exceeds
        # the maximum int value of +2 147 483 647 in Java which causes its
        # value to become negative and completely invalidates the score.
        # Additions
        self.addition_precision_numerator = lambda I, O, R, r: min(bi(bi(O) - bi(I)), bi(R))
        self.addition_precision_denominator = lambda I, O, R, r: bi(bi(O) - bi(I))
        self.addition_recall_numerator = lambda I, O, R, r: self.addition_precision_numerator(I, O, R, r)
        self.addition_recall_denominator = lambda I, O, R, r: bi(bi(R) - bi(I))
        # Keeps
        self.keep_precision_numerator = lambda I, O, R, r: safe_division(min(r * I, r * O, R), min(r * I, r * O))
        self.keep_precision_denominator = lambda I, O, R, r: int(safe_division(min(r * I, r * O, R), min(r * I, R)))
        # Weird: the Joshua implementation uses precision_denominator instead of precision_numerator.
        self.keep_recall_numerator = lambda I, O, R, r: self.keep_precision_denominator(I, O, R, r)
        self.keep_recall_denominator = lambda I, O, R, r: bi(min(r * I, R))
        # Deletions
        self.deletion_precision_numerator = lambda I, O, R, r: min(max(0, r * I - r * O), max(0, r * I - R))
        self.deletion_precision_denominator = lambda I, O, R, r: max(0, r * I - r * O)

    def set_joshuastar_formulas(self):
        # STAR.java implentation in Joshua toolkit by Wei Xu from:
        # https://github.com/cocoxu/simplification
        # Can be downloaded here:
        # https://drive.google.com/file/d/0B1P1xW5xNISsdXdoX1RQNmVSSkE/view?usp=sharing
        # #Criticism The formulas in this implementation don't fit what is said in the paper.
        # The deletion recall is taken into account whereas it shouldn't !
        # Additions
        self.addition_precision_numerator = lambda I, O, R, r: min(bi(bi(O) - bi(I)), bi(R))
        self.addition_precision_denominator = lambda I, O, R, r: bi(bi(O) - bi(I))
        self.addition_recall_numerator = lambda I, O, R, r: self.addition_precision_numerator(I, O, R, r)
        self.addition_recall_denominator = lambda I, O, R, r: bi(bi(R) - bi(I))
        # Keeps
        self.keep_precision_numerator = lambda I, O, R, r: min(r * I, r * O, R)
        self.keep_precision_denominator = lambda I, O, R, r: min(r * I, r * O)
        self.keep_recall_numerator = lambda I, O, R, r: self.keep_precision_numerator(I, O, R, r)
        self.keep_recall_denominator = lambda I, O, R, r: min(r * I, R)
        # Deletions
        self.deletion_precision_numerator = lambda I, O, R, r: min(max(0, r * I - r * O), max(0, r * I - R))
        self.deletion_precision_denominator = lambda I, O, R, r: max(0, r * I - r * O)
        self.deletion_recall_numerator = lambda I, O, R, r: self.deletion_precision_numerator(I, O, R, r)
        self.deletion_recall_denominator = lambda I, O, R, r: max(0, r * I - R)


def sari_preprocess(orig_sent, sys_sent, ref_sents, normalize, lower):
    if normalize:
        # TODO: #Criticism The source sentence should be normalized ideally.
        # This is to reproduce the results in https://github.com/XingxingZhang/dress
        # orig_sent = normalize_sentence(orig_sent)
        sys_sent = normalize_sentence(sys_sent)
        ref_sents = [normalize_sentence(ref_sent) for ref_sent in ref_sents]
    if lower:
        orig_sent = orig_sent.lower()
        sys_sent = sys_sent.lower()
        ref_sents = [ref_sent.lower() for ref_sent in ref_sents]
    return orig_sent, sys_sent, ref_sents


def get_sari_sentence(orig_sent, sys_sent, ref_sents, version='joshuastar', normalize=False, lower=False):
    scorer = SARIScorer(version=version)
    orig_sent, sys_sent, ref_sents = sari_preprocess(orig_sent, sys_sent, ref_sents, normalize, lower)
    scorer.add(orig_sent, sys_sent, ref_sents)
    return scorer.score()


def corpus_sari(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                lowercase: bool = False, tokenizer: str = '13a', version: str = 'joshuastar'):
    scorer = SARIScorer(version=version)
    for orig_sent, sys_sent, *ref_sents in zip(orig_sents, sys_sents, *refs_sents):
        # TODO: The source sentence should be normalized ideally.
        # This is to reproduce the results in https://github.com/XingxingZhang/dress
        # orig_sent = normalize(orig_sent, lowercase, tokenizer)
        sys_sent = normalize(sys_sent, lowercase, tokenizer)
        ref_sents = [normalize(ref_sent, lowercase, tokenizer) for ref_sent in ref_sents]
        scorer.add(orig_sent, sys_sent, ref_sents)
    return 100 * scorer.score()


def get_sari_intermediate_scores(source_file, pred_file, ref_files, version='joshuastar', normalize=False, lower=False):
    scorer = SARIScorer(version=version)
    for orig_sent, sys_sent, *ref_sents in yield_lines_in_parallel([source_file, pred_file] + ref_files):
        orig_sent, sys_sent, ref_sents = sari_preprocess(orig_sent, sys_sent, ref_sents, normalize, lower)
        scorer.add(orig_sent, sys_sent, ref_sents)
    return scorer.get_addition_score(), scorer.get_keep_score(), scorer.get_deletion_score()
