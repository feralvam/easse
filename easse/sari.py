# Re-implementation from

from collections import Counter
from typing import List
import sacrebleu
import sacremoses
from nltk.tokenize.nist import NISTTokenizer

# n-gram order. Don't change this.
NGRAM_ORDER = 4


# def compute_precision_recall(correct, total_precision, total_recall):
#     precision = 0
#     if total_precision > 0:
#         precision = correct / total_precision
#
#     recall = 0
#     if total_recall > 0:
#         recall = correct / total_recall
#
#     return precision, recall
#
#
# def compute_f1(precision, recall):
#     f1 = 0
#     if precision > 0 or recall > 0:
#         f1 = 2 * precision * recall / (precision + recall)
#     return f1
#
#
# def compute_sari(add_hyp_correct, add_hyp_total, add_ref_total,
#                  keep_hyp_correct, keep_hyp_total, keep_ref_total,
#                  del_hyp_correct, del_hyp_total, del_ref_total,
#                  corpus_level=True):
#     add_precision = [0] * NGRAM_ORDER
#     add_recall = [0] * NGRAM_ORDER
#     keep_precision = [0] * NGRAM_ORDER
#     keep_recall = [0] * NGRAM_ORDER
#     del_precision = [0] * NGRAM_ORDER
#     del_recall = [0] * NGRAM_ORDER
#
#     for n in range(NGRAM_ORDER):
#         add_precision[n], add_recall[n] = compute_precision_recall(add_hyp_correct[n], add_hyp_total[n],
#                                                                    add_ref_total[n])
#         keep_precision[n], keep_recall[n] = compute_precision_recall(keep_hyp_correct[n], keep_hyp_total[n],
#                                                                      keep_ref_total[n])
#         del_precision[n], del_recall[n] = compute_precision_recall(del_hyp_correct[n], del_hyp_total[n],
#                                                                    del_ref_total[n])
#
#     avg_add_precision = sum(add_precision) / NGRAM_ORDER
#     avg_add_recall = sum(add_recall) / NGRAM_ORDER
#     avg_keep_precision = sum(keep_precision) / NGRAM_ORDER
#     avg_keep_recall = sum(keep_recall) / NGRAM_ORDER
#     avg_del_precision = sum(del_precision) / NGRAM_ORDER
#     avg_del_recall = sum(del_recall) / NGRAM_ORDER
#
#     add_f1 = compute_f1(avg_add_precision, avg_add_recall)
#
#     keep_f1 = compute_f1(avg_keep_precision, avg_keep_recall)
#
#     if corpus_level:
#         del_score = compute_f1(avg_del_precision, avg_del_recall)
#     else:
#         del_score = avg_del_precision
#
#     sari_score = (add_f1 + keep_f1 + del_score) / 3
#
#     return sari_score


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> List[Counter]:
    ngrams_per_order = []
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        ngrams = Counter()
        for i in range(0, len(tokens) - n + 1):
            ngram = ' '.join(tokens[i: i + n])
            ngrams[ngram] += 1
        ngrams_per_order.append(ngrams)

    return ngrams_per_order


def multiply_counter(c, v):
    c_aux = Counter()
    for k in c.keys():
        c_aux[k] = c[k] * v
    return c_aux


def compute_ngram_stats(orig_sents: List[str], hyp_sents: List[str], refs_sents: List[List[str]]):
    add_hyp_correct = [0] * NGRAM_ORDER
    add_hyp_total = [0] * NGRAM_ORDER
    add_ref_total = [0] * NGRAM_ORDER

    keep_hyp_correct = [0] * NGRAM_ORDER
    keep_hyp_total = [0] * NGRAM_ORDER
    keep_ref_total = [0] * NGRAM_ORDER

    del_hyp_correct = [0] * NGRAM_ORDER
    del_hyp_total = [0] * NGRAM_ORDER
    del_ref_total = [0] * NGRAM_ORDER

    for orig_sent, hyp_sent, ref_sents in zip(orig_sents, hyp_sents, refs_sents):
        orig_ngrams = extract_ngrams(orig_sent)
        hyp_ngrams = extract_ngrams(hyp_sent)

        refs_ngrams = [Counter() for _ in range(NGRAM_ORDER)]
        for ref_sent in ref_sents:
            ref_ngrams = extract_ngrams(ref_sent)
            for n in range(NGRAM_ORDER):
                refs_ngrams[n] += ref_ngrams[n]

        num_refs = len(ref_sents)
        for n in range(NGRAM_ORDER):
            # ADD
            # added by the hypothesis (binary)
            hyp_and_not_orig = set(hyp_ngrams[n]) - set(orig_ngrams[n])
            add_hyp_total[n] += len(hyp_and_not_orig)
            # added by the references (binary)
            ref_and_not_orig = set(refs_ngrams[n]) - set(orig_ngrams[n])
            add_ref_total[n] += len(ref_and_not_orig)
            # added correctly (binary)
            add_hyp_correct[n] += len(hyp_and_not_orig & set(refs_ngrams[n]))

            # KEEP
            # kept by the hypothesis (weighted)
            orig_and_hyp = multiply_counter(orig_ngrams[n], num_refs) & multiply_counter(hyp_ngrams[n], num_refs)
            keep_hyp_total[n] += sum(orig_and_hyp.values())
            # kept by the references (weighted)
            orig_and_ref = multiply_counter(orig_ngrams[n], num_refs) & refs_ngrams[n]
            keep_ref_total[n] += sum(orig_and_ref.values())
            # kept correctly?
            keep_hyp_correct[n] += sum((orig_and_hyp & orig_and_ref).values())

            # DELETE
            # deleted by the hypothesis (weighted)
            orig_and_not_hyp = multiply_counter(orig_ngrams[n], num_refs) - multiply_counter(hyp_ngrams[n], num_refs)
            del_hyp_total[n] += sum(orig_and_not_hyp.values())
            # deleted by the references (weighted)
            orig_and_not_ref = multiply_counter(orig_ngrams[n], num_refs) - refs_ngrams[n]
            del_ref_total[n] += sum(orig_and_not_ref.values())
            # deleted correctly
            del_hyp_correct[n] += sum((orig_and_not_hyp & orig_and_not_ref).values())

    return add_hyp_correct, add_hyp_total, add_ref_total, \
           keep_hyp_correct, keep_hyp_total, keep_ref_total,\
           del_hyp_correct, del_hyp_total, del_ref_total


def compute_f1(hyp_correct, hyp_total, ref_total):
    precision = 0.0
    if hyp_total > 0:
        precision = hyp_correct / hyp_total

    recall = 0.0
    if ref_total > 0:
        recall = hyp_correct / ref_total

    f1 = 0.0
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_sari(add_hyp_correct, add_hyp_total, add_ref_total,
                 keep_hyp_correct, keep_hyp_total, keep_ref_total,
                 del_hyp_correct, del_hyp_total, del_ref_total,
                 corpus_level=True):

    sari_score = 0
    for n in range(NGRAM_ORDER):
        _, _, add_f1_ngram = compute_f1(add_hyp_correct[n], add_hyp_total[n], add_ref_total[n])
        _, _, keep_f1_ngram = compute_f1(keep_hyp_correct[n], keep_hyp_total[n], keep_ref_total[n])
        if corpus_level:
            _, _, del_score_ngram = compute_f1(del_hyp_correct[n], del_hyp_total[n], del_ref_total[n])
        else:
            del_score_ngram, _, _ = compute_f1(del_hyp_correct[n], del_hyp_total[n], del_ref_total[n])

        sari_score += add_f1_ngram / NGRAM_ORDER
        sari_score += keep_f1_ngram / NGRAM_ORDER
        sari_score += del_score_ngram / NGRAM_ORDER

    return sari_score / 3


def normalize(sent, lowercase: bool, tokenizer: str):
    if lowercase:
        sent = sent.lower()

    if tokenizer == "13a":
        normalized_sent = sacrebleu.tokenize_13a(sent)
    elif tokenizer == "intl":
        normalized_sent = sacrebleu.tokenize_v14_international(sent)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sent, return_str=True)
    else:
        normalized_sent = sent

    return normalized_sent


def sari_corpus(orig_sents: List[str], hyp_sents: List[str], refs_sents: List[List[str]],
                lowercase: bool = False, tokenizer: str = '13a'):

    # orig_sents = [normalize(sent, lowercase, tokenizer) for sent in orig_sents]
    hyp_sents = [normalize(sent, lowercase, tokenizer) for sent in hyp_sents]
    refs_sents = [[normalize(sent, lowercase, tokenizer) for sent in ref_sents] for ref_sents in refs_sents]

    stats = compute_ngram_stats(orig_sents, hyp_sents, refs_sents)
    return compute_sari(*stats, corpus_level=True)


def sari_sentence(orig_sent: str, hyp_sent: str, ref_sents: List[str]):
    stats = compute_ngram_stats([orig_sent], [hyp_sent], [ref_sents])
    return compute_sari(*stats, corpus_level=False)
