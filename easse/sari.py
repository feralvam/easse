from collections import Counter
from typing import List
import easse.utils.preprocessing as utils_prep

NGRAM_ORDER = 4


def compute_precision_recall(sys_correct, sys_total, ref_total):
    precision = 0.0
    if sys_total > 0:
        precision = sys_correct / sys_total

    recall = 0.0
    if ref_total > 0:
        recall = sys_correct / ref_total

    return precision, recall


def compute_f1(precision, recall):
    f1 = 0.0
    if precision > 0 or recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_micro_sari(
    add_hyp_correct,
    add_hyp_total,
    add_ref_total,
    keep_hyp_correct,
    keep_hyp_total,
    keep_ref_total,
    del_hyp_correct,
    del_hyp_total,
    del_ref_total,
    use_f1_for_deletion=True,
):
    """
    This is the version described in the original paper. Follows the equations.
    """
    add_precision = [0] * NGRAM_ORDER
    add_recall = [0] * NGRAM_ORDER
    keep_precision = [0] * NGRAM_ORDER
    keep_recall = [0] * NGRAM_ORDER
    del_precision = [0] * NGRAM_ORDER
    del_recall = [0] * NGRAM_ORDER

    for n in range(NGRAM_ORDER):
        add_precision[n], add_recall[n] = compute_precision_recall(
            add_hyp_correct[n], add_hyp_total[n], add_ref_total[n]
        )
        keep_precision[n], keep_recall[n] = compute_precision_recall(
            keep_hyp_correct[n], keep_hyp_total[n], keep_ref_total[n]
        )
        del_precision[n], del_recall[n] = compute_precision_recall(
            del_hyp_correct[n], del_hyp_total[n], del_ref_total[n]
        )

    avg_add_precision = sum(add_precision) / NGRAM_ORDER
    avg_add_recall = sum(add_recall) / NGRAM_ORDER
    avg_keep_precision = sum(keep_precision) / NGRAM_ORDER
    avg_keep_recall = sum(keep_recall) / NGRAM_ORDER
    avg_del_precision = sum(del_precision) / NGRAM_ORDER
    avg_del_recall = sum(del_recall) / NGRAM_ORDER

    add_f1 = compute_f1(avg_add_precision, avg_add_recall)

    keep_f1 = compute_f1(avg_keep_precision, avg_keep_recall)

    if use_f1_for_deletion:
        del_score = compute_f1(avg_del_precision, avg_del_recall)
    else:
        del_score = avg_del_precision

    return add_f1, keep_f1, del_score


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> List[Counter]:
    ngrams_per_order = []
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        ngrams = Counter()
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams[ngram] += 1
        ngrams_per_order.append(ngrams)

    return ngrams_per_order


def multiply_counter(c, v):
    c_aux = Counter()
    for k in c.keys():
        c_aux[k] = c[k] * v

    return c_aux


def compute_ngram_stats(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]]):
    """
    Input:
    orig_sents: list of original sentences (len = n_samples)
    sys_sents: list of system sentences (len = n_samples)
    refs_sents: list of list of reference sentences (shape = (n_references, n_samples))
    """
    assert len(orig_sents) == len(
        sys_sents
    ), "Original sentences and system sentences don't have the same number of samples"
    assert all(
        len(ref_sents) == len(orig_sents) for ref_sents in refs_sents
    ), "Reference sentences don't have the shape (n_references, n_samples)"
    add_sys_correct = [0] * NGRAM_ORDER
    add_sys_total = [0] * NGRAM_ORDER
    add_ref_total = [0] * NGRAM_ORDER

    keep_sys_correct = [0] * NGRAM_ORDER
    keep_sys_total = [0] * NGRAM_ORDER
    keep_ref_total = [0] * NGRAM_ORDER

    del_sys_correct = [0] * NGRAM_ORDER
    del_sys_total = [0] * NGRAM_ORDER
    del_ref_total = [0] * NGRAM_ORDER

    for orig_sent, sys_sent, *ref_sents in zip(orig_sents, sys_sents, *refs_sents):
        orig_ngrams = extract_ngrams(orig_sent)
        sys_ngrams = extract_ngrams(sys_sent)

        refs_ngrams = [Counter() for _ in range(NGRAM_ORDER)]
        for ref_sent in ref_sents:
            ref_ngrams = extract_ngrams(ref_sent)
            for n in range(NGRAM_ORDER):
                refs_ngrams[n] += ref_ngrams[n]

        num_refs = len(ref_sents)
        for n in range(NGRAM_ORDER):
            # ADD
            # added by the hypothesis (binary)
            sys_and_not_orig = set(sys_ngrams[n]) - set(orig_ngrams[n])
            add_sys_total[n] += len(sys_and_not_orig)
            # added by the references (binary)
            ref_and_not_orig = set(refs_ngrams[n]) - set(orig_ngrams[n])
            add_ref_total[n] += len(ref_and_not_orig)
            # added correctly (binary)
            add_sys_correct[n] += len(sys_and_not_orig & set(refs_ngrams[n]))

            # KEEP
            # kept by the hypothesis (weighted)
            orig_and_sys = multiply_counter(orig_ngrams[n], num_refs) & multiply_counter(sys_ngrams[n], num_refs)
            keep_sys_total[n] += sum(orig_and_sys.values())
            # kept by the references (weighted)
            orig_and_ref = multiply_counter(orig_ngrams[n], num_refs) & refs_ngrams[n]
            keep_ref_total[n] += sum(orig_and_ref.values())
            # kept correctly?
            keep_sys_correct[n] += sum((orig_and_sys & orig_and_ref).values())

            # DELETE
            # deleted by the hypothesis (weighted)
            orig_and_not_sys = multiply_counter(orig_ngrams[n], num_refs) - multiply_counter(sys_ngrams[n], num_refs)
            del_sys_total[n] += sum(orig_and_not_sys.values())
            # deleted by the references (weighted)
            orig_and_not_ref = multiply_counter(orig_ngrams[n], num_refs) - refs_ngrams[n]
            del_ref_total[n] += sum(orig_and_not_ref.values())
            # deleted correctly
            del_sys_correct[n] += sum((orig_and_not_sys & orig_and_not_ref).values())

    return (
        add_sys_correct,
        add_sys_total,
        add_ref_total,
        keep_sys_correct,
        keep_sys_total,
        keep_ref_total,
        del_sys_correct,
        del_sys_total,
        del_ref_total,
    )


def compute_precision_recall_f1(sys_correct, sys_total, ref_total):
    precision = 0.0
    if sys_total > 0:
        precision = sys_correct / sys_total

    recall = 0.0
    if ref_total > 0:
        recall = sys_correct / ref_total

    f1 = 0.0
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_macro_sari(
    add_sys_correct,
    add_sys_total,
    add_ref_total,
    keep_sys_correct,
    keep_sys_total,
    keep_ref_total,
    del_sys_correct,
    del_sys_total,
    del_ref_total,
    use_f1_for_deletion=True,
):
    """
    This is the version released as a JAVA implementation and which was used in their experiments,
    as stated by the authors: https://github.com/cocoxu/simplification/issues/8
    """
    add_f1 = 0.0
    keep_f1 = 0.0
    del_f1 = 0.0
    for n in range(NGRAM_ORDER):
        _, _, add_f1_ngram = compute_precision_recall_f1(add_sys_correct[n], add_sys_total[n], add_ref_total[n])
        _, _, keep_f1_ngram = compute_precision_recall_f1(keep_sys_correct[n], keep_sys_total[n], keep_ref_total[n])
        if use_f1_for_deletion:
            _, _, del_score_ngram = compute_precision_recall_f1(del_sys_correct[n], del_sys_total[n], del_ref_total[n])
        else:
            del_score_ngram, _, _ = compute_precision_recall_f1(del_sys_correct[n], del_sys_total[n], del_ref_total[n])
        add_f1 += add_f1_ngram / NGRAM_ORDER
        keep_f1 += keep_f1_ngram / NGRAM_ORDER
        del_f1 += del_score_ngram / NGRAM_ORDER
    return add_f1, keep_f1, del_f1


def get_corpus_sari_operation_scores(
    orig_sents: List[str],
    sys_sents: List[str],
    refs_sents: List[List[str]],
    lowercase: bool = True,
    tokenizer: str = '13a',
    legacy=False,
    use_f1_for_deletion=True,
    use_paper_version=False,
):
    """
    Inputs:
    orig_sents: list of original sentences (len = n_samples)
    sys_sents: list of system sentences (len = n_samples)
    refs_sents: list of list of reference sentences (shape = (n_references, n_samples))
    legacy: Allows reproducing scores reported in previous work.
    It replicates a bug in the original JAVA implementation where only the system outputs and the reference sentences
    are further tokenized.
    In addition, it assumes that all sentences are already lowercased.
    """
    if legacy:
        lowercase = False
    else:
        orig_sents = [utils_prep.normalize(sent, lowercase, tokenizer) for sent in orig_sents]

    sys_sents = [utils_prep.normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[utils_prep.normalize(sent, lowercase, tokenizer) for sent in ref_sents] for ref_sents in refs_sents]

    stats = compute_ngram_stats(orig_sents, sys_sents, refs_sents)

    if not use_paper_version:
        add_score, keep_score, del_score = compute_macro_sari(*stats, use_f1_for_deletion=use_f1_for_deletion)
    else:
        add_score, keep_score, del_score = compute_micro_sari(*stats, use_f1_for_deletion=use_f1_for_deletion)
    return 100.0 * add_score, 100.0 * keep_score, 100.0 * del_score


def corpus_sari(*args, **kwargs):
    add_score, keep_score, del_score = get_corpus_sari_operation_scores(*args, **kwargs)
    return (add_score + keep_score + del_score) / 3
