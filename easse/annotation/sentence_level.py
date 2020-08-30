from typing import List

from easse.annotation.word_level import WordOperationAnnotator
from easse.quality_estimation import count_sentence_splits


class SentenceOperationAnnotator:
    def __init__(
        self,
        align_tool: str = "simalign",
        simalign_method: str = "inter",
        with_clause: bool = False,
        lowercase: bool = False,
        tokenizer: str = "moses",
        verbose: bool = False,
    ):
        self._word_operation_annotator = WordOperationAnnotator(align_tool=align_tool,
                                                                simalign_method=simalign_method,
                                                                include_phrase_level=with_clause,
                                                                iob_labels=False,
                                                                lowercase=lowercase,
                                                                tokenizer=tokenizer,
                                                                verbose=verbose)
        self._with_clause = with_clause

    def identify_operations(
        self,
        orig_sentences: List[str],
        simp_sentences: List[str]
    ):
        orig_labels, simp_labels = self._word_operation_annotator.identify_operations(orig_sentences, simp_sentences)
        labels = []
        for orig_ops, simp_ops, orig_sent, simp_sent in zip(orig_labels, simp_labels, orig_sentences, simp_sentences):
            sent_labels = []
            sent_labels += list(set(orig_ops))
            sent_labels += list(set(simp_ops))
            if count_sentence_splits(orig_sent, simp_sent) > 1:
                sent_labels.append('S')
            final_sent_labels = []
            for label in sent_labels:
                if not self._with_clause and label.upper() in ['DC', 'AC', 'MC']:
                    label = label.upper()[0]
                final_sent_labels.append(label)
            labels.append(list(set(final_sent_labels)))
        return labels


    # def corpus_analyse_operations(self,
    #                                orig_sent,
    #                                sys_sent,
    #                                ref_sents,
    #                                orig_parse,
    #                                sys_parse,
    #                                ref_parses,
    #                                matching_method="itermax"
    #                                ):
    #
    #     word_aligns_orig_sys = self._word_aligner.get_word_aligns(orig_sent.split(), sys_sent.split())
    #
    #     word_aligns_orig_sys = [(a + 1, b + 1) for a, b in word_aligns_orig_sys[matching_method]]
    #     orig_auto_annots, _ = annotate_sentence(
    #         orig_sent.split(), sys_sent.split(), word_aligns_orig_sys, orig_parse, sys_parse
    #     )
    #
    #     all_orig_silver_annots = []
    #     for ref_sent, ref_parse in zip(ref_sents, ref_parses):
    #         word_aligns_orig_ref = aligner.get_word_aligns(orig_sent.split(), ref_sent.split())
    #         word_aligns_orig_ref = [(a + 1, b + 1) for a, b in word_aligns_orig_ref[matching_method]]
    #         orig_silver_annots, _ = annotate_sentence(
    #             orig_sent.split(),
    #             ref_sent.split(),
    #             word_aligns_orig_ref,
    #             orig_parse,
    #             ref_parse,
    #         )
    #         assert len(orig_silver_annots) == len(orig_auto_annots)
    #         all_orig_silver_annots.append(orig_silver_annots)
    #
    #     sys_correct = {'R': 0, 'D': 0, 'K': 0}
    #     sys_total = {'R': 0, 'D': 0, 'K': 0}
    #     ref_total = {'R': 0, 'D': 0, 'K': 0}
    #     for i, token in enumerate(orig_auto_annots):
    #         auto_label = token['label'].split('-')[-1]
    #         ref_tokens = [ref[i] for ref in all_orig_silver_annots]
    #         for ref in ref_tokens:
    #             silver_label = ref['label'].split('-')[-1]
    #             if auto_label == silver_label:
    #                 if auto_label == 'R':
    #                     if token['replace'] == ref['replace']:
    #                         sys_correct[auto_label] += 1
    #                 elif auto_label in ['D', 'K']:
    #                     sys_correct[auto_label] += 1
    #             sys_total[auto_label] += 1
    #             ref_total[auto_label] += 1
    #
    #     return np.amax(curr_sent_scores, axis=0)
