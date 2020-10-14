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
        self._word_operation_annotator = WordOperationAnnotator(
            align_tool=align_tool,
            simalign_method=simalign_method,
            include_phrase_level=with_clause,
            iob_labels=False,
            lowercase=lowercase,
            tokenizer=tokenizer,
            verbose=verbose,
        )
        self._with_clause = with_clause

    def identify_operations(self, orig_sentences: List[str], simp_sentences: List[str]):
        orig_labels, simp_labels = self._word_operation_annotator.identify_operations(orig_sentences, simp_sentences)
        labels = []
        for orig_ops, simp_ops, orig_sent, simp_sent in zip(orig_labels, simp_labels, orig_sentences, simp_sentences):
            sent_labels = []
            sent_labels += list(set(orig_ops))
            sent_labels += list(set(simp_ops))
            if count_sentence_splits(orig_sent, simp_sent) > 1:
                sent_labels.append("S")
            final_sent_labels = []
            for label in sent_labels:
                if not self._with_clause and label.upper() in ["DC", "AC", "MC"]:
                    label = label.upper()[0]
                final_sent_labels.append(label)
            labels.append(list(set(final_sent_labels)))
        return labels
