"""
This module contains functions for annotating simplification operations at the word-level.
"""
from typing import List
from operator import itemgetter
import string

import numpy as np
from tqdm import tqdm
from simalign import SentenceAligner
from sklearn.metrics import f1_score
from nltk.tree import ParentedTree

from easse.aligner.corenlp_utils import syntactic_parse_texts, posTag
from easse.aligner.aligner import MonolingualWordAligner
import easse.utils.preprocessing as utils_prep


# =============================================================================
# Constants
# =============================================================================

SIMOP_LABELS_GROUPS = {
    "add": ["B-A", "B-AC", "I-A", "I-AC"],
    "delete": ["B-D", "B-DC", "I-D", "I-DC"],
    "replace": ["B-R", "I-R"],
    "move": ["B-M", "B-MC", "I-M", "I-MC"],
}
SIMOP_LABELS = [
    "B-A",
    "B-AC",
    "B-D",
    "B-DC",
    "B-M",
    "B-MC",
    "B-R",
    "I-A",
    "I-AC",
    "I-D",
    "I-DC",
    "I-M",
    "I-MC",
    "I-R",
]
ORIG_OPS_LABELS = ["D", "M", "R", "C"]
CLAUSE_SYNT_TAGS = ["SBAR", "CC"]
CHUNK_SYNT_TAGS = ["NP", "VP", "PP"]

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
        if ref_token["label"] == "B-A":
            # find a token in source, with the same index and labeled as 'delete'
            same_index_in_src = [
                src_token
                for src_token in src_annots
                if src_token["index"] == ref_token["index"] and src_token["label"] == "B-D"
            ]
            if same_index_in_src:  # it is not an empty list
                src_token = same_index_in_src[0]  # there is always one at the most
                # check that both tokens have the same part of speech tag
                same_postag = _have_same_postag(src_token["index"], ref_token["index"], src_parse, ref_parse)
                if same_postag:
                    src_token["label"] = "B-R"  # the token in the source must be labeled as replace
                    ref_token["label"] = "O"  # the token in the reference now has no label
                    src_token["replace"] = [ref_token["word"]]  # include the information of the replacement


def _label_group_simop(
    annots,
    parse,
    group_synt_tags,
    old_token_labels,
    new_group_label,
    majority_percent=0.75,
):
    """
    Annotate a sequence of tokens with the same operation label and which belong to the same syntactic group,
    with a given group operation label.
    """

    parse_tree = ParentedTree.fromstring(parse["sentences"][0]["parse"])
    num_tokens = len(annots)

    for ptr_token in range(0, num_tokens):
        token = annots[ptr_token]
        if token["label"] in old_token_labels:
            # get the subtree of the token
            treepos = parse_tree.leaf_treeposition(token["index"] - 1)
            subtree = parse_tree[treepos[:-1]]

            # find if it belongs to the specified syntactic group
            parent = subtree.parent()
            while parent and parent.label() not in group_synt_tags:
                parent = parent.parent()

            if parent:  # the token is inside one of the specified syntactic groups
                # get the index of the first token in the group, according to the parse tree
                begin = parse_tree.treepositions("leaves").index(parent.treeposition() + parent.leaf_treeposition(0))

                # count the number of tokens in the group that have been labeled with the same operation
                with_same_label = 0
                group_tokens = parent.leaves()
                for gt in group_tokens:
                    if gt == token["word"]:  # check if the word in the group has been labeled
                        if token["label"] in old_token_labels:
                            with_same_label += 1
                        ptr_token += 1
                        if ptr_token == num_tokens:  # check that there are still tokens annotated
                            break
                        token = annots[ptr_token]

                # check if the majority of the tokens in the group have been labeled with the same operation
                if with_same_label >= majority_percent * len(group_tokens):
                    end = begin + with_same_label

                    # label the first token in the group
                    token = annots[begin]
                    token["label"] = "B-" + new_group_label
                    group_begin = token["index"]
                    token["groupbegin"] = group_begin

                    # label the rest of the tokens in the group
                    for i in range(begin + 1, end):
                        token = annots[i]
                        token["label"] = "I-" + new_group_label
                        token["groupbegin"] = group_begin
                    ptr_token -= 1
                else:
                    ptr_token = begin


def _label_move(src_annots, ref_annots, aligns):
    """
    Annotate 'move' checking if the relative index of a token in source changes in reference, considering
    preceding deletions, additions and multi-token replacements.
    """

    shift_left = 0
    for src_token in src_annots:
        # check if the token has been labeled to be deleted or as part of a replace
        if src_token["label"] in ["B-D", "B-DC", "I-DC", "I-D", "I-R"]:
            shift_left += 1
        else:
            # get its position in the reference (using the word alignments)
            ref_index_list = [ref_index for src_index, ref_index in aligns if src_index == src_token["index"]]
            if ref_index_list:
                ref_index = ref_index_list[0]
            else:
                ref_index = src_token["index"]

            # count the number 'add' to the reference up until the new position of the source token
            shift_right = 0
            for ref_token in ref_annots:
                if ref_token["index"] < ref_index:
                    if ref_token["label"] in ["B-A", "B-AC", "I-AC", "I-A"]:
                        shift_right += 1
                else:
                    break

            # check if the token needs to be moved
            if (src_token["index"] - shift_left + shift_right) != ref_index:
                if src_token["label"] == "O":  # the token is kept and moved
                    new_label = "B-M"
                else:
                    new_label = src_token["label"]  # in any other case, the label stays the same
                src_token["label"] = new_label
                src_token["move"] = ref_index


def _label_delete_replace(src, ref, aligns):
    """Annotate deletions and replacements in the source sentence."""
    src_annots = []
    for token_index, token_word in enumerate(src, start=1):
        src_token = {"index": token_index, "word": token_word, "label": ""}
        # do not label punctuations
        if token_word in string.punctuation:
            src_token["label"] = "O"
        else:
            # get the indexes of all the words in the reference to which the current token in source is aligned to
            aligns_list = [ref_index for src_index, ref_index in aligns if src_index == token_index]
            # check if the token is aligned
            if aligns_list:
                # check if it has been aligned to only one token and if they are exactly the same
                if len(aligns_list) == 1 and token_word.lower() == ref[aligns_list[0] - 1].lower():
                    # it is a 'keep'
                    src_token["label"] = "O"
                else:
                    # it is a 'replacement'
                    src_token["label"] = "B-R"
                    src_token["replace"] = []
                    # recover all the tokens in reference for which this token in source is replaced
                    aligns_list.sort()
                    for ref_index in aligns_list:
                        src_token["replace"].append(ref[ref_index - 1])
            else:
                # label as delete
                src_token["label"] = "B-D"

        src_annots.append(src_token)

    src_annots = sorted(src_annots, key=itemgetter("index"))

    return src_annots


def _label_add_replace(ref, aligns, src_annots):
    """Annotate additions in the reference sentence. Improve replacements annotation in the source sentence."""

    ref_annots = []
    for token_index, token_word in enumerate(ref, start=1):
        ref_token = {"index": token_index, "word": token_word, "label": ""}
        if token_word in string.punctuation:
            ref_token["label"] = "O"
        else:
            # get the indexes of all the tokens in the source to which the current token in reference is aligned
            aligns_list = [src_index for src_index, ref_index in aligns if ref_index == token_index]
            # check if the token is aligned
            if aligns_list:
                # it is the replacement of some word(s) in the source
                ref_token["label"] = "O"  # the token in the reference has no label
                if len(aligns_list) > 1:
                    # it is the replacement of a phrase in source, so the source token annotations should be changed
                    aligns_list.sort()
                    for i in range(1, len(aligns_list)):  # token 0 already has 'B-R' because of label_delete_replace
                        src_index = aligns_list[i]
                        src_token = [src_token for src_token in src_annots if src_token["index"] == src_index][0]
                        src_token["label"] = "I-R"
                        src_token["replace"] = []  # token with 'B-R' has all the replacement tokens
            else:
                # label as 'add'
                ref_token["label"] = "B-A"

        ref_annots.append(ref_token)

    ref_annots = sorted(ref_annots, key=itemgetter("index"))

    return ref_annots


# =============================================================================
# Main Functions
# =============================================================================


def annotate_sentence(src, ref, aligns, src_parse, ref_parse, include_phrase_level=False):
    """Annotate all the simplification operations in the sentence pair src-ref."""
    # token-level delete, add and replace
    src_annots = _label_delete_replace(src, ref, aligns)
    ref_annots = _label_add_replace(ref, aligns, src_annots)

    # simple heuristic to improve token-level replacements
    _improve_replace(src_annots, ref_annots, src_parse, ref_parse)

    # token-level move
    _label_move(src_annots, ref_annots, aligns)

    if include_phrase_level:
        # clause-level delete and add
        _label_group_simop(src_annots, src_parse, CLAUSE_SYNT_TAGS, ["B-D", "I-D"], "DC")
        _label_group_simop(ref_annots, ref_parse, CLAUSE_SYNT_TAGS, ["B-A", "I-A"], "AC")

        # chunk-level delete and add
        _label_group_simop(
            src_annots,
            src_parse,
            CHUNK_SYNT_TAGS,
            ["B-D", "I-D"],
            "D",
            majority_percent=1,
        )
        _label_group_simop(
            ref_annots,
            ref_parse,
            CHUNK_SYNT_TAGS,
            ["B-A", "I-A"],
            "A",
            majority_percent=1,
        )

        # clause-level move
        _label_group_simop(src_annots, src_parse, CLAUSE_SYNT_TAGS, ["B-M", "I-M"], "MC")

        # chunk-level move
        _label_group_simop(
            src_annots,
            src_parse,
            CHUNK_SYNT_TAGS,
            ["B-M", "I-M"],
            "M",
            majority_percent=1,
        )

    return src_annots, ref_annots


def _from_annots_to_labels(sent_annots, labels_to_include=None, default_label="O"):
    if labels_to_include is None:
        labels_to_include = SIMOP_LABELS
    labels = []
    for token in sent_annots:
        label = token["label"]  # label = token["label"].split("-")[-1]
        if label not in labels_to_include:
            label = default_label
        labels.append(label)

    return labels


def _remove_iob_labels(ops_labels):
    return [label.split("-")[-1] for label in ops_labels]


class WordOperationAnnotator:
    def __init__(
        self,
        align_tool: str = "simalign",
        simalign_method: str = "inter",
        include_phrase_level: bool = False,
        iob_labels: bool = False,
        lowercase: bool = False,
        tokenizer: str = "moses",
        verbose: bool = False,
    ):
        if align_tool == "simalign":
            matching_methods = {"inter": "a", "mwmf": "m", "itermax": "i"}
            assert simalign_method in matching_methods.keys()
            self._word_aligner = SentenceAligner(
                model="bert",
                token_type="bpe",
                matching_methods=matching_methods[simalign_method],
            )
        elif align_tool == "mwa":
            self._word_aligner = MonolingualWordAligner()
        else:
            print("Unidentified alignment tool.")
            return

        self._include_phrase_level = include_phrase_level
        self._iob_labels = iob_labels
        self._lowercase = lowercase
        self._tokenizer = tokenizer
        self._verbose = verbose

    def analyse_operations(
        self,
        orig_sentences: List[str],
        sys_sentences: List[str],
        refs_sentences: List[List[str]],
        operations=None,
        as_str: bool = False,
    ):
        if operations is None:
            operations = ["D", "R", "M", "C"]

        sentence_scores = self.compute_operations_sentence_scores(
            orig_sentences, sys_sentences, refs_sentences, operations
        )
        label_scores = np.mean(sentence_scores, axis=0)
        assert len(label_scores) == len(operations)
        score_per_label = dict(zip(operations, label_scores))

        if as_str:
            score_per_label = " ".join([f"{label}={100. * score:.2f}" for label, score in score_per_label.items()])

        return score_per_label

    def compute_operations_sentence_scores(
        self,
        orig_sentences: List[str],
        sys_sentences: List[str],
        refs_sentences: List[List[str]],
        operations=None,
    ):
        if operations is None:
            operations = ["D", "R", "M", "C"]

        simops_orig_sys, _ = self.identify_operations(orig_sentences, sys_sentences)

        num_refs = len(refs_sentences)
        all_orig_sents = []
        all_ref_sents = []
        for orig_sent, *ref_sents in zip(orig_sentences, *refs_sentences):
            all_orig_sents += [orig_sent] * num_refs
            all_ref_sents += ref_sents

        all_simops_orig_refs, _ = self.identify_operations(all_orig_sents, all_ref_sents)

        sentence_scores = []
        for i, orig_silver_labels in tqdm(enumerate(simops_orig_sys)):
            curr_ref_scores = []
            for orig_auto_labels in all_simops_orig_refs[i * num_refs : i * num_refs + num_refs]:
                if not self._iob_labels:
                    orig_silver_labels = _remove_iob_labels(orig_silver_labels)
                    orig_auto_labels = _remove_iob_labels(orig_auto_labels)
                assert len(orig_silver_labels) == len(orig_auto_labels)
                f1_per_label = f1_score(
                    orig_silver_labels,
                    orig_auto_labels,
                    labels=operations,
                    average=None,
                )
                curr_ref_scores.append(f1_per_label)
            sentence_scores.append(np.amax(curr_ref_scores, axis=0))
        return np.stack(sentence_scores, axis=0)

    def identify_operations(self, orig_sentences: List[str], simp_sentences: List[str]):
        orig_sentences = [utils_prep.normalize(sent, self._lowercase, self._tokenizer) for sent in orig_sentences]
        simp_sentences = [utils_prep.normalize(sent, self._lowercase, self._tokenizer) for sent in simp_sentences]

        all_parses = syntactic_parse_texts(
            orig_sentences + simp_sentences,
            with_constituency_parse=self._include_phrase_level,
            verbose=self._verbose,
        )
        orig_parses = all_parses[: len(orig_sentences)]
        simp_parses = all_parses[len(orig_sentences) :]

        orig_labels_per_sentence = []
        simp_labels_per_sentence = []
        for orig_sent, simp_sent, orig_parse, simp_parse in tqdm(
            zip(orig_sentences, simp_sentences, orig_parses, simp_parses),
            disable=(not self._verbose),
        ):
            word_aligns_orig_simp = self._get_word_alignments(orig_sent, orig_parse, simp_sent, simp_parse)
            orig_annots, simp_annots = annotate_sentence(
                orig_sent.split(),
                simp_sent.split(),
                word_aligns_orig_simp,
                orig_parse,
                simp_parse,
            )
            orig_labels = _from_annots_to_labels(orig_annots, default_label="C")
            simp_labels = _from_annots_to_labels(simp_annots, default_label="O")

            orig_labels_per_sentence.append(orig_labels)
            simp_labels_per_sentence.append(simp_labels)

        return orig_labels_per_sentence, simp_labels_per_sentence

    def _get_word_alignments(self, orig_sent: str, orig_parse, sys_sent: str, sys_parse):
        if isinstance(self._word_aligner, SentenceAligner):
            word_aligns = self._word_aligner.get_word_aligns(orig_sent.split(), sys_sent.split())
            word_aligns = [(a + 1, b + 1) for a, b in word_aligns[list(word_aligns.keys())[0]]]
        elif isinstance(self._word_aligner, MonolingualWordAligner):
            assert orig_parse is not None
            assert sys_parse is not None
            word_aligns = self._word_aligner.get_word_aligns(orig_parse, sys_parse)[0]
        else:
            word_aligns = None
        return word_aligns
