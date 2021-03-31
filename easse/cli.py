from pathlib import Path

import click

from easse.fkgl import corpus_fkgl
from easse.utils.helpers import read_lines
from easse.quality_estimation import corpus_quality_estimation
from easse.sari import corpus_sari, get_corpus_sari_operation_scores
from easse.bleu import corpus_bleu, corpus_averaged_sentence_bleu
from easse.compression import corpus_f1_token
from easse.utils.constants import (
    VALID_TEST_SETS,
    VALID_METRICS,
    DEFAULT_METRICS,
)
from easse.utils.resources import get_orig_sents, get_refs_sents
from easse.report import write_html_report, write_multiple_systems_html_report


def get_sys_sents(test_set, sys_sents_path=None):
    # Get system sentences to be evaluated
    if sys_sents_path is not None:
        return read_lines(sys_sents_path)
    else:
        # read the system output
        with click.get_text_stream("stdin", encoding="utf-8") as system_output_file:
            return system_output_file.read().splitlines()


def get_orig_and_refs_sents(test_set, orig_sents_path=None, refs_sents_paths=None):
    # Get original and reference sentences
    if test_set == "custom":
        assert orig_sents_path is not None
        assert refs_sents_paths is not None
        if type(refs_sents_paths) == str:
            refs_sents_paths = refs_sents_paths.split(",")
        orig_sents = read_lines(orig_sents_path)
        refs_sents = [read_lines(ref_sents_path) for ref_sents_path in refs_sents_paths]
    else:
        orig_sents = get_orig_sents(test_set)
        refs_sents = get_refs_sents(test_set)
    # Final checks
    assert all(
        [len(orig_sents) == len(ref_sents) for ref_sents in refs_sents]
    ), f'Not same number of lines for test_set={test_set}, orig_sents_path={orig_sents_path}, refs_sents_paths={refs_sents_paths}'  # noqa: E501
    return orig_sents, refs_sents


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
def cli():
    pass


def common_options(function):
    function = click.option(
        "--test_set",
        "-t",
        type=click.Choice(VALID_TEST_SETS),
        required=True,
        help="Test set to use.",
    )(function)
    function = click.option(
        "--orig_sents_path",
        type=click.Path(),
        default=None,
        help='Path to the source sentences. Only used when test_set == "custom".',
    )(function)
    function = click.option(
        "--refs_sents_paths",
        type=str,
        default=None,
        help='Comma-separated list of path(s) to the references(s). Only used when test_set == "custom".',
    )(function)
    function = click.option(
        "--lowercase/--no-lowercase",
        "-lc/--no-lc",
        default=True,
        help="Compute case-insensitive scores for all metrics. ",
    )(function)
    function = click.option(
        "--tokenizer",
        "-tok",
        type=click.Choice(["13a", "intl", "moses", "penn", "none"]),
        default="13a",
        help="Tokenization method to use.",
    )(function)
    function = click.option(
        "--metrics",
        "-m",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=(
            f'Comma-separated list of metrics to compute. Valid: {",".join(VALID_METRICS)}'
            " (SAMSA is disabled by default for the sake of speed)."
        ),
    )(function)
    return function


@cli.command("evaluate")
@common_options
@click.option(
    "--analysis",
    "-a",
    is_flag=True,
    help=f"Perform word-level transformation analysis.",
)
@click.option(
    "--quality_estimation",
    "-q",
    is_flag=True,
    help="Compute quality estimation features.",
)
@click.option(
    "--sys_sents_path",
    "-i",
    type=click.Path(),
    default=None,
    help="Path to the system predictions input file that is to be evaluated.",
)
def _evaluate_system_output(*args, **kwargs):
    kwargs["metrics"] = kwargs.pop("metrics").split(",")
    metrics_scores = evaluate_system_output(*args, **kwargs)

    def recursive_round(obj):
        def is_castable_to_float(obj):
            try:
                float(obj)
            except (ValueError, TypeError):
                return False
            return True

        if is_castable_to_float(obj):
            return round(obj, 3)
        if type(obj) is dict:
            return {key: recursive_round(value) for key, value in obj.items()}
        return obj

    print(recursive_round(metrics_scores))


def evaluate_system_output(
    test_set,
    sys_sents_path=None,
    orig_sents_path=None,
    refs_sents_paths=None,
    tokenizer="13a",
    lowercase=True,
    metrics=DEFAULT_METRICS,
    analysis=False,
    quality_estimation=False,
):
    """
    Evaluate a system output with automatic metrics.
    """
    for metric in metrics:
        assert metric in VALID_METRICS, f'"{metric}" is not a valid metric. Choose among: {VALID_METRICS}'
    sys_sents = get_sys_sents(test_set, sys_sents_path)
    orig_sents, refs_sents = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths)

    # compute each metric
    metrics_scores = {}
    if "bleu" in metrics:
        metrics_scores["bleu"] = corpus_bleu(
            sys_sents,
            refs_sents,
            force=True,
            tokenizer=tokenizer,
            lowercase=lowercase,
        )

    if "sent_bleu" in metrics:
        metrics_scores["sent_bleu"] = corpus_averaged_sentence_bleu(
            sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase
        )

    if "sari" in metrics:
        metrics_scores["sari"] = corpus_sari(
            orig_sents,
            sys_sents,
            refs_sents,
            tokenizer=tokenizer,
            lowercase=lowercase,
        )

    if "sari_legacy" in metrics:
        metrics_scores["sari_legacy"] = corpus_sari(
            orig_sents,
            sys_sents,
            refs_sents,
            tokenizer=tokenizer,
            lowercase=lowercase,
            legacy=True,
        )

    if "sari_by_operation" in metrics:
        (
            metrics_scores["sari_add"],
            metrics_scores["sari_keep"],
            metrics_scores["sari_del"],
        ) = get_corpus_sari_operation_scores(
            orig_sents,
            sys_sents,
            refs_sents,
            tokenizer=tokenizer,
            lowercase=lowercase,
        )

    if "samsa" in metrics:
        from easse.samsa import corpus_samsa

        metrics_scores["samsa"] = corpus_samsa(
            orig_sents,
            sys_sents,
            tokenizer=tokenizer,
            lowercase=lowercase,
            verbose=True,
        )

    if "fkgl" in metrics:
        metrics_scores["fkgl"] = corpus_fkgl(sys_sents, tokenizer=tokenizer)

    if "f1_token" in metrics:
        metrics_scores["f1_token"] = corpus_f1_token(sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase)

    if "bertscore" in metrics:
        from easse.bertscore import corpus_bertscore  # Inline import to use EASSE without installing all dependencies

        (
            metrics_scores["bertscore_precision"],
            metrics_scores["bertscore_recall"],
            metrics_scores["bertscore_f1"],
        ) = corpus_bertscore(sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase)

    if analysis:
        from easse.annotation.word_level import (
            WordOperationAnnotator,
        )  # Inline import to use EASSE without installing all dependencies

        word_operation_annotator = WordOperationAnnotator(tokenizer=tokenizer, lowercase=lowercase, verbose=True)
        metrics_scores["word_level_analysis"] = word_operation_annotator.analyse_operations(
            orig_sents, sys_sents, refs_sents, as_str=True
        )

    if quality_estimation:
        metrics_scores["quality_estimation"] = corpus_quality_estimation(
            orig_sents, sys_sents, tokenizer=tokenizer, lowercase=lowercase
        )

    return metrics_scores


@cli.command("report")
@common_options
@click.option(
    "--sys_sents_path",
    "-i",
    type=click.Path(),
    default=None,
    help="""Path to the system predictions input file that is to be evaluated.
              You can also input a comma-separated list of files to compare multiple systems.""",
)
@click.option(
    "--report_path",
    "-p",
    type=click.Path(),
    default="easse_report.html",
    help="Path to the output HTML report.",
)
def _report(*args, **kwargs):
    kwargs["metrics"] = kwargs.pop("metrics").split(",")
    if kwargs["sys_sents_path"] is not None and len(kwargs["sys_sents_path"].split(",")) > 1:
        # If we got multiple systems as input, split the paths and rename the key
        kwargs["sys_sents_paths"] = kwargs.pop("sys_sents_path").split(",")
        multiple_systems_report(*args, **kwargs)
    else:
        report(*args, **kwargs)


def report(
    test_set,
    sys_sents_path=None,
    orig_sents_path=None,
    refs_sents_paths=None,
    report_path="easse_report.html",
    tokenizer="13a",
    lowercase=True,
    metrics=DEFAULT_METRICS,
):
    """
    Create a HTML report file with automatic metrics, plots and samples.
    """
    sys_sents = get_sys_sents(test_set, sys_sents_path)
    orig_sents, refs_sents = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths)
    write_html_report(
        report_path,
        orig_sents,
        sys_sents,
        refs_sents,
        test_set=test_set,
        lowercase=lowercase,
        tokenizer=tokenizer,
        metrics=metrics,
    )


def multiple_systems_report(
    test_set,
    sys_sents_paths,
    orig_sents_path=None,
    refs_sents_paths=None,
    report_path="easse_report.html",
    tokenizer="13a",
    lowercase=True,
    metrics=DEFAULT_METRICS,
    system_names=None,
):
    """
    Create a HTML report file comparing multiple systems with automatic metrics, plots and samples.
    """
    sys_sents_list = [read_lines(path) for path in sys_sents_paths]
    orig_sents, refs_sents = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths)
    if system_names is None:
        system_names = [Path(path).name for path in sys_sents_paths]
    write_multiple_systems_html_report(
        report_path,
        orig_sents,
        sys_sents_list,
        refs_sents,
        system_names=system_names,
        test_set=test_set,
        lowercase=lowercase,
        tokenizer=tokenizer,
        metrics=metrics,
    )
