import click
import sacrebleu

from easse.annotation.word_level import corpus_analyse_operations
from easse.fkgl import corpus_fkgl
from easse.utils.helpers import read_lines
from easse.quality_estimation import corpus_quality_estimation
from easse.sari import corpus_sari
from easse.samsa import corpus_samsa
from easse.splitting import corpus_macro_avg_sent_bleu
from easse.compression import corpus_f1_token
from easse.utils.constants import VALID_TEST_SETS, VALID_METRICS, DEFAULT_METRICS
from easse.utils.resources import get_orig_sents, get_refs_sents
from easse.report import write_html_report


def get_sents(test_set, orig_sents_path=None, sys_sents_path=None, refs_sents_paths=None):
    # Get system sentences to be evaluated
    if sys_sents_path is not None:
        sys_sents = read_lines(sys_sents_path)
    else:
        # read the system output
        with click.get_text_stream('stdin', encoding='utf-8') as system_output_file:
            sys_sents = system_output_file.read().splitlines()
    # Get original and reference sentences
    if test_set == 'custom':
        assert orig_sents_path is not None
        assert refs_sents_paths is not None
        if type(refs_sents_paths) == str:
            refs_sents_paths = refs_sents_paths.split(',')
        orig_sents = read_lines(orig_sents_path)
        refs_sents = [read_lines(ref_sents_path) for ref_sents_path in refs_sents_paths]
    else:
        orig_sents = get_orig_sents(test_set)
        refs_sents = get_refs_sents(test_set)
    # Final checks
    assert len(sys_sents) == len(orig_sents)
    assert all([len(sys_sents) == len(ref_sents) for ref_sents in refs_sents])
    return orig_sents, sys_sents, refs_sents


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option()
def cli():
    pass


def common_options(function):
    function = click.option(
            '--test_set', '-t', type=click.Choice(VALID_TEST_SETS), required=True,
            help='Test set to use.',
    )(function)
    function = click.option(
            '--sys_sents_path', type=click.Path(), default=None,
            help='Path to the system predictions input file that is to be evaluated.',
    )(function)
    function = click.option(
            '--orig_sents_path', type=click.Path(), default=None,
            help='Path to the source sentences. Only used when test_set == "custom".',
    )(function)
    function = click.option(
            '--refs_sents_paths', type=str, default=None,
            help='Comma-separated list of path(s) to the references(s). Only used when test_set == "custom".',
    )(function)
    function = click.option(
        '--lowercase/--no-lowercase', '-lc/--no-lc', default=True,
        help='Compute case-insensitive scores for all metrics. ',
    )(function)
    function = click.option(
            '--tokenizer', '-tok', type=click.Choice(['13a', 'intl', 'moses', 'penn', 'none']), default='13a',
            help='Tokenization method to use.',
    )(function)
    function = click.option(
            '--metrics', '-m', type=str, default=','.join(DEFAULT_METRICS),
            help=(f'Comma-separated list of metrics to compute. Valid: {",".join(VALID_METRICS)}'
                  ' (SAMSA is disabled by default for the sake of speed).'),
    )(function)
    return function


@cli.command('evaluate')
@common_options
@click.option('--analysis', '-a', is_flag=True,
              help=f'Perform word-level transformation analysis.')
@click.option('--quality_estimation', '-q', is_flag=True,
              help='Compute quality estimation features.')
def _evaluate_system_output(*args, **kwargs):
    metrics_scores = evaluate_system_output(*args, **kwargs)
    print(metrics_scores)


def evaluate_system_output(
        test_set,
        sys_sents_path=None,
        orig_sents_path=None,
        refs_sents_paths=None,
        tokenizer='13a',
        lowercase=True,
        metrics=','.join(DEFAULT_METRICS),
        analysis=False,
        quality_estimation=False,
        ):
    """
    Evaluate a system output with automatic metrics.
    """
    # get the metrics that need to be computed
    metrics = metrics.split(',')
    orig_sents, sys_sents, refs_sents = get_sents(test_set, orig_sents_path, sys_sents_path, refs_sents_paths)

    # compute each metric
    metrics_scores = {}
    if 'bleu' in metrics:
        metrics_scores["bleu"] = sacrebleu.corpus_bleu(sys_sents, refs_sents, force=True,
                                                       tokenize=tokenizer, lowercase=lowercase).score

    if 'sent_bleu' in metrics:
        metrics_scores["sent_bleu"] = corpus_macro_avg_sent_bleu(sys_sents, refs_sents,
                                                                 tokenizer=tokenizer, lowercase=lowercase)

    if 'sari' in metrics:
        metrics_scores["sari"] = corpus_sari(orig_sents, sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase)

    if 'samsa' in metrics:
        metrics_scores["samsa"] = corpus_samsa(orig_sents, sys_sents, tokenizer=tokenizer, verbose=True, lowercase=lowercase)

    if 'fkgl' in metrics:
        metrics_scores["fkgl"] = corpus_fkgl(sys_sents, tokenizer=tokenizer)

    if 'f1_token' in metrics:
        metrics_scores["f1_token"] = corpus_f1_token(sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase)

    if analysis:
        metrics_scores["word_level_analysis"] = corpus_analyse_operations(orig_sents, sys_sents, refs_sents,
                                                                          verbose=False, as_str=True)

    if quality_estimation:
        quality_estimation_scores = corpus_quality_estimation(
                orig_sents,
                sys_sents,
                tokenizer=tokenizer,
                lowercase=lowercase
                )
        metrics_scores["quality_estimation"] = {k: round(v, 2) for k, v in quality_estimation_scores.items()}

    return metrics_scores


@cli.command('report')
@common_options
@click.option('--report_path', '-p', type=click.Path(), default='report.html',
              help='Path to the output HTML report.')
def _report(*args, **kwargs):
    report(*args, **kwargs)


def report(
        test_set,
        sys_sents_path=None,
        orig_sents_path=None,
        refs_sents_paths=None,
        report_path='report.html',
        tokenizer='13a',
        lowercase=True,
        metrics=','.join(DEFAULT_METRICS)
        ):
    """
    Create a HTML report file with automatic metrics, plots and samples.
    """
    orig_sents, sys_sents, refs_sents = get_sents(test_set, orig_sents_path, sys_sents_path, refs_sents_paths)
    write_html_report(
            report_path, orig_sents, sys_sents, refs_sents, test_set_name=test_set,
            lowercase=lowercase, tokenizer=tokenizer, metrics=metrics,
            )
