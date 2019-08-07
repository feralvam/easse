import click
import sacrebleu

from easse.annotation.word_level import corpus_analyse_operations
from easse.fkgl import corpus_fkgl
from easse.utils.helpers import read_lines
from easse.quality_estimation import corpus_quality_estimation
from easse.sari import corpus_sari
from easse.samsa import corpus_samsa
from easse.utils.resources import (get_turk_orig_sents, get_turk_refs_sents,
                                   get_pwkp_orig_sents, get_pwkp_refs_sents,
                                   get_hsplit_orig_sents, get_hsplit_refs_sents)
from easse.utils.constants import VALID_TEST_SETS, VALID_METRICS, DEFAULT_METRICS
from easse.report import write_html_report


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option()
def cli():
    pass


def common_options(function):
    function = click.option(
            '--test_set', '-t', type=click.Choice(VALID_TEST_SETS), required=True,
            help='test set to use.',
    )(function)
    function = click.option(
            '--input_path', '-i', type=click.Path(), default=None,
            help='Path to the system predictions input file that is to be evaluated.',
    )(function)
    function = click.option(
            '--tokenizer', '-tok', type=click.Choice(['13a', 'intl', 'moses', 'plain']), default='13a',
            help='Tokenization method to use.',
    )(function)
    function = click.option(
            '--metrics', '-m', type=str, default=','.join(DEFAULT_METRICS),
            help=(f'Comma-separated list of metrics to compute. Valid: {",".join(VALID_METRICS)}'
                  ' (SAMSA is disabled by default for the sake of speed'),
    )(function)
    return function


@cli.command('evaluate')
@common_options
@click.option('--analysis', '-a', is_flag=True,
              help=f'Perform word-level transformation analysis.')
@click.option('--quality_estimation', '-q', is_flag=True,
              help='Perform quality estimation.')
def _evaluate_system_output(*args, **kwargs):
    evaluate_system_output(*args, **kwargs)


def evaluate_system_output(
        test_set,
        input_path=None,
        tokenizer='13a',
        metrics=','.join(VALID_METRICS),
        analysis=False,
        quality_estimation=False,
        ):
    """
    Evaluate a system output with automatic metrics.
    """
    if input_path is not None:
        sys_output = read_lines(input_path)
    else:
        # read the system output
        with click.get_text_stream('stdin', encoding='utf-8') as system_output_file:
            sys_output = system_output_file.read().splitlines()

    # get the metrics that need to be computed
    metrics = metrics.split(',')

    load_orig_sents = ('sari' in metrics) or ('samsa' in metrics) or analysis or quality_estimation
    load_refs_sents = ('sari' in metrics) or ('bleu' in metrics) or analysis
    # get the references from the test set
    if test_set in ['turk', 'turk_valid']:
        lowercase = False
        phase = 'test' if test_set == 'turk' else 'valid'
        if load_orig_sents:
            orig_sents = get_turk_orig_sents(phase=phase)
        if load_refs_sents:
            refs_sents = get_turk_refs_sents(phase=phase)

    if test_set in ['pwkp', 'pwkp_valid']:
        lowercase = True
        phase = 'test' if test_set == 'pwkp' else 'valid'
        if load_orig_sents:
            orig_sents = get_pwkp_orig_sents(phase=phase)
        if load_refs_sents:
            refs_sents = get_pwkp_refs_sents(phase=phase)

    if test_set == 'hsplit':
        sys_output = sys_output[:70]
        lowercase = True
        if load_orig_sents:
            orig_sents = get_hsplit_orig_sents()
        if load_refs_sents:
            refs_sents = get_hsplit_refs_sents()

    if load_orig_sents:
        assert len(sys_output) == len(orig_sents)
    if load_refs_sents:
        assert len(sys_output) == len(refs_sents[0])

    # compute each metric
    if 'bleu' in metrics:
        bleu_score = sacrebleu.corpus_bleu(sys_output, refs_sents,
                                           force=True, tokenize=tokenizer, lowercase=lowercase).score
        click.echo(f'BLEU: {bleu_score:.2f}')

    if 'sari' in metrics:
        sari_score = corpus_sari(orig_sents, sys_output, refs_sents, tokenizer=tokenizer, lowercase=lowercase)
        click.echo(f'SARI: {sari_score:.2f}')

    if 'samsa' in metrics:
        samsa_score = corpus_samsa(orig_sents, sys_output, tokenizer=tokenizer, verbose=True, lowercase=lowercase)
        click.echo(f'SAMSA: {samsa_score:.2f}')

    if 'fkgl' in metrics:
        fkgl_score = corpus_fkgl(sys_output, tokenizer=tokenizer)
        click.echo(f'FKGL: {fkgl_score:.2f}')

    if analysis:
        word_level_analysis = corpus_analyse_operations(orig_sents, sys_output, refs_sents,
                                                        verbose=False, as_str=True)
        click.echo(f'Word-level Analysis: {word_level_analysis}')

    if quality_estimation:
        quality_estimation_scores = corpus_quality_estimation(
                orig_sents,
                sys_output,
                tokenizer=tokenizer,
                lowercase=lowercase
                )
        quality_estimation_scores = {k: round(v, 2) for k, v in quality_estimation_scores.items()}
        click.echo(f'Quality estimation: {quality_estimation_scores}')


@cli.command('report')
@common_options
@click.option('--report_path', '-p', type=click.Path(), default='report.html',
              help='Path to the output HTML report.')
def _report(*args, **kwargs):
    report(*args, **kwargs)


def report(test_set, input_path=None, report_path='report.html', tokenizer='13a', metrics=','.join(DEFAULT_METRICS)):
    """
    Create a HTML report file with automatic metrics, plots and samples.
    """
    if input_path is not None:
        sys_output = read_lines(input_path)
    else:
        # read the system output
        with click.get_text_stream('stdin', encoding='utf-8') as system_output_file:
            sys_output = system_output_file.read().splitlines()
    if test_set in ['turk', 'turk_valid']:
        lowercase = False
        phase = 'test' if test_set == 'turk' else 'valid'
        refs_sents = get_turk_refs_sents(phase=phase)
        orig_sents = get_turk_orig_sents(phase=phase)
    if test_set == 'hsplit':
        sys_output = sys_output[:70]
        lowercase = True
        refs_sents = get_hsplit_refs_sents()
        orig_sents = get_hsplit_orig_sents()
    write_html_report(
            report_path, orig_sents, sys_output, refs_sents, test_set_name=test_set,
            lowercase=lowercase, tokenizer=tokenizer, metrics=metrics,
            )
