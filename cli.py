import click


@click.command()
@click.argument('system_output', type=click.Path(exists=True))
@click.argument('test_set', type=click.Choice(['turk', 'pwkp']))
def evaluate(system_output, test_set):
    """
    Evaluate a system_output in a test_set
    """

    return


@click.command()
@click.option('-n', "--name", required=True,
              help="Name of the test set. If not given, the folder name with be used.")
@click.option('-o', "--orig_file", required=True,
              help="Path of the text file with the original sentences.")
@click.option('-s', "--simp_file", required=True,
              help="Path to the text file with the simplified sentences.")
def register(name, orig_file, simp_file):
    """
    Registers a new test set so that some pre-processing is performed and future computations are faster.
    """
    return


@click.command()
@click.argument('test_set')
@click.option('-sb', "--sort_by", type=click.Choice(['bleu', 'sari', 'samsa']),
              help="Metric to use for sorting the systems' scores.")
def ranking(test_set, sort_by):
    """
    Prints the ranking of all systems in a given test_set.
    """
    return
