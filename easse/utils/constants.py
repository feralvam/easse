from pathlib import Path


# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PACKAGE_DIR / 'resources'
TOOLS_DIR = RESOURCES_DIR / 'tools'
DATA_DIR = RESOURCES_DIR / 'data'
STANFORD_CORENLP_DIR = TOOLS_DIR / 'stanford-corenlp-full-2018-10-05'
UCCA_DIR = TOOLS_DIR / 'ucca-bilstm-1.3.10'
UCCA_PARSER_PATH = UCCA_DIR / 'models/ucca-bilstm'
TEST_SETS_PATHS = {
        ('turkcorpus_test', 'orig'): DATA_DIR / f'test_sets/turkcorpus/test.8turkers.tok.norm',
        ('turkcorpus_test', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/test.8turkers.tok.turk.{i}' for i in range(8)],
        ('turkcorpus_valid', 'orig'): DATA_DIR / f'test_sets/turkcorpus/tune.8turkers.tok.norm',
        ('turkcorpus_valid', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/tune.8turkers.tok.turk.{i}' for i in range(8)],
        ('pwkp_test', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.test.src',
        ('pwkp_test', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.test.dst'],
        ('pwkp_valid', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.valid.src',
        ('pwkp_valid', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.valid.dst'],
        ('hsplit_test', 'orig'): DATA_DIR / f'test_sets/hsplit/hsplit.tok.src',
        ('hsplit_test', 'refs'): [DATA_DIR / f'test_sets/hsplit/hsplit.tok.{i+1}' for i in range(4)],
}
SYSTEM_OUTPUTS_DIR = DATA_DIR / 'system_outputs'
SYSTEM_OUTPUTS_DIRS_MAP = {
        'turkcorpus_test': SYSTEM_OUTPUTS_DIR / 'turkcorpus/test',
        'turkcorpus_valid': SYSTEM_OUTPUTS_DIR / 'turkcorpus/valid',
        'pwkp_test': SYSTEM_OUTPUTS_DIR / 'pwkp/test',
}

# Constants
VALID_TEST_SETS = ['turkcorpus_test', 'turkcorpus_valid', 'pwkp_test', 'pwkp_valid', 'hsplit_test', 'custom']
VALID_METRICS = ['bleu', 'sari', 'samsa', 'fkgl']
DEFAULT_METRICS = [m for m in VALID_METRICS if m != 'samsa']  # HACK: SAMSA is too long to compute
