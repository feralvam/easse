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
        ('turk', 'orig'): DATA_DIR / f'test_sets/turk/test.8turkers.tok.norm',
        ('turk', 'refs'): [DATA_DIR / f'test_sets/turk/test.8turkers.tok.turk.{i}' for i in range(8)],
        ('turk_valid', 'orig'): DATA_DIR / f'test_sets/turk/tune.8turkers.tok.norm',
        ('turk_valid', 'refs'): [DATA_DIR / f'test_sets/turk/tune.8turkers.tok.turk.{i}' for i in range(8)],
        ('pwkp', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.test.src',
        ('pwkp', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.test.dst'],
        ('pwkp_valid', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.valid.src',
        ('pwkp_valid', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.valid.dst'],
        ('hsplit', 'orig'): DATA_DIR / f'test_sets/hsplit/hsplit.tok.src',
        ('hsplit', 'refs'): [DATA_DIR / f'test_sets/hsplit/hsplit.tok.{i+1}' for i in range(4)],
}

# Constants
VALID_TEST_SETS = ['turk', 'turk_valid', 'pwkp', 'pwkp_valid', 'hsplit', 'custom']
VALID_METRICS = ['bleu', 'sari', 'samsa', 'fkgl']
DEFAULT_METRICS = [m for m in VALID_METRICS if m != 'samsa']  # HACK: SAMSA is too long to compute
