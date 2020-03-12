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
        ('turkcorpus_test', 'orig'): DATA_DIR / f'test_sets/turkcorpus/test.truecase.detok.orig',
        ('turkcorpus_test', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/test.truecase.detok.simp.{i}' for i in range(8)],
        ('turkcorpus_test_legacy', 'orig'): DATA_DIR / f'test_sets/turkcorpus/legacy/test.8turkers.tok.norm',
        ('turkcorpus_test_legacy', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/legacy/test.8turkers.tok.turk.{i}' for i in range(8)],
        ('turkcorpus_valid', 'orig'): DATA_DIR / f'test_sets/turkcorpus/tune.truecase.detok.orig',
        ('turkcorpus_valid', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/tune.truecase.detok.simp.{i}' for i in range(8)],
        ('turkcorpus_valid_legacy', 'orig'): DATA_DIR / f'test_sets/turkcorpus/legacy/tune.8turkers.tok.norm',
        ('turkcorpus_valid_legacy', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/legacy/tune.8turkers.tok.turk.{i}' for i in range(8)],
        ('pwkp_test', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.test.orig',
        ('pwkp_test', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.test.simp'],
        ('pwkp_valid', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.valid.orig',
        ('pwkp_valid', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.valid.simp'],
        ('hsplit_test', 'orig'): DATA_DIR / f'test_sets/hsplit/hsplit.recase.detok.src',
        ('hsplit_test', 'refs'): [DATA_DIR / f'test_sets/hsplit/hsplit.recase.detok.{i+1}' for i in range(4)],
        ('wikisplit_test', 'orig'): DATA_DIR / f'test_sets/wikisplit/wikisplit.test.untok.orig',
        ('wikisplit_test', 'refs'): [DATA_DIR / f'test_sets/wikisplit/wikisplit.test.untok.split'],
        ('wikisplit_valid', 'orig'): DATA_DIR / f'test_sets/wikisplit/wikisplit.valid.untok.orig',
        ('wikisplit_valid', 'refs'): [DATA_DIR / f'test_sets/wikisplit/wikisplit.valid.untok.split'],
        ('asset_test', 'orig'): DATA_DIR / f'test_sets/asset/asset.test.orig',
        ('asset_test', 'refs'): [DATA_DIR / f'test_sets/asset/asset.test.simp.{i}' for i in range(10)],
        ('asset_valid', 'orig'): DATA_DIR / f'test_sets/asset/asset.valid.orig',
        ('asset_valid', 'refs'): [DATA_DIR / f'test_sets/asset/asset.valid.simp.{i}' for i in range(10)],
        ('googlecomp_test', 'orig'): DATA_DIR / f'test_sets/googlecomp/googlecomp.test.orig',
        ('googlecomp_test', 'refs'): [DATA_DIR / f'test_sets/googlecomp/googlecomp.test.comp'],
        ('googlecomp_valid', 'orig'): DATA_DIR / f'test_sets/googlecomp/googlecomp.valid.orig',
        ('googlecomp_valid', 'refs'): [DATA_DIR / f'test_sets/googlecomp/googlecomp.valid.comp'],
}
SYSTEM_OUTPUTS_DIR = DATA_DIR / 'system_outputs'
SYSTEM_OUTPUTS_DIRS_MAP = {
        'turkcorpus_test': SYSTEM_OUTPUTS_DIR / 'turkcorpus/test',
        'turkcorpus_valid': SYSTEM_OUTPUTS_DIR / 'turkcorpus/valid',
        'pwkp_test': SYSTEM_OUTPUTS_DIR / 'pwkp/test',
}

# Constants
VALID_TEST_SETS = ['turkcorpus_test', 'turkcorpus_valid','turkcorpus_test_legacy', 'turkcorpus_valid_legacy',
                   'pwkp_test', 'pwkp_valid', 'hsplit_test', 'wikisplit_test', 'wikisplit_valid', 'asset_test',
                   'asset_valid', 'googlecomp_test', 'googlecomp_valid', 'custom']
VALID_METRICS = ['bleu', 'sari', 'samsa', 'fkgl', 'sent_bleu', 'f1_token']
DEFAULT_METRICS = [m for m in VALID_METRICS if m not in ['samsa', 'sent_bleu', 'f1_token']]
# HACK: Only simplification-specific metrics, but SAMSA which is too long to compute
