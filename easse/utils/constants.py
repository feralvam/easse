from pathlib import Path


# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PACKAGE_DIR / "resources"
TOOLS_DIR = RESOURCES_DIR / "tools"
DATA_DIR = RESOURCES_DIR / "data"
STANFORD_CORENLP_DIR = TOOLS_DIR / "stanford-corenlp-full-2018-10-05"
UCCA_DIR = TOOLS_DIR / "ucca-bilstm-1.3.10"
UCCA_PARSER_PATH = UCCA_DIR / "models/ucca-bilstm"
TEST_SETS_PATHS = {
    ('asset_test', 'orig'): DATA_DIR / f'test_sets/asset/asset.test.orig',
    ('asset_test', 'refs'): [DATA_DIR / f'test_sets/asset/asset.test.simp.{i}' for i in range(10)],
    ('asset_valid', 'orig'): DATA_DIR / f'test_sets/asset/asset.valid.orig',
    ('asset_valid', 'refs'): [DATA_DIR / f'test_sets/asset/asset.valid.simp.{i}' for i in range(10)],
    ('turkcorpus_test', 'orig'): DATA_DIR / f'test_sets/turkcorpus/test.truecase.detok.orig',
    ('turkcorpus_test', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/test.truecase.detok.simp.{i}' for i in range(8)],
    ('turkcorpus_valid', 'orig'): DATA_DIR / f'test_sets/turkcorpus/tune.truecase.detok.orig',
    ('turkcorpus_valid', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/tune.truecase.detok.simp.{i}' for i in range(8)],
    ('turkcorpus_test_legacy', 'orig'): DATA_DIR / f'test_sets/turkcorpus/legacy/test.8turkers.tok.norm',
    ('turkcorpus_test_legacy', 'refs'): [
        DATA_DIR / f'test_sets/turkcorpus/legacy/test.8turkers.tok.turk.{i}' for i in range(8)
    ],
    ('turkcorpus_valid_legacy', 'orig'): DATA_DIR / f'test_sets/turkcorpus/legacy/tune.8turkers.tok.norm',
    ('turkcorpus_valid_legacy', 'refs'): [
        DATA_DIR / f'test_sets/turkcorpus/legacy/tune.8turkers.tok.turk.{i}' for i in range(8)
    ],
    ('pwkp_test', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.test.orig',
    ('pwkp_test', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.test.simp'],
    ('pwkp_valid', 'orig'): DATA_DIR / f'test_sets/pwkp/pwkp.valid.orig',
    ('pwkp_valid', 'refs'): [DATA_DIR / f'test_sets/pwkp/pwkp.valid.simp'],
    ('hsplit_test', 'orig'): DATA_DIR / f'test_sets/hsplit/hsplit.tok.src',
    ('hsplit_test', 'refs'): [DATA_DIR / f'test_sets/hsplit/hsplit.tok.{i+1}' for i in range(4)],
    ('wikisplit_test', 'orig'): DATA_DIR / f'test_sets/wikisplit/wikisplit.test.untok.orig',
    ('wikisplit_test', 'refs'): [DATA_DIR / f'test_sets/wikisplit/wikisplit.test.untok.split'],
    ('wikisplit_valid', 'orig'): DATA_DIR / f'test_sets/wikisplit/wikisplit.valid.untok.orig',
    ('wikisplit_valid', 'refs'): [DATA_DIR / f'test_sets/wikisplit/wikisplit.valid.untok.split'],
    ('googlecomp_test', 'orig'): DATA_DIR / f'test_sets/googlecomp/googlecomp.test.orig',
    ('googlecomp_test', 'refs'): [DATA_DIR / f'test_sets/googlecomp/googlecomp.test.comp'],
    ('googlecomp_valid', 'orig'): DATA_DIR / f'test_sets/googlecomp/googlecomp.valid.orig',
    ('googlecomp_valid', 'refs'): [DATA_DIR / f'test_sets/googlecomp/googlecomp.valid.comp'],
    ('qats_test', 'orig'): DATA_DIR / f'test_sets/qats/qats.test.orig',
    ('qats_test', 'refs'): [DATA_DIR / f'test_sets/qats/qats.test.simp'],
}
SYSTEM_OUTPUTS_DIR = DATA_DIR / "system_outputs"
SYSTEM_OUTPUTS_DIRS_MAP = {
    "turkcorpus_test": SYSTEM_OUTPUTS_DIR / "turkcorpus/test",
    "turkcorpus_valid": SYSTEM_OUTPUTS_DIR / "turkcorpus/valid",
    "pwkp_test": SYSTEM_OUTPUTS_DIR / "pwkp/test",
}

# Constants
VALID_TEST_SETS = list(set([test_set for test_set, language in TEST_SETS_PATHS.keys()])) + ['custom']
VALID_METRICS = [
    'bleu',
    'sari',
    'samsa',
    'fkgl',
    'sent_bleu',
    'f1_token',
    'sari_legacy',
    'sari_by_operation',
    'bertscore',
]
DEFAULT_METRICS = ['bleu', 'sari', 'fkgl']
