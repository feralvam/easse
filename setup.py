from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='easse',
    version='0.1',
    description='Easier Automatic Sentence Simplification Evaluation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    author='Fernando Alva-Manchego <feralvam@gmail.com>, Louis Martin <louismartincs@gmail.com>',
    url='https://github.com/feralvam/easse',
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    entry_points={'console_scripts': [
        "easse = easse.cli:cli"
    ]},
    install_requires=[
        'sacrebleu', 'sacremoses', 'stanfordnlp', 'tupa>=1.3.10', 'nltk', 'click',
        'tseval @ git+https://github.com/facebookresearch/text-simplification-evaluation.git',
        ],
)
