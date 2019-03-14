from setuptools import setup, find_packages

setup(
    name='easse',
    version='0.1',
    description='Easier Automatic Sentence Simplification Evaluation',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    author='Fernando Alva-Manchego',
    author_email='feralvam@gmail.com',
    url='https://github.com/feralvam/easse',
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    entry_points={'console_scripts': [
        "easse = easse.cli.cli:cli"
    ]},
)
