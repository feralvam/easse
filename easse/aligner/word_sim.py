from easse.aligner.config import *
from nltk.corpus.reader.wordlist import MWAPPDBCorpusReader

PPDB_PATH = 'resources/ppdb-1.0-xxxl-lexical.extended.synonyms.uniquepairs'


def loadPPDB(ppdb_file_name=PPDB_PATH):

    global ppdbSim
    global ppdbDict

    for entry in MWAPPDBCorpusReader(root="resources", fileids=ppdb_file_name).entries():
        ppdbDict[entry] = ppdbSim


def present_in_ppdb(word1, word2):

    global ppdbDict

    return ((word1.lower(), word2.lower()) in ppdbDict) or ((word2.lower(), word1.lower()) in ppdbDict)


def get_cannonical_word(word):
    if len(word) > 1:
        canonical_word = word.replace('.', '')
        canonical_word = canonical_word.replace('-', '')
        canonical_word = canonical_word.replace(',', '')
    else:
        canonical_word = word
    return canonical_word


def wordRelatedness(word1, pos1, word2, pos2):

    global stemmer
    global ppdbSim
    global punctuations

    canonical_word1 = get_cannonical_word(word1)
    canonical_word2 = get_cannonical_word(word2)

    if canonical_word1.lower() == canonical_word2.lower():
        return 1

    if stemmer.stem(word1).lower() == stemmer.stem(word2).lower():
        return 1

    if canonical_word1.isdigit() and canonical_word2.isdigit() and canonical_word1 != canonical_word2:
        return 0

    if (
        pos1.lower() == 'cd'
        and pos2.lower() == 'cd'
        and (not canonical_word1.isdigit() and not canonical_word2.isdigit())
        and canonical_word1 != canonical_word2
    ):
        return 0

    # stopwords can be similar to only stopwords
    if (word1.lower() in stopwords and word2.lower() not in stopwords) or (
        word1.lower() not in stopwords and word2.lower() in stopwords
    ):
        return 0

    # punctuations can only be either identical or totally dissimilar
    if word1 in punctuations or word2 in punctuations:
        return 0

    if present_in_ppdb(word1.lower(), word2.lower()):
        return ppdbSim
    else:
        return 0


# loadPPDB()
