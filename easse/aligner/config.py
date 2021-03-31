from nltk.corpus import stopwords
from nltk import SnowballStemmer

ppdbDict = {}
ppdbSim = 0.9
theta1 = 0.9

stemmer = SnowballStemmer('english')

punctuations = [
    '(',
    '-lrb-',
    '.',
    ',',
    '-',
    '?',
    '!',
    ';',
    '_',
    ':',
    '{',
    '}',
    '[',
    '/',
    ']',
    '...',
    '"',
    '\'',
    ')',
    '-rrb-',
]

stopwords = stopwords.words('english')
