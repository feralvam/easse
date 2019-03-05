from easse.aligner.aligner import *

# aligning strings (output indexes start at 1)
sentence1 = "Four men died in an accident."
sentence2 = "4 people are dead from a collision."

alignments = align(sentence1, sentence2)

print(alignments[0])
print(alignments[1])
print()


# aligning sets of tokens (output indexes start at 1)
sentence1 = ['Four', 'men', 'died', 'in', 'an', 'accident', '.']
sentence2 = ['4', 'people', 'are', 'dead', 'from', 'a', 'collision', '.']

alignments = align(sentence1, sentence2)

print(alignments[0])
print(alignments[1])


