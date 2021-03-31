from easse.aligner.config import *


def isSublist(A, B):
    # returns True if A is a sublist of B, False otherwise
    return set(A).issubset(set(B))


def findAllCommonContiguousSublists(A, B, turnToLowerCases=True):
    # this is a very inefficient implementation, you can use suffix trees to devise a much faster method
    # returns all the contiguous sublists in order of decreasing length
    # output format (0-indexed):
    # [
    #    [[indices in 'A' for utils sublist 1], [indices in 'B' for utils sublist 1]],
    #    ...,
    #    [[indices in 'A' for utils sublist n], [indices in 'B' for utils sublist n]]
    # ]

    a = []
    b = []
    for item in A:
        a.append(item)
    for item in B:
        b.append(item)

    if turnToLowerCases:
        for i in range(len(a)):
            a[i] = a[i].lower()
        for i in range(len(b)):
            b[i] = b[i].lower()

    commonContiguousSublists = []

    swapped = False
    if len(a) > len(b):
        temp = a
        a = b
        b = temp
        swapped = True

    maxSize = len(a)
    for size in range(maxSize, 0, -1):
        startingIndicesForA = [item for item in range(0, len(a) - size + 1)]
        startingIndicesForB = [item for item in range(0, len(b) - size + 1)]
        for i in startingIndicesForA:
            for j in startingIndicesForB:
                if a[i : i + size] == b[j : j + size]:
                    # check if a contiguous superset has already been inserted; don't insert this one in that case
                    alreadyInserted = False
                    currentAIndices = [item for item in range(i, i + size)]
                    currentBIndices = [item for item in range(j, j + size)]
                    for item in commonContiguousSublists:
                        if isSublist(currentAIndices, item[0]) and isSublist(currentBIndices, item[1]):
                            alreadyInserted = True
                            break
                    if not alreadyInserted:
                        commonContiguousSublists.append([currentAIndices, currentBIndices])

    if swapped:
        for item in commonContiguousSublists:
            temp = item[0]
            item[0] = item[1]
            item[1] = temp

    return commonContiguousSublists


def findTextualNeighborhood(sentenceDetails, wordIndex, leftSpan, rightSpan):
    # return the lemmas in the span [wordIndex-leftSpan, wordIndex+rightSpan]
    # and the positions actually available, left and right

    global punctuations

    sentenceLength = len(sentenceDetails)

    startWordIndex = max(1, wordIndex - leftSpan)
    endWordIndex = min(sentenceLength, wordIndex + rightSpan)

    lemmas = []
    wordIndices = []
    for item in sentenceDetails[startWordIndex - 1 : wordIndex - 1]:
        if item[3] not in stopwords + punctuations:
            lemmas.append(item[3])
            wordIndices.append(item[1])
    for item in sentenceDetails[wordIndex:endWordIndex]:
        if item[3] not in stopwords + punctuations:
            lemmas.append(item[3])
            wordIndices.append(item[1])
    return [wordIndices, lemmas, wordIndex - startWordIndex, endWordIndex - wordIndex]


def isAcronym(word, namedEntity):
    # returns whether 'word' is an acronym of 'namedEntity', which is a list of the component words
    canonicalWord = word.replace('.', '')
    if not canonicalWord.isupper() or len(canonicalWord) != len(namedEntity) or canonicalWord.lower() in ['a', 'i']:
        return False

    acronym = True
    for i in range(len(canonicalWord)):
        if canonicalWord[i] != namedEntity[i][0]:
            acronym = False
            break

    return acronym


def group_sentence_alignments(sent1_parse_lst, sent2_parse_lst, sent_aligns):
    sent1_parse = []
    sent2_parse = []
    sent1_map = {}
    sent2_map = {}
    for index_pair in sent_aligns:
        sent1_index, sent2_index = map(int, index_pair.strip().split('\t'))
        print(sent1_index, sent2_index)
        sent1_added = sent1_index in sent1_map
        sent2_added = sent2_index in sent2_map

        if sent1_added and not sent2_added:  # it is a split
            sent_group_index = sent1_map[sent1_index]
            if len(sent1_parse[sent_group_index]) > 1:  # check to prevent M-to-N alignments
                sent_group_index = len(sent1_parse)
                sent1_map[sent1_index] = sent_group_index
                sent2_map[sent2_index] = sent_group_index
                sent2_parse.insert(sent_group_index, [sent2_parse_lst[sent2_index]])
                sent1_parse.insert(sent_group_index, [sent1_parse_lst[sent1_index]])
            else:
                sent2_map[sent2_index] = sent_group_index
                sent2_parse[sent_group_index].append(sent2_parse_lst[sent2_index])
        elif sent2_added and not sent1_added:  # it is a join
            sent_group_index = sent2_map[sent2_index]
            if len(sent2_parse[sent_group_index]) > 1:  # check to prevent M-to-N alignments
                sent_group_index = len(sent2_parse)
                sent1_map[sent1_index] = sent_group_index
                sent2_map[sent2_index] = sent_group_index
                sent1_parse.insert(sent_group_index, [sent1_parse_lst[sent1_index]])
                sent2_parse.insert(sent_group_index, [sent2_parse_lst[sent2_index]])
            else:
                sent1_map[sent1_index] = sent_group_index
                sent1_parse[sent_group_index].append(sent1_parse_lst[sent1_index])
        else:
            # not sent1_added and not sent2_added: it is a new pair. We treat as 1-to-1
            # sent1_added and sent2_added: it is a M-to-N (not supported). We add it as a 1-to-1
            sent_group_index = len(sent1_parse)
            sent1_map[sent1_index] = sent_group_index
            sent2_map[sent2_index] = sent_group_index
            sent1_parse.insert(sent_group_index, [sent1_parse_lst[sent1_index]])
            sent2_parse.insert(sent_group_index, [sent2_parse_lst[sent2_index]])

    return zip(sent1_parse, sent2_parse)
