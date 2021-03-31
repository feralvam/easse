from easse.aligner.word_sim import *
from easse.aligner.utils import *
from easse.aligner.corenlp_utils import *


def alignNouns(source, target, sourceParseResult, targetParseResult, existingAlignments):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

    global ppdbSim
    global theta1

    nounAlignments = []

    sourceWordIndices = [i + 1 for i in range(len(source))]
    targetWordIndices = [i + 1 for i in range(len(target))]

    sourceWordIndicesAlreadyAligned = sorted(list(set([item[0] for item in existingAlignments])))
    targetWordIndicesAlreadyAligned = sorted(list(set([item[1] for item in existingAlignments])))

    sourceWords = [item[2] for item in source]
    targetWords = [item[2] for item in target]

    sourceLemmas = [item[3] for item in source]
    targetLemmas = [item[3] for item in target]

    sourcePosTags = [item[4] for item in source]
    targetPosTags = [item[4] for item in target]

    sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
    targetDParse = dependencyParseAndPutOffsets(targetParseResult)

    numberOfNounsInSource = 0

    evidenceCountsMatrix = {}
    relativeAlignmentsMatrix = {}
    wordSimilarities = {}

    # construct the two matrices in the following loop
    for i in sourceWordIndices:
        if i in sourceWordIndicesAlreadyAligned or (
            sourcePosTags[i - 1][0].lower() != "n" and sourcePosTags[i - 1].lower() != "prp"
        ):
            continue

        numberOfNounsInSource += 1

        for j in targetWordIndices:
            if j in targetWordIndicesAlreadyAligned or (
                targetPosTags[j - 1][0].lower() != "n" and targetPosTags[j - 1].lower() != "prp"
            ):
                continue

            if (
                max(
                    wordRelatedness(
                        sourceWords[i - 1],
                        sourcePosTags[i - 1],
                        targetWords[j - 1],
                        targetPosTags[j - 1],
                    ),
                    wordRelatedness(
                        sourceLemmas[i - 1],
                        sourcePosTags[i - 1],
                        targetLemmas[j - 1],
                        targetPosTags[j - 1],
                    ),
                )
                < ppdbSim
            ):
                continue

            wordSimilarities[(i, j)] = max(
                wordRelatedness(
                    sourceWords[i - 1],
                    sourcePosTags[i - 1],
                    targetWords[j - 1],
                    targetPosTags[j - 1],
                ),
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                ),
            )

            sourceWordParents = findParents(sourceDParse, i, sourceWords[i - 1])
            sourceWordChildren = findChildren(sourceDParse, i, sourceWords[i - 1])
            targetWordParents = findParents(targetDParse, j, targetWords[j - 1])
            targetWordChildren = findChildren(targetDParse, j, targetWords[j - 1])

            # search for utils or equivalent parents
            groupOfSimilarRelationsForNounParent = [
                "pos",
                "nn",
                "prep_of",
                "prep_in",
                "prep_at",
                "prep_for",
            ]
            group1OfSimilarRelationsForVerbParent = ["agent", "nsubj", "xsubj"]
            group2OfSimilarRelationsForVerbParent = [
                "ccomp",
                "dobj",
                "nsubjpass",
                "rel",
                "partmod",
            ]
            group3OfSimilarRelationsForVerbParent = [
                "tmod" "prep_in",
                "prep_at",
                "prep_on",
            ]
            group4OfSimilarRelationsForVerbParent = ["iobj", "prep_to"]

            for ktem in sourceWordParents:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + nounAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsForNounParent
                            and ltem[2] in groupOfSimilarRelationsForNounParent
                        )
                        or (
                            ktem[2] in group1OfSimilarRelationsForVerbParent
                            and ltem[2] in group1OfSimilarRelationsForVerbParent
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsForVerbParent
                            and ltem[2] in group2OfSimilarRelationsForVerbParent
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsForVerbParent
                            and ltem[2] in group3OfSimilarRelationsForVerbParent
                        )
                        or (
                            ktem[2] in group4OfSimilarRelationsForVerbParent
                            and ltem[2] in group4OfSimilarRelationsForVerbParent
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for utils or equivalent children
            groupOfSimilarRelationsForNounChild = [
                "pos",
                "nn" "prep_of",
                "prep_in",
                "prep_at",
                "prep_for",
            ]
            groupOfSimilarRelationsForVerbChild = ["infmod", "partmod", "rcmod"]
            groupOfSimilarRelationsForAdjectiveChild = ["amod", "rcmod"]

            for ktem in sourceWordChildren:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + nounAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsForNounChild
                            and ltem[2] in groupOfSimilarRelationsForNounChild
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsForVerbChild
                            and ltem[2] in groupOfSimilarRelationsForVerbChild
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsForAdjectiveChild
                            and ltem[2] in groupOfSimilarRelationsForAdjectiveChild
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent parent-child relations
            groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild = [
                ["nsubj"],
                ["amod", "rcmod"],
            ]
            groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild = [
                ["ccomp", "dobj", "nsubjpass", "rel", "partmod"],
                ["infmod", "partmod", "rcmod"],
            ]
            group1OfSimilarRelationsInOppositeDirectionForNounParentAndChild = [
                ["conj_and"],
                ["conj_and"],
            ]
            group2OfSimilarRelationsInOppositeDirectionForNounParentAndChild = [
                ["conj_or"],
                ["conj_or"],
            ]
            group3OfSimilarRelationsInOppositeDirectionForNounParentAndChild = [
                ["conj_nor"],
                ["conj_nor"],
            ]

            for ktem in sourceWordParents:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + nounAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                        )
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent child-parent relations
            for ktem in sourceWordChildren:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + nounAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                        )
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

    # now use the collected stats to align
    for n in range(numberOfNounsInSource):

        maxEvidenceCountForCurrentPass = 0
        maxOverallValueForCurrentPass = 0
        indexPairWithStrongestTieForCurrentPass = [-1, -1]

        for i in sourceWordIndices:
            if (
                i in sourceWordIndicesAlreadyAligned
                or sourcePosTags[i - 1][0].lower() != "n"
                or sourceLemmas[i - 1] in stopwords
            ):
                continue

            for j in targetWordIndices:
                if (
                    j in targetWordIndicesAlreadyAligned
                    or targetPosTags[j - 1][0].lower() != "n"
                    or targetLemmas[j - 1] in stopwords
                ):
                    continue

                if (i, j) in evidenceCountsMatrix and theta1 * wordSimilarities[(i, j)] + (
                    1 - theta1
                ) * evidenceCountsMatrix[(i, j)] > maxOverallValueForCurrentPass:
                    maxOverallValueForCurrentPass = (
                        theta1 * wordSimilarities[(i, j)] + (1 - theta1) * evidenceCountsMatrix[(i, j)]
                    )
                    maxEvidenceCountForCurrentPass = evidenceCountsMatrix[(i, j)]
                    indexPairWithStrongestTieForCurrentPass = [i, j]

        if maxEvidenceCountForCurrentPass > 0:
            nounAlignments.append(indexPairWithStrongestTieForCurrentPass)
            sourceWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[0])
            targetWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[1])
            for item in relativeAlignmentsMatrix[
                (
                    indexPairWithStrongestTieForCurrentPass[0],
                    indexPairWithStrongestTieForCurrentPass[1],
                )
            ]:
                if (
                    item[0] != 0
                    and item[1] != 0
                    and item[0] not in sourceWordIndicesAlreadyAligned
                    and item[1] not in targetWordIndicesAlreadyAligned
                ):
                    nounAlignments.append(item)
                    sourceWordIndicesAlreadyAligned.append(item[0])
                    targetWordIndicesAlreadyAligned.append(item[1])
        else:
            break

    return nounAlignments


def alignMainVerbs(source, target, sourceParseResult, targetParseResult, existingAlignments):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

    global ppdbSim
    global theta1

    mainVerbAlignments = []

    sourceWordIndices = [i + 1 for i in range(len(source))]
    targetWordIndices = [i + 1 for i in range(len(target))]

    sourceWordIndicesAlreadyAligned = sorted(list(set([item[0] for item in existingAlignments])))
    targetWordIndicesAlreadyAligned = sorted(list(set([item[1] for item in existingAlignments])))

    sourceWords = [item[2] for item in source]
    targetWords = [item[2] for item in target]

    sourceLemmas = [item[3] for item in source]
    targetLemmas = [item[3] for item in target]

    sourcePosTags = [item[4] for item in source]
    targetPosTags = [item[4] for item in target]

    sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
    targetDParse = dependencyParseAndPutOffsets(targetParseResult)

    numberOfMainVerbsInSource = 0

    evidenceCountsMatrix = {}
    relativeAlignmentsMatrix = {}
    wordSimilarities = {}

    # construct the two matrices in the following loop
    for i in sourceWordIndices:
        if (
            i in sourceWordIndicesAlreadyAligned
            or sourcePosTags[i - 1][0].lower() != "v"
            or sourceLemmas[i - 1] in stopwords
        ):
            continue

        numberOfMainVerbsInSource += 1

        for j in targetWordIndices:
            if (
                j in targetWordIndicesAlreadyAligned
                or targetPosTags[j - 1][0].lower() != "v"
                or targetLemmas[j - 1] in stopwords
            ):
                continue

            if (
                max(
                    wordRelatedness(
                        sourceWords[i - 1],
                        sourcePosTags[i - 1],
                        targetWords[j - 1],
                        targetPosTags[j - 1],
                    ),
                    wordRelatedness(
                        sourceLemmas[i - 1],
                        sourcePosTags[i - 1],
                        targetLemmas[j - 1],
                        targetPosTags[j - 1],
                    ),
                )
                < ppdbSim
            ):
                continue

            wordSimilarities[(i, j)] = max(
                wordRelatedness(
                    sourceWords[i - 1],
                    sourcePosTags[i - 1],
                    targetWords[j - 1],
                    targetPosTags[j - 1],
                ),
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                ),
            )

            sourceWordParents = findParents(sourceDParse, i, sourceWords[i - 1])
            sourceWordChildren = findChildren(sourceDParse, i, sourceWords[i - 1])
            targetWordParents = findParents(targetDParse, j, targetWords[j - 1])
            targetWordChildren = findChildren(targetDParse, j, targetWords[j - 1])

            # search for utils or equivalent children
            group1OfSimilarRelationsForNounChild = ["agent", "nsubj" "xsubj"]
            group2OfSimilarRelationsForNounChild = [
                "ccomp",
                "dobj" "nsubjpass",
                "rel",
                "partmod",
            ]
            group3OfSimilarRelationsForNounChild = [
                "tmod",
                "prep_in",
                "prep_at",
                "prep_on",
            ]
            group4OfSimilarRelationsForNounChild = ["iobj", "prep_to"]
            groupOfSimilarRelationsForVerbChild = ["purpcl", "xcomp"]

            for ktem in sourceWordChildren:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + mainVerbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in group1OfSimilarRelationsForNounChild
                            and ltem[2] in group1OfSimilarRelationsForNounChild
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsForNounChild
                            and ltem[2] in group2OfSimilarRelationsForNounChild
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsForNounChild
                            and ltem[2] in group3OfSimilarRelationsForNounChild
                        )
                        or (
                            ktem[2] in group4OfSimilarRelationsForNounChild
                            and ltem[2] in group4OfSimilarRelationsForNounChild
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsForVerbChild
                            and ltem[2] in groupOfSimilarRelationsForVerbChild
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for utils or equivalent parents
            groupOfSimilarRelationsForNounParent = [
                "infmod",
                "partmod",
                "rcmod",
            ]
            groupOfSimilarRelationsForVerbParent = ["purpcl", "xcomp"]

            for ktem in sourceWordParents:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + mainVerbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsForNounParent
                            and ltem[2] in groupOfSimilarRelationsForNounParent
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsForVerbParent
                            and ltem[2] in groupOfSimilarRelationsForVerbParent
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent parent-child pairs
            groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild = [
                ["cop", "csubj"],
                ["acomp"],
            ]
            group1OfSimilarRelationsInOppositeDirectionForVerbParentAndChild = [
                ["csubj"],
                ["csubjpass"],
            ]
            group2OfSimilarRelationsInOppositeDirectionForVerbParentAndChild = [
                ["conj_and"],
                ["conj_and"],
            ]
            group3OfSimilarRelationsInOppositeDirectionForVerbParentAndChild = [
                ["conj_or"],
                ["conj_or"],
            ]
            group4OfSimilarRelationsInOppositeDirectionForVerbParentAndChild = [
                ["conj_nor"],
                ["conj_nor"],
            ]

            for ktem in sourceWordParents:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + mainVerbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                        )
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                        )
                        or (
                            ktem[2] in group4OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                            and ltem[2] in group4OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent child-parent pairs
            for ktem in sourceWordChildren:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + mainVerbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                        )
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                        )
                        or (
                            ktem[2] in group4OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                            and ltem[2] in group4OfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

    # now use the collected stats to align
    for n in range(numberOfMainVerbsInSource):

        maxEvidenceCountForCurrentPass = 0
        maxOverallValueForCurrentPass = 0
        indexPairWithStrongestTieForCurrentPass = [-1, -1]

        for i in sourceWordIndices:
            if (
                i in sourceWordIndicesAlreadyAligned
                or sourcePosTags[i - 1][0].lower() != "v"
                or sourceLemmas[i - 1] in stopwords
            ):
                continue

            for j in targetWordIndices:
                if (
                    j in targetWordIndicesAlreadyAligned
                    or targetPosTags[j - 1][0].lower() != "v"
                    or targetLemmas[j - 1] in stopwords
                ):
                    continue

                if (i, j) in evidenceCountsMatrix and theta1 * wordSimilarities[(i, j)] + (
                    1 - theta1
                ) * evidenceCountsMatrix[(i, j)] > maxOverallValueForCurrentPass:
                    maxOverallValueForCurrentPass = (
                        theta1 * wordSimilarities[(i, j)] + (1 - theta1) * evidenceCountsMatrix[(i, j)]
                    )
                    maxEvidenceCountForCurrentPass = evidenceCountsMatrix[(i, j)]
                    indexPairWithStrongestTieForCurrentPass = [i, j]

        if maxEvidenceCountForCurrentPass > 0:
            mainVerbAlignments.append(indexPairWithStrongestTieForCurrentPass)
            sourceWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[0])
            targetWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[1])
            for item in relativeAlignmentsMatrix[
                (
                    indexPairWithStrongestTieForCurrentPass[0],
                    indexPairWithStrongestTieForCurrentPass[1],
                )
            ]:
                if (
                    item[0] != 0
                    and item[1] != 0
                    and item[0] not in sourceWordIndicesAlreadyAligned
                    and item[1] not in targetWordIndicesAlreadyAligned
                ):
                    mainVerbAlignments.append(item)
                    sourceWordIndicesAlreadyAligned.append(item[0])
                    targetWordIndicesAlreadyAligned.append(item[1])
        else:
            break

    return mainVerbAlignments


def alignAdjectives(source, target, sourceParseResult, targetParseResult, existingAlignments):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

    global ppdbSim
    global theta1

    adjectiveAlignments = []

    sourceWordIndices = [i + 1 for i in range(len(source))]
    targetWordIndices = [i + 1 for i in range(len(target))]

    sourceWordIndicesAlreadyAligned = sorted(list(set([item[0] for item in existingAlignments])))
    targetWordIndicesAlreadyAligned = sorted(list(set([item[1] for item in existingAlignments])))

    sourceWords = [item[2] for item in source]
    targetWords = [item[2] for item in target]

    sourceLemmas = [item[3] for item in source]
    targetLemmas = [item[3] for item in target]

    sourcePosTags = [item[4] for item in source]
    targetPosTags = [item[4] for item in target]

    sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
    targetDParse = dependencyParseAndPutOffsets(targetParseResult)

    numberOfAdjectivesInSource = 0

    evidenceCountsMatrix = {}
    relativeAlignmentsMatrix = {}
    wordSimilarities = {}

    # construct the two matrices in the following loop
    for i in sourceWordIndices:
        if i in sourceWordIndicesAlreadyAligned or sourcePosTags[i - 1][0].lower() != "j":
            continue

        numberOfAdjectivesInSource += 1

        for j in targetWordIndices:
            if j in targetWordIndicesAlreadyAligned or targetPosTags[j - 1][0].lower() != "j":
                continue

            if (
                max(
                    wordRelatedness(
                        sourceWords[i - 1],
                        sourcePosTags[i - 1],
                        targetWords[j - 1],
                        targetPosTags[j - 1],
                    ),
                    wordRelatedness(
                        sourceLemmas[i - 1],
                        sourcePosTags[i - 1],
                        targetLemmas[j - 1],
                        targetPosTags[j - 1],
                    ),
                )
                < ppdbSim
            ):
                continue

            wordSimilarities[(i, j)] = max(
                wordRelatedness(
                    sourceWords[i - 1],
                    sourcePosTags[i - 1],
                    targetWords[j - 1],
                    targetPosTags[j - 1],
                ),
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                ),
            )

            sourceWordParents = findParents(sourceDParse, i, sourceWords[i - 1])
            sourceWordChildren = findChildren(sourceDParse, i, sourceWords[i - 1])
            targetWordParents = findParents(targetDParse, j, targetWords[j - 1])
            targetWordChildren = findChildren(targetDParse, j, targetWords[j - 1])

            # search for utils or equivalent parents
            groupOfSimilarRelationsForNounParent = ["amod", "rcmod"]

            for ktem in sourceWordParents:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adjectiveAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsForNounParent
                            and ltem[2] in groupOfSimilarRelationsForNounParent
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for utils children
            for ktem in sourceWordChildren:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adjectiveAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (ktem[2] == ltem[2]):
                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent parent-child pair
            groupOfSimilarRelationsInOppositeDirectionForNounParentAndChild = [
                ["amod", "rcmod"],
                ["nsubj"],
            ]
            groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild = [
                ["acomp"],
                ["cop", "csubj"],
            ]
            group1OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild = [
                ["conj_and"],
                ["conj_and"],
            ]
            group2OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild = [
                ["conj_or"],
                ["conj_or"],
            ]
            group3OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild = [
                ["conj_nor"],
                ["conj_nor"],
            ]

            for ktem in sourceWordParents:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adjectiveAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                        )
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent child-parent pair
            for ktem in sourceWordChildren:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adjectiveAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForNounParentAndChild[1]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForNounParentAndChild[0]
                        )
                        or (
                            ktem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[1]
                            and ltem[2] in groupOfSimilarRelationsInOppositeDirectionForVerbParentAndChild[0]
                        )
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[1]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForAdjectiveParentAndChild[0]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

    # now use the collected stats to align
    for n in range(numberOfAdjectivesInSource):

        maxEvidenceCountForCurrentPass = 0
        maxOverallValueForCurrentPass = 0
        indexPairWithStrongestTieForCurrentPass = [-1, -1]

        for i in sourceWordIndices:
            if (
                i in sourceWordIndicesAlreadyAligned
                or sourcePosTags[i - 1][0].lower() != "j"
                or sourceLemmas[i - 1] in stopwords
            ):
                continue

            for j in targetWordIndices:
                if (
                    j in targetWordIndicesAlreadyAligned
                    or targetPosTags[j - 1][0].lower() != "j"
                    or targetLemmas[j - 1] in stopwords
                ):
                    continue

                if (i, j) in evidenceCountsMatrix and theta1 * wordSimilarities[(i, j)] + (
                    1 - theta1
                ) * evidenceCountsMatrix[(i, j)] > maxOverallValueForCurrentPass:
                    maxOverallValueForCurrentPass = (
                        theta1 * wordSimilarities[(i, j)] + (1 - theta1) * evidenceCountsMatrix[(i, j)]
                    )
                    maxEvidenceCountForCurrentPass = evidenceCountsMatrix[(i, j)]
                    indexPairWithStrongestTieForCurrentPass = [i, j]

        if maxEvidenceCountForCurrentPass > 0:
            adjectiveAlignments.append(indexPairWithStrongestTieForCurrentPass)
            sourceWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[0])
            targetWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[1])
            for item in relativeAlignmentsMatrix[
                (
                    indexPairWithStrongestTieForCurrentPass[0],
                    indexPairWithStrongestTieForCurrentPass[1],
                )
            ]:
                if (
                    item[0] != 0
                    and item[1] != 0
                    and item[0] not in sourceWordIndicesAlreadyAligned
                    and item[1] not in targetWordIndicesAlreadyAligned
                ):
                    adjectiveAlignments.append(item)
                    sourceWordIndicesAlreadyAligned.append(item[0])
                    targetWordIndicesAlreadyAligned.append(item[1])
        else:
            break

    return adjectiveAlignments


def alignAdverbs(source, target, sourceParseResult, targetParseResult, existingAlignments):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

    global ppdbSim
    global theta1

    adverbAlignments = []

    sourceWordIndices = [i + 1 for i in range(len(source))]
    targetWordIndices = [i + 1 for i in range(len(target))]

    sourceWordIndicesAlreadyAligned = sorted(list(set([item[0] for item in existingAlignments])))
    targetWordIndicesAlreadyAligned = sorted(list(set([item[1] for item in existingAlignments])))

    sourceWords = [item[2] for item in source]
    targetWords = [item[2] for item in target]

    sourceLemmas = [item[3] for item in source]
    targetLemmas = [item[3] for item in target]

    sourcePosTags = [item[4] for item in source]
    targetPosTags = [item[4] for item in target]

    sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
    targetDParse = dependencyParseAndPutOffsets(targetParseResult)

    numberOfAdverbsInSource = 0

    evidenceCountsMatrix = {}
    relativeAlignmentsMatrix = {}
    wordSimilarities = {}

    for i in sourceWordIndices:
        if i in sourceWordIndicesAlreadyAligned or (sourcePosTags[i - 1][0].lower() != "r"):
            continue

        numberOfAdverbsInSource += 1

        for j in targetWordIndices:
            if j in targetWordIndicesAlreadyAligned or (targetPosTags[j - 1][0].lower() != "r"):
                continue

            if (
                max(
                    wordRelatedness(
                        sourceWords[i - 1],
                        sourcePosTags[i - 1],
                        targetWords[j - 1],
                        targetPosTags[j - 1],
                    ),
                    wordRelatedness(
                        sourceLemmas[i - 1],
                        sourcePosTags[i - 1],
                        targetLemmas[j - 1],
                        targetPosTags[j - 1],
                    ),
                )
                < ppdbSim
            ):
                continue

            wordSimilarities[(i, j)] = max(
                wordRelatedness(
                    sourceWords[i - 1],
                    sourcePosTags[i - 1],
                    targetWords[j - 1],
                    targetPosTags[j - 1],
                ),
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                ),
            )

            sourceWordParents = findParents(sourceDParse, i, sourceWords[i - 1])
            sourceWordChildren = findChildren(sourceDParse, i, sourceWords[i - 1])
            targetWordParents = findParents(targetDParse, j, targetWords[j - 1])
            targetWordChildren = findChildren(targetDParse, j, targetWords[j - 1])

            # search for utils parents
            for ktem in sourceWordParents:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adverbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (ktem[2] == ltem[2]):
                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for utils children
            for ktem in sourceWordChildren:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adverbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (ktem[2] == ltem[2]):
                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent parent-child relationships
            group1OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild = [
                ["conj_and"],
                ["conj_and"],
            ]
            group2OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild = [
                ["conj_or"],
                ["conj_or"],
            ]
            group3OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild = [
                ["conj_nor"],
                ["conj_nor"],
            ]

            for ktem in sourceWordParents:
                for ltem in targetWordChildren:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adverbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[0]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[1]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[0]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[1]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[0]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[1]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

            # search for equivalent child-parent relationships
            for ktem in sourceWordChildren:
                for ltem in targetWordParents:
                    if (
                        (ktem[0], ltem[0]) in existingAlignments + adverbAlignments
                        or max(
                            wordRelatedness(
                                ktem[1],
                                sourcePosTags[ktem[0] - 1],
                                ltem[1],
                                targetPosTags[ltem[0] - 1],
                            ),
                            wordRelatedness(
                                sourceLemmas[ktem[0] - 1],
                                sourcePosTags[ktem[0] - 1],
                                targetLemmas[ltem[0] - 1],
                                targetPosTags[ltem[0] - 1],
                            ),
                        )
                        >= ppdbSim
                    ) and (
                        (ktem[2] == ltem[2])
                        or (
                            ktem[2] in group1OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[1]
                            and ltem[2] in group1OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[0]
                        )
                        or (
                            ktem[2] in group2OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[1]
                            and ltem[2] in group2OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[0]
                        )
                        or (
                            ktem[2] in group3OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[1]
                            and ltem[2] in group3OfSimilarRelationsInOppositeDirectionForAdverbParentAndChild[0]
                        )
                    ):

                        if (i, j) in evidenceCountsMatrix:
                            evidenceCountsMatrix[(i, j)] += max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )
                        else:
                            evidenceCountsMatrix[(i, j)] = max(
                                wordRelatedness(
                                    ktem[1],
                                    sourcePosTags[ktem[0] - 1],
                                    ltem[1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                                wordRelatedness(
                                    sourceLemmas[ktem[0] - 1],
                                    sourcePosTags[ktem[0] - 1],
                                    targetLemmas[ltem[0] - 1],
                                    targetPosTags[ltem[0] - 1],
                                ),
                            )

                        if (i, j) in relativeAlignmentsMatrix:
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])
                        else:
                            relativeAlignmentsMatrix[(i, j)] = []
                            relativeAlignmentsMatrix[(i, j)].append([ktem[0], ltem[0]])

    # now use the collected stats to align
    for n in range(numberOfAdverbsInSource):

        maxEvidenceCountForCurrentPass = 0
        maxOverallValueForCurrentPass = 0
        indexPairWithStrongestTieForCurrentPass = [-1, -1]

        for i in sourceWordIndices:
            if (
                i in sourceWordIndicesAlreadyAligned
                or sourcePosTags[i - 1][0].lower() != "r"
                or sourceLemmas[i - 1] in stopwords
            ):
                continue

            for j in targetWordIndices:
                if (
                    j in targetWordIndicesAlreadyAligned
                    or targetPosTags[j - 1][0].lower() != "r"
                    or targetLemmas[j - 1] in stopwords
                ):
                    continue

                if (i, j) in evidenceCountsMatrix and theta1 * wordSimilarities[(i, j)] + (
                    1 - theta1
                ) * evidenceCountsMatrix[(i, j)] > maxOverallValueForCurrentPass:
                    maxOverallValueForCurrentPass = (
                        theta1 * wordSimilarities[(i, j)] + (1 - theta1) * evidenceCountsMatrix[(i, j)]
                    )
                    maxEvidenceCountForCurrentPass = evidenceCountsMatrix[(i, j)]
                    indexPairWithStrongestTieForCurrentPass = [i, j]

        if maxEvidenceCountForCurrentPass > 0:
            adverbAlignments.append(indexPairWithStrongestTieForCurrentPass)
            sourceWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[0])
            targetWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[1])
            for item in relativeAlignmentsMatrix[
                (
                    indexPairWithStrongestTieForCurrentPass[0],
                    indexPairWithStrongestTieForCurrentPass[1],
                )
            ]:
                if (
                    item[0] != 0
                    and item[1] != 0
                    and item[0] not in sourceWordIndicesAlreadyAligned
                    and item[1] not in targetWordIndicesAlreadyAligned
                ):
                    adverbAlignments.append(item)
                    sourceWordIndicesAlreadyAligned.append(item[0])
                    targetWordIndicesAlreadyAligned.append(item[1])
        else:
            break

    return adverbAlignments


def alignNamedEntities(source, target, sourceParseResult, targetParseResult, existingAlignments):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

    global punctuations

    alignments = []

    sourceNamedEntities = ner(sourceParseResult)
    sourceNamedEntities = sorted(sourceNamedEntities, key=len)

    targetNamedEntities = ner(targetParseResult)
    targetNamedEntities = sorted(targetNamedEntities, key=len)

    # learn from the other sentence that a certain word/phrase is a named entity (learn for source from target)
    for item in source:
        alreadyIncluded = False
        for jtem in sourceNamedEntities:
            if item[1] in jtem[1]:
                alreadyIncluded = True
                break
        if alreadyIncluded or (len(item[2]) > 0 and not item[2][0].isupper()):
            continue
        for jtem in targetNamedEntities:
            if item[2] in jtem[2]:
                # construct the item
                newItem = [[item[0]], [item[1]], [item[2]], jtem[3]]

                # check if the current item is part of a named entity part of which has already been added (by checking contiguousness)
                partOfABiggerName = False
                for k in range(len(sourceNamedEntities)):
                    if sourceNamedEntities[k][1][len(sourceNamedEntities[k][1]) - 1] == newItem[1][0] - 1:
                        sourceNamedEntities[k][0].append(newItem[0][0])
                        sourceNamedEntities[k][1].append(newItem[1][0])
                        sourceNamedEntities[k][2].append(newItem[2][0])
                        partOfABiggerName = True
                if not partOfABiggerName:
                    sourceNamedEntities.append(newItem)
            elif isAcronym(item[2], jtem[2]) and [[item[0]], [item[1]], [item[2]], jtem[3]] not in sourceNamedEntities:
                sourceNamedEntities.append([[item[0]], [item[1]], [item[2]], jtem[3]])

    # learn from the other sentence that a certain word/phrase is a named entity (learn for target from source)
    for item in target:
        alreadyIncluded = False
        for jtem in targetNamedEntities:
            if item[1] in jtem[1]:
                alreadyIncluded = True
                break
        if alreadyIncluded or (len(item[2]) > 0 and not item[2][0].isupper()):
            continue
        for jtem in sourceNamedEntities:
            if item[2] in jtem[2]:
                # construct the item
                newItem = [[item[0]], [item[1]], [item[2]], jtem[3]]

                # check if the current item is part of a named entity part of which has already been added (by checking contiguousness)
                partOfABiggerName = False
                for k in range(len(targetNamedEntities)):
                    if targetNamedEntities[k][1][len(targetNamedEntities[k][1]) - 1] == newItem[1][0] - 1:
                        targetNamedEntities[k][0].append(newItem[0][0])
                        targetNamedEntities[k][1].append(newItem[1][0])
                        targetNamedEntities[k][2].append(newItem[2][0])
                        partOfABiggerName = True
                if not partOfABiggerName:
                    targetNamedEntities.append(newItem)
            elif isAcronym(item[2], jtem[2]) and [[item[0]], [item[1]], [item[2]], jtem[3]] not in targetNamedEntities:
                targetNamedEntities.append([[item[0]], [item[1]], [item[2]], jtem[3]])

    sourceWords = []
    targetWords = []

    for item in sourceNamedEntities:
        for jtem in item[1]:
            if item[3] in ["PERSON", "ORGANIZATION", "LOCATION"]:
                sourceWords.append(source[jtem - 1][2])
    for item in targetNamedEntities:
        for jtem in item[1]:
            if item[3] in ["PERSON", "ORGANIZATION", "LOCATION"]:
                targetWords.append(target[jtem - 1][2])

    if len(sourceNamedEntities) == 0 or len(targetNamedEntities) == 0:
        return []

    sourceNamedEntitiesAlreadyAligned = []
    targetNamedEntitiesAlreadyAligned = []

    # align all full matches
    for item in sourceNamedEntities:
        if item[3] not in ["PERSON", "ORGANIZATION", "LOCATION"]:
            continue

        # do not align if the current source entity is present more than once
        count = 0
        for ktem in sourceNamedEntities:
            if ktem[2] == item[2]:
                count += 1
        if count > 1:
            continue

        for jtem in targetNamedEntities:
            if jtem[3] not in ["PERSON", "ORGANIZATION", "LOCATION"]:
                continue

            # do not align if the current target entity is present more than once
            count = 0
            for ktem in targetNamedEntities:
                if ktem[2] == jtem[2]:
                    count += 1
            if count > 1:
                continue

            # get rid of dots and hyphens
            canonicalItemWord = [i.replace(".", "") for i in item[2]]
            canonicalItemWord = [i.replace("-", "") for i in item[2]]
            canonicalJtemWord = [j.replace(".", "") for j in jtem[2]]
            canonicalJtemWord = [j.replace("-", "") for j in jtem[2]]

            if canonicalItemWord == canonicalJtemWord:
                for k in range(len(item[1])):
                    if ([item[1][k], jtem[1][k]]) not in alignments:
                        alignments.append([item[1][k], jtem[1][k]])
                sourceNamedEntitiesAlreadyAligned.append(item)
                targetNamedEntitiesAlreadyAligned.append(jtem)

    # align acronyms with their elaborations
    for item in sourceNamedEntities:
        if item[3] not in ["PERSON", "ORGANIZATION", "LOCATION"]:
            continue
        for jtem in targetNamedEntities:
            if jtem[3] not in ["PERSON", "ORGANIZATION", "LOCATION"]:
                continue

            if len(item[2]) == 1 and isAcronym(item[2][0], jtem[2]):
                for i in range(len(jtem[1])):
                    if [item[1][0], jtem[1][i]] not in alignments:
                        alignments.append([item[1][0], jtem[1][i]])
                        sourceNamedEntitiesAlreadyAligned.append(item[1][0])
                        targetNamedEntitiesAlreadyAligned.append(jtem[1][i])

            elif len(jtem[2]) == 1 and isAcronym(jtem[2][0], item[2]):
                for i in range(len(item[1])):
                    if [item[1][i], jtem[1][0]] not in alignments:
                        alignments.append([item[1][i], jtem[1][0]])
                        sourceNamedEntitiesAlreadyAligned.append(item[1][i])
                        targetNamedEntitiesAlreadyAligned.append(jtem[1][0])

    # align subset matches
    for item in sourceNamedEntities:
        if item[3] not in ["PERSON", "ORGANIZATION", "LOCATION"] or item in sourceNamedEntitiesAlreadyAligned:
            continue

        # do not align if the current source entity is present more than once
        count = 0
        for ktem in sourceNamedEntities:
            if ktem[2] == item[2]:
                count += 1
        if count > 1:
            continue

        for jtem in targetNamedEntities:
            if jtem[3] not in ["PERSON", "ORGANIZATION", "LOCATION"] or jtem in targetNamedEntitiesAlreadyAligned:
                continue

            if item[3] != jtem[3]:
                continue

            # do not align if the current target entity is present more than once
            count = 0
            for ktem in targetNamedEntities:
                if ktem[2] == jtem[2]:
                    count += 1
            if count > 1:
                continue

            # find if the first is a part of the second
            if isSublist(item[2], jtem[2]):
                unalignedWordIndicesInTheLongerName = []
                for ktem in jtem[1]:
                    unalignedWordIndicesInTheLongerName.append(ktem)
                for k in range(len(item[2])):
                    for l in range(len(jtem[2])):
                        if item[2][k] == jtem[2][l] and [item[1][k], jtem[1][l]] not in alignments:
                            alignments.append([item[1][k], jtem[1][l]])
                            if jtem[1][l] in unalignedWordIndicesInTheLongerName:
                                unalignedWordIndicesInTheLongerName.remove(jtem[1][l])
                for k in range(len(item[1])):  # the shorter name
                    for l in range(len(jtem[1])):  # the longer name
                        # find if the current term in the longer name has already been aligned (before calling alignNamedEntities()), do not align it in that case
                        alreadyInserted = False
                        for mtem in existingAlignments:
                            if mtem[1] == jtem[1][l]:
                                alreadyInserted = True
                                break
                        if jtem[1][l] not in unalignedWordIndicesInTheLongerName or alreadyInserted:
                            continue
                        if (
                            [item[1][k], jtem[1][l]] not in alignments
                            and target[jtem[1][l] - 1][2] not in sourceWords
                            and item[2][k] not in punctuations
                            and jtem[2][l] not in punctuations
                        ):
                            alignments.append([item[1][k], jtem[1][l]])

            # else find if the second is a part of the first
            elif isSublist(jtem[2], item[2]):
                unalignedWordIndicesInTheLongerName = []
                for ktem in item[1]:
                    unalignedWordIndicesInTheLongerName.append(ktem)
                for k in range(len(jtem[2])):
                    for l in range(len(item[2])):
                        if jtem[2][k] == item[2][l] and [item[1][l], jtem[1][k]] not in alignments:
                            alignments.append([item[1][l], jtem[1][k]])
                            if item[1][l] in unalignedWordIndicesInTheLongerName:
                                unalignedWordIndicesInTheLongerName.remove(item[1][l])
                for k in range(len(jtem[1])):  # the shorter name
                    for l in range(len(item[1])):  # the longer name
                        # find if the current term in the longer name has already been aligned (before calling alignNamedEntities()), do not align it in that case
                        alreadyInserted = False
                        for mtem in existingAlignments:
                            if mtem[0] == item[1][k]:
                                alreadyInserted = True
                                break
                        if item[1][l] not in unalignedWordIndicesInTheLongerName or alreadyInserted:
                            continue
                        if (
                            [item[1][l], jtem[1][k]] not in alignments
                            and source[item[1][k] - 1][2] not in targetWords
                            and item[2][l] not in punctuations
                            and jtem[2][k] not in punctuations
                        ):
                            alignments.append([item[1][l], jtem[1][k]])
                            # unalignedWordIndicesInTheLongerName.remove(jtem[1][l])

    return alignments


def alignWords(source, target, sourceParseResult, targetParseResult):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

    # function returns the word alignments from source to target - each alignment returned is of the following form:
    # [
    #  [[source word character begin offset, source word character end offset], source word index, source word, source word lemma],
    #  [[target word character begin offset, target word character end offset], target word index, target word, target word lemma]
    # ]

    global punctuations

    sourceWordIndices = [i + 1 for i in range(len(source))]
    targetWordIndices = [i + 1 for i in range(len(target))]

    alignments = []
    sourceWordIndicesAlreadyAligned = []
    targetWordIndicesAlreadyAligned = []

    sourceWords = [item[2] for item in source]
    targetWords = [item[2] for item in target]

    sourceLemmas = [item[3] for item in source]
    targetLemmas = [item[3] for item in target]

    sourcePosTags = [item[4] for item in source]
    targetPosTags = [item[4] for item in target]

    # align the sentence ending punctuation first
    if (sourceWords[len(source) - 1] in [".", "!"] and targetWords[len(target) - 1] in [".", "!"]) or sourceWords[
        len(source) - 1
    ] == targetWords[len(target) - 1]:
        alignments.append([len(source), len(target)])
        sourceWordIndicesAlreadyAligned.append(len(source))
        targetWordIndicesAlreadyAligned.append(len(target))
    elif sourceWords[len(source) - 2] in [".", "!"] and targetWords[len(target) - 1] in [".", "!"]:
        alignments.append([len(source) - 1, len(target)])
        sourceWordIndicesAlreadyAligned.append(len(source) - 1)
        targetWordIndicesAlreadyAligned.append(len(target))
    elif sourceWords[len(source) - 1] in [".", "!"] and targetWords[len(target) - 2] in [".", "!"]:
        alignments.append([len(source), len(target) - 1])
        sourceWordIndicesAlreadyAligned.append(len(source))
        targetWordIndicesAlreadyAligned.append(len(target) - 1)
    elif sourceWords[len(source) - 2] in [".", "!"] and targetWords[len(target) - 2] in [".", "!"]:
        alignments.append([len(source) - 1, len(target) - 1])
        sourceWordIndicesAlreadyAligned.append(len(source) - 1)
        targetWordIndicesAlreadyAligned.append(len(target) - 1)

    # align all (>=2)-gram matches with at least one content word
    commonContiguousSublists = findAllCommonContiguousSublists(sourceWords, targetWords, True)
    for item in commonContiguousSublists:
        allStopWords = True
        for jtem in item:
            if jtem not in stopwords and jtem not in punctuations:
                allStopWords = False
                break
        if len(item[0]) >= 2 and not allStopWords:
            for j in range(len(item[0])):
                if (
                    item[0][j] + 1 not in sourceWordIndicesAlreadyAligned
                    and item[1][j] + 1 not in targetWordIndicesAlreadyAligned
                    and [item[0][j] + 1, item[1][j] + 1] not in alignments
                ):
                    alignments.append([item[0][j] + 1, item[1][j] + 1])
                    sourceWordIndicesAlreadyAligned.append(item[0][j] + 1)
                    targetWordIndicesAlreadyAligned.append(item[1][j] + 1)

    # align hyphenated word groups
    for i in sourceWordIndices:
        if i in sourceWordIndicesAlreadyAligned:
            continue
        if "-" in sourceWords[i - 1] and sourceWords[i - 1] != "-":
            tokens = sourceWords[i - 1].split("-")
            commonContiguousSublists = findAllCommonContiguousSublists(tokens, targetWords)
            for item in commonContiguousSublists:
                if len(item[0]) > 1:
                    for jtem in item[1]:
                        if [i, jtem + 1] not in alignments:
                            alignments.append([i, jtem + 1])
                            sourceWordIndicesAlreadyAligned.append(i)
                            targetWordIndicesAlreadyAligned.append(jtem + 1)

    for i in targetWordIndices:
        if i in targetWordIndicesAlreadyAligned:
            continue
        if "-" in target[i - 1][2] and target[i - 1][2] != "-":
            tokens = target[i - 1][2].split("-")
            commonContiguousSublists = findAllCommonContiguousSublists(sourceWords, tokens)
            for item in commonContiguousSublists:
                if len(item[0]) > 1:
                    for jtem in item[0]:
                        if [jtem + 1, i] not in alignments:
                            alignments.append([jtem + 1, i])
                            sourceWordIndicesAlreadyAligned.append(jtem + 1)
                            targetWordIndicesAlreadyAligned.append(i)

    # align named entities
    neAlignments = alignNamedEntities(source, target, sourceParseResult, targetParseResult, alignments)
    for item in neAlignments:
        if item not in alignments:
            alignments.append(item)
            if item[0] not in sourceWordIndicesAlreadyAligned:
                sourceWordIndicesAlreadyAligned.append(item[0])
            if item[1] not in targetWordIndicesAlreadyAligned:
                targetWordIndicesAlreadyAligned.append(item[1])

    # align words based on word and dependency match
    sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
    targetDParse = dependencyParseAndPutOffsets(targetParseResult)

    mainVerbAlignments = alignMainVerbs(source, target, sourceParseResult, targetParseResult, alignments)
    for item in mainVerbAlignments:
        if item not in alignments:
            alignments.append(item)
            if item[0] not in sourceWordIndicesAlreadyAligned:
                sourceWordIndicesAlreadyAligned.append(item[0])
            if item[1] not in targetWordIndicesAlreadyAligned:
                targetWordIndicesAlreadyAligned.append(item[1])

    nounAlignments = alignNouns(source, target, sourceParseResult, targetParseResult, alignments)
    for item in nounAlignments:
        if item not in alignments:
            alignments.append(item)
            if item[0] not in sourceWordIndicesAlreadyAligned:
                sourceWordIndicesAlreadyAligned.append(item[0])
            if item[1] not in targetWordIndicesAlreadyAligned:
                targetWordIndicesAlreadyAligned.append(item[1])

    adjectiveAlignments = alignAdjectives(source, target, sourceParseResult, targetParseResult, alignments)
    for item in adjectiveAlignments:
        if item not in alignments:
            alignments.append(item)
            if item[0] not in sourceWordIndicesAlreadyAligned:
                sourceWordIndicesAlreadyAligned.append(item[0])
            if item[1] not in targetWordIndicesAlreadyAligned:
                targetWordIndicesAlreadyAligned.append(item[1])

    adverbAlignments = alignAdverbs(source, target, sourceParseResult, targetParseResult, alignments)
    for item in adverbAlignments:
        if item not in alignments:
            alignments.append(item)
            if item[0] not in sourceWordIndicesAlreadyAligned:
                sourceWordIndicesAlreadyAligned.append(item[0])
            if item[1] not in targetWordIndicesAlreadyAligned:
                targetWordIndicesAlreadyAligned.append(item[1])

    # collect evidence from textual neighborhood for aligning content words
    wordSimilarities = {}
    textualNeighborhoodSimilarities = {}
    sourceWordIndicesBeingConsidered = []
    targetWordIndicesBeingConsidered = []

    for i in sourceWordIndices:
        if i in sourceWordIndicesAlreadyAligned or sourceLemmas[i - 1] in stopwords + punctuations + [
            "'s",
            "'d",
            "'ll",
        ]:
            continue

        for j in targetWordIndices:
            if j in targetWordIndicesAlreadyAligned or targetLemmas[j - 1] in stopwords + punctuations + [
                "'s",
                "'d",
                "'ll",
            ]:
                continue

            wordSimilarities[(i, j)] = max(
                wordRelatedness(
                    sourceWords[i - 1],
                    sourcePosTags[i - 1],
                    targetWords[j - 1],
                    targetPosTags[j - 1],
                ),
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                ),
            )
            sourceWordIndicesBeingConsidered.append(i)
            targetWordIndicesBeingConsidered.append(j)

            # textual neighborhood similarities
            sourceNeighborhood = findTextualNeighborhood(source, i, 3, 3)
            targetNeighborhood = findTextualNeighborhood(target, j, 3, 3)
            evidence = 0
            for k in range(len(sourceNeighborhood[0])):
                for l in range(len(targetNeighborhood[0])):
                    if (sourceNeighborhood[1][k] not in stopwords + punctuations) and (
                        (sourceNeighborhood[0][k], targetNeighborhood[0][l]) in alignments
                        or (
                            wordRelatedness(
                                sourceNeighborhood[1][k],
                                "none",
                                targetNeighborhood[1][l],
                                "none",
                            )
                            >= ppdbSim
                        )
                    ):
                        evidence += wordRelatedness(
                            sourceNeighborhood[1][k],
                            "none",
                            targetNeighborhood[1][l],
                            "none",
                        )
            textualNeighborhoodSimilarities[(i, j)] = evidence

    numOfUnalignedWordsInSource = len(sourceWordIndicesBeingConsidered)

    # now align: find the best alignment in each iteration of the following loop and include in alignments if good enough
    for item in range(numOfUnalignedWordsInSource):
        highestWeightedSim = 0
        bestWordSim = 0
        bestSourceIndex = -1
        bestTargetIndex = -1

        for i in sourceWordIndicesBeingConsidered:
            if i in sourceWordIndicesAlreadyAligned:
                continue

            for j in targetWordIndicesBeingConsidered:
                if j in targetWordIndicesAlreadyAligned:
                    continue

                if (i, j) not in wordSimilarities:
                    continue

                theta2 = 1 - theta1
                if (
                    theta1 * wordSimilarities[(i, j)] + theta2 * textualNeighborhoodSimilarities[(i, j)]
                    > highestWeightedSim
                ):
                    highestWeightedSim = (
                        theta1 * wordSimilarities[(i, j)] + theta2 * textualNeighborhoodSimilarities[(i, j)]
                    )
                    bestSourceIndex = i
                    bestTargetIndex = j
                    bestWordSim = wordSimilarities[(i, j)]
                    bestTextNeighborhoodSim = textualNeighborhoodSimilarities[(i, j)]

        if bestWordSim >= ppdbSim and [bestSourceIndex, bestTargetIndex] not in alignments:
            if sourceLemmas[bestSourceIndex - 1] not in stopwords:
                alignments.append([bestSourceIndex, bestTargetIndex])
                sourceWordIndicesAlreadyAligned.append(bestSourceIndex)
                targetWordIndicesAlreadyAligned.append(bestTargetIndex)

        if bestSourceIndex in sourceWordIndicesBeingConsidered:
            sourceWordIndicesBeingConsidered.remove(bestSourceIndex)
        if bestTargetIndex in targetWordIndicesBeingConsidered:
            targetWordIndicesBeingConsidered.remove(bestTargetIndex)

    # look if any remaining word is a part of a hyphenated word
    for i in sourceWordIndices:
        if i in sourceWordIndicesAlreadyAligned:
            continue
        if "-" in sourceWords[i - 1] and sourceWords[i - 1] != "-":
            tokens = sourceWords[i - 1].split("-")
            commonContiguousSublists = findAllCommonContiguousSublists(tokens, targetWords)
            for item in commonContiguousSublists:
                if len(item[0]) == 1 and target[item[1][0]][3] not in stopwords:
                    for jtem in item[1]:
                        if [i, jtem + 1] not in alignments and jtem + 1 not in targetWordIndicesAlreadyAligned:
                            alignments.append([i, jtem + 1])
                            sourceWordIndicesAlreadyAligned.append(i)
                            targetWordIndicesAlreadyAligned.append(jtem + 1)

    for i in targetWordIndices:
        if i in targetWordIndicesAlreadyAligned:
            continue
        if "-" in target[i - 1][2] and target[i - 1][2] != "-":
            tokens = target[i - 1][2].split("-")
            commonContiguousSublists = findAllCommonContiguousSublists(sourceWords, tokens)
            for item in commonContiguousSublists:
                if len(item[0]) == 1 and source[item[0][0]][3] not in stopwords:
                    for jtem in item[0]:
                        if [jtem + 1, i] not in alignments and i not in targetWordIndicesAlreadyAligned:
                            alignments.append([jtem + 1, i])
                            sourceWordIndicesAlreadyAligned.append(jtem + 1)
                            targetWordIndicesAlreadyAligned.append(i)

    # collect evidence from dependency neighborhood for aligning stopwords
    wordSimilarities = {}
    dependencyNeighborhoodSimilarities = {}
    sourceWordIndicesBeingConsidered = []
    targetWordIndicesBeingConsidered = []

    for i in sourceWordIndices:
        if sourceLemmas[i - 1] not in stopwords or i in sourceWordIndicesAlreadyAligned:
            continue

        for j in targetWordIndices:
            if targetLemmas[j - 1] not in stopwords or j in targetWordIndicesAlreadyAligned:
                continue

            if (sourceLemmas[i - 1] != targetLemmas[j - 1]) and (
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                )
                < ppdbSim
            ):
                continue

            wordSimilarities[(i, j)] = max(
                wordRelatedness(
                    sourceWords[i - 1],
                    sourcePosTags[i - 1],
                    targetWords[j - 1],
                    targetPosTags[j - 1],
                ),
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                ),
            )

            sourceWordIndicesBeingConsidered.append(i)
            targetWordIndicesBeingConsidered.append(j)

            sourceWordParents = findParents(sourceDParse, i, sourceWords[i - 1])
            sourceWordChildren = findChildren(sourceDParse, i, sourceWords[i - 1])
            targetWordParents = findParents(targetDParse, j, targetWords[j - 1])
            targetWordChildren = findChildren(targetDParse, j, targetWords[j - 1])

            evidence = 0

            for item in sourceWordParents:
                for jtem in targetWordParents:
                    if [item[0], jtem[0]] in alignments:
                        evidence += 1
            for item in sourceWordChildren:
                for jtem in targetWordChildren:
                    if [item[0], jtem[0]] in alignments:
                        evidence += 1

            dependencyNeighborhoodSimilarities[(i, j)] = evidence

    numOfUnalignedWordsInSource = len(sourceWordIndicesBeingConsidered)

    # now align: find the best alignment in each iteration of the following loop and include in alignments if good enough
    for item in range(numOfUnalignedWordsInSource):
        highestWeightedSim = 0
        bestWordSim = 0
        bestSourceIndex = -1
        bestTargetIndex = -1

        for i in sourceWordIndicesBeingConsidered:
            for j in targetWordIndicesBeingConsidered:
                if (i, j) not in wordSimilarities:
                    continue
                theta2 = 1 - theta1
                if (
                    theta1 * wordSimilarities[(i, j)] + theta2 * dependencyNeighborhoodSimilarities[(i, j)]
                    > highestWeightedSim
                ):
                    highestWeightedSim = (
                        theta1 * wordSimilarities[(i, j)] + theta2 * dependencyNeighborhoodSimilarities[(i, j)]
                    )
                    bestSourceIndex = i
                    bestTargetIndex = j
                    bestWordSim = wordSimilarities[(i, j)]
                    bestDependencyNeighborhoodSim = dependencyNeighborhoodSimilarities[(i, j)]

        if (
            bestWordSim >= ppdbSim
            and bestDependencyNeighborhoodSim > 0
            and [bestSourceIndex, bestTargetIndex] not in alignments
        ):
            alignments.append([bestSourceIndex, bestTargetIndex])
            sourceWordIndicesAlreadyAligned.append(bestSourceIndex)
            targetWordIndicesAlreadyAligned.append(bestTargetIndex)

        if bestSourceIndex in sourceWordIndicesBeingConsidered:
            sourceWordIndicesBeingConsidered.remove(bestSourceIndex)
        if bestTargetIndex in targetWordIndicesBeingConsidered:
            targetWordIndicesBeingConsidered.remove(bestTargetIndex)

    # collect evidence from textual neighborhood for aligning stopwords and punctuations
    wordSimilarities = {}
    textualNeighborhoodSimilarities = {}
    sourceWordIndicesBeingConsidered = []
    targetWordIndicesBeingConsidered = []

    for i in sourceWordIndices:
        if (
            sourceLemmas[i - 1] not in stopwords + punctuations + ["'s", "'d", "'ll"]
        ) or i in sourceWordIndicesAlreadyAligned:
            continue

        for j in targetWordIndices:
            if (
                targetLemmas[j - 1] not in stopwords + punctuations + ["'s", "'d", "'ll"]
            ) or j in targetWordIndicesAlreadyAligned:
                continue

            if (
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                )
                < ppdbSim
            ):
                continue

            wordSimilarities[(i, j)] = max(
                wordRelatedness(
                    sourceWords[i - 1],
                    sourcePosTags[i - 1],
                    targetWords[j - 1],
                    targetPosTags[j - 1],
                ),
                wordRelatedness(
                    sourceLemmas[i - 1],
                    sourcePosTags[i - 1],
                    targetLemmas[j - 1],
                    targetPosTags[j - 1],
                ),
            )

            sourceWordIndicesBeingConsidered.append(i)
            targetWordIndicesBeingConsidered.append(j)

            # textual neighborhood evidence
            evidence = 0

            if [i - 1, j - 1] in alignments:
                evidence += 1

            if [i + 1, j + 1] in alignments:
                evidence += 1

            try:
                textualNeighborhoodSimilarities[(i, j)] = evidence
            except ZeroDivisionError:
                textualNeighborhoodSimilarities[(i, j)] = 0

    numOfUnalignedWordsInSource = len(sourceWordIndicesBeingConsidered)

    # now align: find the best alignment in each iteration of the following loop and include in alignments if good enough
    for item in range(numOfUnalignedWordsInSource):
        highestWeightedSim = 0
        bestWordSim = 0
        bestSourceIndex = -1
        bestTargetIndex = -1

        for i in sourceWordIndicesBeingConsidered:
            if i in sourceWordIndicesAlreadyAligned:
                continue

            for j in targetWordIndicesBeingConsidered:
                if j in targetWordIndicesAlreadyAligned:
                    continue

                if (i, j) not in wordSimilarities:
                    continue

                theta2 = 1 - theta1
                if (
                    theta1 * wordSimilarities[(i, j)] + theta2 * textualNeighborhoodSimilarities[(i, j)]
                    > highestWeightedSim
                ):
                    highestWeightedSim = (
                        theta1 * wordSimilarities[(i, j)] + theta2 * textualNeighborhoodSimilarities[(i, j)]
                    )
                    bestSourceIndex = i
                    bestTargetIndex = j
                    bestWordSim = wordSimilarities[(i, j)]
                    bestTextNeighborhoodSim = textualNeighborhoodSimilarities[(i, j)]

        if (
            bestWordSim >= ppdbSim
            and bestTextNeighborhoodSim > 0
            and [bestSourceIndex, bestTargetIndex] not in alignments
        ):
            alignments.append([bestSourceIndex, bestTargetIndex])
            sourceWordIndicesAlreadyAligned.append(bestSourceIndex)
            targetWordIndicesAlreadyAligned.append(bestTargetIndex)

        if bestSourceIndex in sourceWordIndicesBeingConsidered:
            sourceWordIndicesBeingConsidered.remove(bestSourceIndex)
        if bestTargetIndex in targetWordIndicesBeingConsidered:
            targetWordIndicesBeingConsidered.remove(bestTargetIndex)

    alignments = [item for item in alignments if item[0] != 0 and item[1] != 0]

    return alignments


class MonolingualWordAligner:
    def get_word_aligns(self, sentence1ParseResult, sentence2ParseResult):
        sentence1Lemmatized = lemmatize(sentence1ParseResult)
        sentence2Lemmatized = lemmatize(sentence2ParseResult)

        sentence1PosTagged = posTag(sentence1ParseResult)
        sentence2PosTagged = posTag(sentence2ParseResult)

        sentence1LemmasAndPosTags = []
        for i in range(len(sentence1Lemmatized)):
            sentence1LemmasAndPosTags.append([])
        for i in range(len(sentence1Lemmatized)):
            for item in sentence1Lemmatized[i]:
                sentence1LemmasAndPosTags[i].append(item)
            sentence1LemmasAndPosTags[i].append(sentence1PosTagged[i][3])

        sentence2LemmasAndPosTags = []
        for i in range(len(sentence2Lemmatized)):
            sentence2LemmasAndPosTags.append([])
        for i in range(len(sentence2Lemmatized)):
            for item in sentence2Lemmatized[i]:
                sentence2LemmasAndPosTags[i].append(item)
            sentence2LemmasAndPosTags[i].append(sentence2PosTagged[i][3])

        myWordAlignments = alignWords(
            sentence1LemmasAndPosTags,
            sentence2LemmasAndPosTags,
            sentence1ParseResult,
            sentence2ParseResult,
        )
        myWordAlignmentTokens = [
            [
                str(sentence1Lemmatized[item[0] - 1][2]),
                str(sentence2Lemmatized[item[1] - 1][2]),
            ]
            for item in myWordAlignments
        ]

        return [myWordAlignments, myWordAlignmentTokens]
