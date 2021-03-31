from functools import lru_cache


def get_lcs(seq1, seq2):
    '''Returns the longest common subsequence using memoization (only in local scope)'''

    @lru_cache(maxsize=None)
    def recursive_lcs(seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return []
        if seq1[-1] == seq2[-1]:
            return recursive_lcs(seq1[:-1], seq2[:-1]) + [seq1[-1]]
        else:
            return max(recursive_lcs(seq1[:-1], seq2), recursive_lcs(seq1, seq2[:-1]), key=lambda seq: len(seq))

    try:
        return recursive_lcs(tuple(seq1), tuple(seq2))
    except RecursionError as e:
        print(e)
        # TODO: Handle this case
        return []


def get_lcs_alignment(seq1, seq2):
    '''Same as get_lcs() but returning the alignments of indexes instead of words'''

    @lru_cache()
    def recursive_lcs(seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return []
        if seq1[-1] == seq2[-1]:
            return recursive_lcs(seq1[:-1], seq2[:-1]) + [(len(seq1) - 1, len(seq2) - 1)]
        else:
            return max(recursive_lcs(seq1[:-1], seq2), recursive_lcs(seq1, seq2[:-1]), key=lambda seq: len(seq))

    return recursive_lcs(tuple(seq1), tuple(seq2))
