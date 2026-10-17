"""Exact weighted AUC draws without materializing sampled rows or zero products."""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def draws(codes, positive_slots, positive_groups, starts, block, n_groups):
    n = codes.size
    result = np.empty(starts.shape[0], dtype=np.float64)
    delta = np.empty(n+1, dtype=np.int32)
    total = np.empty(n_groups, dtype=np.int32)
    positive = np.empty(positive_groups.size, dtype=np.int32)
    for draw in range(starts.shape[0]):
        delta[:] = 0
        total[:] = 0
        positive[:] = 0
        for b in range(starts.shape[1]):
            length = min(block, n-b*block)  # the original draw truncates its final block
            first = starts[draw, b]
            end = first+length
            delta[first] += 1
            if end <= n:
                delta[end] -= 1
            else:
                delta[n] -= 1
                delta[0] += 1
                delta[end-n] -= 1
        weight, n_positive = 0, 0
        for row in range(n):
            weight += delta[row]
            total[codes[row]] += weight
            slot = positive_slots[row]
            if slot >= 0:
                positive[slot] += weight
                n_positive += weight
        if n_positive == 0 or n_positive == n:
            result[draw] = np.nan
            continue
        below, positive_below, cursor = 0, 0, 0
        numerator = 0.
        for group in range(n_groups):
            count = total[group]
            if cursor < positive_groups.size and positive_groups[cursor] == group:
                p = positive[cursor]
                numerator += float(p)*(float(below-positive_below)+.5*float(count-p))
                positive_below += p
                cursor += 1
            below += count
        result[draw] = numerator/(float(n_positive)*float(n-n_positive))
    return result
