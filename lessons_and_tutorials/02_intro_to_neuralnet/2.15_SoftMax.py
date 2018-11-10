import numpy as np

def softmax(L):
    # Write a function that takes as input a list of numbers,
    # and returns the list of values given by the softmax function.
    probs = [];
    norm_fac = np.sum(np.exp(L));
    for l_k in L:
        p_k = np.exp(l_k) / norm_fac;
        probs.append(p_k);
    return probs;

# def softmax(L):
#     expL = np.exp(L);
#     sumExpL = np.sum(expL);
#     res = [];
#     for k in expL:
#         p_k = k * 1.0 / sumExpL;
#         res.append(p_k);
#     return res;
