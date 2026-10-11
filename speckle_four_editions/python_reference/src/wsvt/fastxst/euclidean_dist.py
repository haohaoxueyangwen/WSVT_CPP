#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 08 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: euclidean_dist.py
@Software	: WaveletSBI
@Desc		: euclidean distance calculation for cross-correlation
'''

import numba as nb
from numba import prange
import numpy as np


@nb.jit('float32[:,:](float32[:], float32[:,:,:])', nopython=True)
def dist_numba(v1, arry2):
    """
    dist_numba for cross-correlation calculation using numba

    Args:
        v1 (ndarray): 1D data
        arry2 (ndarray): 2D data

    Returns:
        cross correlation efficients
    """
    c = np.empty(arry2.shape[0:2], dtype=np.float32)
    s = 0

    for ii in prange(arry2.shape[0]):
        for jj in range(arry2.shape[1]):
            s = 0
            for kk in range(arry2.shape[2]):
                # s += (v1[kk]*arry2[ii, jj, kk])
                s += (v1[kk] - arry2[ii, jj, kk])**2

            c[ii, jj] = -s

    return c.astype(np.float32)
