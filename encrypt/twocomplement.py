'''
Descripttion: 
Author: wjz
Date: 2023-02-14 20:44:50
LastEditors: wjz
LastEditTime: 2023-05-07 09:58:43
'''
import numpy as np
from joblib import Parallel, delayed
import warnings
import multiprocessing
from numba import njit, prange
N_JOBS = multiprocessing.cpu_count()
class TwoComplement(object):

    def __init__(self, int_bits):
        super(TwoComplement, self).__init__()

    @classmethod
    def true_to_two(cls, value, int_bits):
        mod = 2 ** int_bits
        value = value % mod
        return value

    @classmethod
    def two_to_true(cls, value, int_bits):
        border = 2 ** (int_bits - 1)
        offset = - 2 ** int_bits
        ret = np.where(value < border, value, value + offset)
        return ret
    # @classmethod
    # def two_to_true(cls,two_comp, bit_width=8):
    #     def two_comp_lit_to_ori(lit, _bit_width):  # convert 2's complement coding of neg value to its original form
    #         return - 1 * (2 ** (_bit_width - 1) - lit)

    #     if two_comp.any() < 0:
    #         raise Exception("Error: not expecting negtive value")
    #     # two_com_string = bin(two_comp)[2:].zfill(bit_width+pad_zero)
    #     # 取得对应标志位
    #     sign = two_comp >> (bit_width - 1)
    #     literal = two_comp & (2 ** (bit_width - 1) - 1)
        
    #     if sign == 0:  # positive value 0000
    #         return literal
    #     elif sign == 4:  # positive value, 0100
    #         return literal
    #     elif sign == 1:  # positive overflow, 0001 正溢
    #         return pow(2, bit_width - 1) - 1
    #     elif sign == 3:  # negtive value, 0011
    #         return two_comp_lit_to_ori(literal, bit_width)
    #     elif sign == 7:  # negtive value, 0111
    #         return two_comp_lit_to_ori(literal, bit_width)
    #     elif sign == 6:  # negtive overflow, 0110 下溢
    #         print('neg overflow: ' + str(two_comp))
    #         return - (pow(2, bit_width - 1) - 1)
    #     else:  # unrecognized overflow
    #         print('unrecognized overflow: ' + str(two_comp))
    #         warnings.warn('Overflow detected, consider using longer r_max')
    #         return - (pow(2, bit_width - 1) - 1)
        
        
    # def true_to_two(input, bit_width):
    #     result = np.zeros(len(input), dtype=np.int32)
    #     for i in prange(len(input)):
    #         if input[i] >= 0:
    #             result[i] = input[i]
    #         else:
    #             result[i] = 2 ** (bit_width + 1) + input[i]
    #     return result



    # def two_to_true(two_comp, bit_width=8):
        
    #     def two_comp_lit_to_ori(lit, _bit_width):  # convert 2's complement coding of neg value to its original form
    #         return - 1 * (2 ** (_bit_width - 1) - lit)
        
    #     literal = two_comp & (2 ** (bit_width - 1) - 1)
    #     return two_comp_lit_to_ori(literal, bit_width)
        
        


