#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import random
import gmpy2

POWMOD_GMP_SIZE = pow(2, 64)

# 模幂运算：gmpy2.powmod(a, n, p)
# 对于给定的整数p,n,a,计算aⁿ mod p
def powmod(a, b, c):
    """
    return int: (a ** b) % c
    """

    if a == 1:
        return 1

    if max(a, b, c) < POWMOD_GMP_SIZE:
        return pow(a, b, c)

    else:
        return int(gmpy2.powmod(a, b, c))

# 模逆运算
def invert(a, b):
    """return int: x, where a * x == 1 mod b
    """
    # 模逆运算：gmpy2.invert（a，c）  # 对a，求b，使得a*b（mod c）=1
    x = int(gmpy2.invert(a, b))

    if x == 0:
        raise ZeroDivisionError('invert(a, b) no inverse exists')

    return x

# 获取特定位数
def getprimeover(n):
    """return a random n-bit prime number
    """
    # 可以为变量r赋予一个高精度的大整数（长度可达50位）
    r = gmpy2.mpz(random.SystemRandom().getrandbits(n))
    # r获取特定位数
    r = gmpy2.bit_set(r, n - 1)
    # 获取r的下一个素数
    return int(gmpy2.next_prime(r))


def isqrt(n):
    """ return the integer square root of N """

    return int(gmpy2.isqrt(n))


