import random
from functools import reduce
import numpy as np

# Efficient Methods for Practical Fully-Homomorphic Symmetric-key 
# Encryption, Randomization, and Verification
rd = np.random.RandomState(888) 
class SymmetricKey(object):
    """
    Symmetric encryption key
    """

    def __init__(self,m,key_size):
        self.m = m
        self.key_size =key_size
        pass
    def generatekey(self):
        p,q = generate_prime(self.m,self.key_size)
        print("p,q: ",p,q)
        self.f = [p[x]*q[x] for x in range(self.m)]
        print("f: ",self.f)
        self.N = reduce(lambda x,y:x*y,self.f)
        print("N: ",self.N)
        
        #　设置随机种子，保证每次生成的随机数一样，可以不设置（去除下面一行代码，将所有的 rd 替换成 np.random 即可）
        
        self.K = np.mod(rd.randint(0, 2**self.key_size,(2,2)),self.N)
        print("K: ",self.K)
        self.K_inv = np.linalg.pinv(self.K)
        print("K_inv: ",self.K_inv)
        return self.f,self.N,self.K,self.K_inv
        
        
        # f = p*q
        # N = f
        # k = random mod N
        # k_inv 
         
        
        pass
    def encrypt(self, x,ktuple=None):
        """
        Encryption method
        :param plaintext:
        :return:
        """
        r = rd.randint(0,2**self.key_size)
        # X = np.array([[x,0],[0,r]])
        X = np.matrix([[x,0],[0,r]])
        print("X: ",X)
        # enc_x = np.matmul(np.matmul(self.K,X),self.K_inv)
        enc_x = self.K * X * self.K_inv
        print("enc_x: ",enc_x)
        return enc_x

    def decrypt(self, enc_x):
        """
        Decryption method
        :param ciphertext:
        :return:
        """
        # X = np.matmul(np.matmul(self.K_inv,enc_x),self.K)
        X = self.K_inv * enc_x * self.K 
        # print(X)
        return X.A[0][0]
    
    
# 检测大整数是否是素数,如果是素数,就返回True,否则返回False
# rabin算法
def rabin_miller(num):
    s = num - 1
    t = 0
    while s % 2 == 0:
        s = s // 2
        t += 1

    for trials in range(5):
        a = random.randrange(2, num - 1)
        v = pow(a, s, num)
        if v != 1:
            i = 0
            while v != (num - 1):
                if i == t - 1:
                    return False
                else:
                    i = i + 1
                    v = (v ** 2) % num
    return True


def is_prime(num):
    # 排除0,1和负数
    if num < 2:
        return False

    # 创建小素数的列表,可以大幅加快速度
    # 如果是小素数,那么直接返回true
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    if num in small_primes:
        return True

    # 如果大数是这些小素数的倍数,那么就是合数,返回false
    for prime in small_primes:
        if num % prime == 0:
            return False

    # 如果这样没有分辨出来,就一定是大整数,那么就调用rabin算法
    return rabin_miller(num)


# 得到大整数,默认位数为1024
def get_prime(key_size=1024):
    while True:
        num = random.randrange(2**(key_size-1), 2**key_size)
        if is_prime(num):
            return num    


def generate_prime(m,key_size):
        p,q = [get_prime(key_size//2) for x in range(m)],[get_prime(key_size//2) for x in range(m)]
        return p,q
        pass
       
if __name__=="__main__":
    key = SymmetricKey(m=2,key_size=20)
    print("ing")
    # p = [x for x in range(2)]
    # print(p)
    # print(generate_prime(2,1024))
    ktuple = key.generatekey()
    x1,x2 = 10.911111,3
    enc_x1 = key.encrypt(x1)
    enc_x2 = key.encrypt(x2)
    # 乘法同态
    enc_mult = enc_x1*enc_x2
    x_mult = key.decrypt(enc_mult)
    print("乘法同态: ",x_mult)
    # 加法同态
    enc_add = enc_x1+enc_x2
    x_add = key.decrypt(enc_add)
    print("加法同态: ",x_add)
    # 除法同态
    # enc_div = enc_x1/enc_x2
    # x_div = key.decrypt(enc_div)
    # print("除法同态: ",x_div)
    
    # print(type(x_add))
    # key.decrypt(enc_x2)
    
    