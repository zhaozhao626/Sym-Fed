from functools import reduce
import hashlib
import os
import time
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
# from encrypt.aes_prp import PsuedoRandomPermutation
import numpy as np
from multiprocessing import Pool, cpu_count

# N_JOBS = cpu_count()
N_JOBS = 1
# LOGGER = log_utils.getLogger()
BITS_PER_BYTES = 8

# class PsuedoRandomPermutation:
#     pass

def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


def _static_prepare_encrypt_single(begin, end, hash_algorithm,
                            int_bits, index_prefix_for_add):
    len = end - begin
    merge_size = 256 // int_bits
    merge_num = (len - 1) // merge_size + 1
    add_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)

        # i + begin can guarantee to be unique globally
        index_for_add = index_prefix_for_add + (i + begin).to_bytes(8, 'big')

        add_term_bytes = hashlib.new(hash_algorithm, index_for_add).digest()
        add_term_s = int.from_bytes(add_term_bytes, 'big')

        mask = (1 << int_bits) - 1
        for j in range(b, e):
            add_term = add_term_s & mask
            add_terms.append(add_term)
            add_term_s >>= int_bits

    return add_terms

    """
    该函数是一个内部函数，用于为加密操作准备参数。具体来说，该函数接受以下参数：

    begin和end：表示要加密的数据的起始索引和结束索引。
    prp_seed：表示用于伪随机置换的种子。
    int_bits：表示用于加密的整数的位数。
    index_prefix_for_add和index_prefix_for_minus：表示用于生成加法项和减法项的哈希前缀。
    函数实现的主要功能是将要加密的数据分成多个块，对于每个块生成一个加法项和一个减法项。具体来说，函数实现的步骤如下：

    计算要加密的数据的长度len，以及每个块的大小merge_size和块数merge_num。其中，merge_size是根据int_bits计算得到的，merge_num是根据len和merge_size计算得到的。
    对于每个块i，计算起始索引b和结束索引e。
    生成用于生成加法项和减法项的索引index_for_add和index_for_minus。其中，索引的前缀是index_prefix_for_add和index_prefix_for_minus，后缀是块的全局索引i + begin。
    使用哈希算法计算索引的哈希值，得到一个bytes对象。
    将bytes对象转换成整数add_term_s和minus_term_s，作为加法项和减法项的种子。
    使用掩码mask将种子分成多个整数，每个整数的长度是int_bits。
    将每个整数作为加法项和减法项，并添加到add_terms和minus_terms列表中。
    最后，将add_terms和minus_terms作为结果返回。
    该函数的返回值是一个包含加法项和减法项的元组，每个元素是一个整数列表，表示要用于加密的参数。
    """
def _static_prepare_encrypt(begin, end, hash_algorithm, int_bits,
                            index_prefix_for_add, index_prefix_for_minus):

    len = end - begin
    if int_bits > 256:
        merge_size = 1
    else:
        # 每个块的大小merge_size和块数merge_num
        merge_size = 256 // int_bits
    merge_num = (len - 1) // merge_size + 1
    add_terms = []
    minus_terms = []
    # print("len:",len)
    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)
        # print(b)
        # print(e)
        # i + begin can guarantee to be unique globally
        index_for_add = index_prefix_for_add + (i + begin).to_bytes(8, 'big')
        index_for_minus = index_prefix_for_minus + (i + begin).to_bytes(8, 'big')

        add_term_bytes = hashlib.new(hash_algorithm, index_for_add).digest()
        add_term_s = int.from_bytes(add_term_bytes, 'big')
        minus_term_bytes = hashlib.new(hash_algorithm, index_for_minus).digest()
        minus_term_s = int.from_bytes(minus_term_bytes, 'big')
        
        mask = (1 << int_bits) - 1
        for j in range(b, e):
            add_term = add_term_s & mask
            add_terms.append(add_term)
            add_term_s >>= int_bits

            minus_term = minus_term_s & mask
            minus_terms.append(minus_term)
            minus_term_s >>= int_bits

    return add_terms, minus_terms

# def _static_prepare_encrypt(begin, end, hash_algorithm, int_bits,
#                             index_prefix_for_add, index_prefix_for_minus):
#     len = end - begin
#     merge_size = 256 // int_bits
#     # print(merge_size)
#     merge_num = (len - 1) // merge_size + 1
#     add_terms = []
#     minus_terms = []
#     # print(merge_num)
#     for i in range(merge_num):
#         b = i * merge_size
#         e = min((i + 1) * merge_size, len)
#         # print(b,e)
#         index_for_add = index_prefix_for_add + np.array([i + begin]).astype(np.uint64).tobytes()
#         index_for_minus = index_prefix_for_minus + np.array([i + begin]).astype(np.uint64).tobytes()

#         add_term_bytes = hashlib.new(hash_algorithm, index_for_add).digest()
#         # add_term_s = np.frombuffer(add_term_bytes, dtype=np.uint256)
#         add_term_s = int.from_bytes(add_term_bytes, 'big')    
#         # print(add_term_s)
#         minus_term_bytes = hashlib.new(hash_algorithm, index_for_minus).digest()
#         # minus_term_s = np.frombuffer(minus_term_bytes, dtype=np.uint64)[0]
#         minus_term_s = int.from_bytes(minus_term_bytes, 'big')

#         mask = (1 << int_bits) - 1
#         add_terms.append(np.right_shift(np.bitwise_and(add_term_s, mask << np.arange(0, (e-b)*int_bits, int_bits, dtype=np.uint64)),
#                                         np.arange(0, (e-b)*int_bits, int_bits, dtype=np.uint64)))
#         minus_terms.append(np.right_shift(np.bitwise_and(minus_term_s, mask << np.arange(0, (e-b)*int_bits, int_bits, dtype=np.uint64)),
#                                           np.arange(0, (e-b)*int_bits, int_bits, dtype=np.uint64)))
#         # print(np.right_shift(np.bitwise_and(add_term_s, mask << np.arange(0, (e-b)*int_bits, int_bits, dtype=np.uint64)),
#         #                                 np.arange(0, (e-b)*int_bits, int_bits, dtype=np.uint64)))
#         # print(add_terms)
#     return np.concatenate(add_terms).tolist(), np.concatenate(minus_terms).tolist()


def _static_prepare_decrypt_single(begin, end, hash_algorithm, int_bits, index_prefix_for_minus_list):


    len = end - begin
    merge_size = 256 // int_bits
    merge_num = (len - 1) // merge_size + 1
    minus_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)

        # i + begin can guarantee to be unique globally
        index_for_minus_list = [k + (i + begin).to_bytes(8, 'big') for k in index_prefix_for_minus_list ]
        minus_term_s_list = [int.from_bytes(hashlib.new(hash_algorithm, k).digest(), 'big') for k in index_for_minus_list]

        mask = (1 << int_bits) - 1
        for j in range(b, e):
            minus_term = 0

            for k_idx, k in enumerate(minus_term_s_list):
                minus_term += k
                minus_term_s_list[k_idx] >>= int_bits

            minus_terms.append(minus_term & mask)

    return minus_terms


def _static_prepare_decrypt(begin, end, hash_algorithm, int_bits,
                            index_prefix_for_add_list, index_prefix_for_minus_list):


    len = end - begin
    if int_bits>256:
        merge_size = 1
    else:
        merge_size = 256 // int_bits
    merge_num = (len - 1) // merge_size + 1
    add_terms = []
    minus_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)

        # i + begin can guarantee to be unique globally
        index_for_add_list = [k + (i + begin).to_bytes(8, 'big') for k in index_prefix_for_add_list ]
        index_for_minus_list = [k + (i + begin).to_bytes(8, 'big') for k in index_prefix_for_minus_list ]

        add_term_s_list = [int.from_bytes(hashlib.new(hash_algorithm, k).digest(), 'big') for k in index_for_add_list]
        minus_term_s_list = [int.from_bytes(hashlib.new(hash_algorithm, k).digest(), 'big') for k in index_for_minus_list]

        mask = (1 << int_bits) - 1
        for j in range(b, e):
            add_term = 0
            minus_term = 0
            for k_idx, k in enumerate(add_term_s_list):
                add_term += k
                add_term_s_list[k_idx] >>= int_bits

            for k_idx, k in enumerate(minus_term_s_list):
                minus_term += k
                minus_term_s_list[k_idx] >>= int_bits

            add_terms.append(add_term & mask)
            minus_terms.append(minus_term & mask)

    return add_terms, minus_terms




class FlasheCipher_Hash(object):

    def __init__(self, int_bits, mask="double"):
        super(FlasheCipher_Hash, self).__init__()
        self.hash_algorithm = None
        self.masking_scheme = mask
        self.masks = None
        self.total = None
        # # 伪随机函数生成
        # self.prp = PsuedoRandomPermutation()
        # self.prp_seed = None
        # self.prp_seed_len = 256

        self.idx = None
        self.index_prefix_for_add = None
        self.index_prefix_for_minus = None

        self.iter_index = -1
        self.iter_index_bytes = None

        self.int_bits = int_bits

        # for preparing encryption / decryption
        self.num_clients = None
        self.next_iter_encrypt_prepared = {}
        self.next_iter_decrypt_prepared = {}
        self.next_iter_decrypt_prepared_idx = {}
        self.num_params = None

        self.encrypt_base = 0
        self.decrypt_base = 0
    # 设置客户端数
    def set_num_clients(self, num_clients):
        self.num_clients = num_clients
    def set_hash_algorithm(self,hash_algorithm):
        self.hash_algorithm = hash_algorithm
        
    # # 生成伪随机函数种子，用于生成密钥
    # def generate_prp_seed(self, assigned_seed=None):
    #     if assigned_seed is None:
    #         seed = os.urandom(
    #             self.prp_seed_len // BITS_PER_BYTES)
    #     else:
    #         if isinstance(assigned_seed, int):
    #             seed = int(assigned_seed & int(
    #                 2 ** self.prp_seed_len - 1)).to_bytes(
    #                 self.prp_seed_len, 'big')
    #         else:
    #             seed = int(int.from_bytes(assigned_seed, 'big') & int(
    #                 2 ** self.prp_seed_len - 1)).to_bytes(
    #                 self.prp_seed_len, 'big')

    #     self.prp_seed = seed
    #     self.prp.generate_key(assigned_key=seed)

    # def get_prp_seed(self):
    #     return self.prp_seed
    # 设置
    def set_iter_index(self, iter_index):
        self.encrypt_base = 0
        self.decrypt_base = 0
        self.iter_index = iter_index
        self.iter_index_bytes = iter_index.to_bytes(4, 'big')
    # 单个掩码
    def set_idx_list_single(self, raw_idx_list=None, mode="encrypt"):
        if mode == "encrypt":
            self.index_prefix_for_add = self.iter_index_bytes \
                                        + self.idx.to_bytes(4, 'big')
        else:
            if self.masks is None:
                self.index_prefix_for_minus = [
                    self.iter_index_bytes + idx.to_bytes(4, 'big') for idx in raw_idx_list
                ]
            else:
                temp_minus = None
                for client_idx, mask in enumerate(self.masks):
                    pool_inputs = []
                    pool = Pool(N_JOBS)
                    index_prefix_for_minus = self.iter_index_bytes + int(client_idx).to_bytes(4, 'big')
                    for begin, end in chunks_idx(range(len(mask)), N_JOBS):
                        pool_inputs.append([begin, end, self.hash_algorithm, self.int_bits, index_prefix_for_minus])

                    # this is correct, i.e., using encrypt not decrypt
                    pool_outputs = pool.starmap(_static_prepare_encrypt_single, pool_inputs)
                    pool.close()
                    pool.join()

                    t_minus = []
                    for ret in pool_outputs:  # ret is list
                        t_minus += ret
                    a = np.zeros(self.total, dtype=object)
                    a[mask] = t_minus

                    if temp_minus is None:
                        temp_minus = a
                    else:
                        temp_minus += a
                    temp_minus &= ((1 << self.int_bits) - 1)
                    # LOGGER.info(f"{client_idx} {mask[:5]}")

                # LOGGER.info(f"dec {temp_minus[:5]}")
                self.next_iter_decrypt_prepared["minus"] = temp_minus
    # 设置标识符
    def set_idx_list(self, raw_idx_list=None, mode="encrypt"):
        if self.masking_scheme == "single":
            return self.set_idx_list_single(raw_idx_list, mode)

        if mode == "encrypt":
            self.index_prefix_for_add = self.iter_index_bytes \
                                        + self.idx.to_bytes(4, 'big')
            self.index_prefix_for_minus = self.iter_index_bytes \
                                          + (self.idx + 1).to_bytes(4, 'big')
        else:
            # if self.masks is None:
                raw_idx_list.sort()
                temp_add, temp_minus = [], []
                for idx in raw_idx_list:
                    if len(temp_add) == 0:
                        temp_add.append(idx + 1)
                        temp_minus.append(idx)
                    else:
                        if idx == temp_add[-1]:
                            temp_add = temp_add[:-1] + [idx + 1]
                        else:
                            temp_add.append(idx + 1)
                            temp_minus.append(idx)

                self.index_prefix_for_add = []
                self.index_prefix_for_minus = []

                for idx in temp_add:
                    if 'add' in self.next_iter_decrypt_prepared_idx and idx in self.next_iter_decrypt_prepared_idx['add']:
                        pass
                    else:
                        self.index_prefix_for_add.append(
                            self.iter_index_bytes + idx.to_bytes(4, 'big')
                        )

                for idx in temp_minus:
                    if 'minus' in self.next_iter_decrypt_prepared_idx and idx in self.next_iter_decrypt_prepared_idx['minus']:
                        pass
                    else:
                        self.index_prefix_for_minus.append(
                            self.iter_index_bytes + idx.to_bytes(4, 'big')
                        )
    # 获取标识符
    def get_idx_list(self):
        return [self.idx]
    # 多线程加密-单掩码
    def _multiprocessing_encrypt_single(self, value):
        pool_inputs = []
        pool = Pool(N_JOBS)

        # LOGGER.info(f'encrypt {self.index_prefix_for_add} {self.index_prefix_for_minus}')
        for begin, end in chunks_idx(range(len(value)), N_JOBS):
            pool_inputs.append([begin, end, self.hash_algorithm, self.int_bits,
                                self.index_prefix_for_add])
        pool_outputs = pool.starmap(_static_prepare_encrypt_single, pool_inputs)
        pool.close()
        pool.join()

        temp_add = []
        for ret in pool_outputs:  # ret is list
            temp_add += ret

        self.next_iter_encrypt_prepared['add'] = np.array(temp_add, dtype=object)
        # LOGGER.info(f"enc {self.idx} {temp_add[:5]}")

        ret = value + self.next_iter_encrypt_prepared['add']
        ret &= ((1 << self.int_bits) - 1)
        if 'add' in self.next_iter_encrypt_prepared:
            del self.next_iter_encrypt_prepared['add']
        return ret
    # 多线程加密-双掩码 生成随机数--core function
    def _multiprocessing_encrypt(self, value):
        if 'add' not in self.next_iter_encrypt_prepared:
            pool_inputs = []
            # pool = Pool(N_JOBS)

            # LOGGER.info(f'encrypt {self.index_prefix_for_add} {self.index_prefix_for_minus}')
            for begin, end in chunks_idx(range(len(value)), N_JOBS):
                pool_inputs.append([begin, end, self.hash_algorithm, self.int_bits,
                                    self.index_prefix_for_add, self.index_prefix_for_minus])
            # print(pool_inputs)
            # pool_outputs = pool.starmap(_static_prepare_encrypt, pool_inputs)
        
            pool_outputs = Parallel(n_jobs=N_JOBS)(
            delayed(_static_prepare_encrypt)(*pool_input) for pool_input in pool_inputs)
            # pool.close()
            # pool.join()

            temp_add = []
            temp_minus = []
            for ret in pool_outputs:  # ret is list
                temp_add += ret[0]
                temp_minus += ret[1]
            
            # print("temp_add:",temp_add )
            # temp_add = np.concatenate([ret[0] for ret in pool_outputs]).astype(np.int64)
            # print("temp_add:",temp_add )
            # temp_minus = np.concatenate([ret[1] for ret in pool_outputs]).astype(np.int64)
            self.next_iter_encrypt_prepared['add'] = np.array(temp_add, dtype=object)
            self.next_iter_encrypt_prepared['minus'] = np.array(temp_minus, dtype=object)

        # LOGGER.info(f"encrypt add {len(self.next_iter_encrypt_prepared['add'])} {self.next_iter_encrypt_prepared['add'][0]}")
        # LOGGER.info(f"encrypt minus {len(self.next_iter_encrypt_prepared['minus'])} {self.next_iter_encrypt_prepared['minus'][0]}")
        
        # print("value:",value)
        # print("add:",self.next_iter_encrypt_prepared['add'])
        # print("minus:",self.next_iter_encrypt_prepared['minus'])
        # first_random = np.array(first_random, dtype=object)
        ret = value + self.next_iter_encrypt_prepared['add'] - self.next_iter_encrypt_prepared['minus']
        # test = value + self.next_iter_encrypt_prepared['add'] - self.next_iter_encrypt_prepared['minus']
        # print(ret)
        ret &= ((1 << self.int_bits) - 1)

        if 'add' in self.next_iter_encrypt_prepared:
            del self.next_iter_encrypt_prepared['add']
        if 'minus' in self.next_iter_encrypt_prepared:
            del self.next_iter_encrypt_prepared['minus']

        return ret
    # 加密操作
    def encrypt(self, plaintext):
        if self.hash_algorithm is not None:
            # 首先设置标识符
            if self.masking_scheme == "double":
                self.set_idx_list(mode="encrypt")  # for both enc and agg
            else:
                self.set_idx_list_single(mode="encrypt")
            if not isinstance(plaintext, np.ndarray):
                return None
            else:
                if self.masking_scheme == "double":
                    return self._multiprocessing_encrypt(plaintext)
                else:
                    return self._multiprocessing_encrypt_single(plaintext)
        else:
            return None
    # 解密操作，单掩码
    def _multiprocessing_decrypt_single(self, value):
        if self.masks is None:
            pool_inputs = []
            pool = Pool(N_JOBS)

            if self.masks is None:
                for begin, end in chunks_idx(range(len(value)), N_JOBS):
                    pool_inputs.append([begin, end, self.hash_algorithm, self.int_bits, self.index_prefix_for_minus])
                pool_outputs = pool.starmap(_static_prepare_decrypt_single, pool_inputs)
            else:
                pass
            pool.close()
            pool.join()

            temp_minus = []
            for ret in pool_outputs:  # ret is list
                temp_minus += ret
            self.next_iter_decrypt_prepared['minus'] = np.array(temp_minus, dtype=object)

            # LOGGER.info(f"decrypt add {len(self.next_iter_decrypt_prepared['add'])} {self.next_iter_decrypt_prepared['add'][0]}")
            # LOGGER.info(f"decrypt minus {len(self.next_iter_decrypt_prepared['minus'])} {self.next_iter_decrypt_prepared['minus'][0]}")

        else:
            pass  # already generate at set_idx_list_single()

        ret = value - self.next_iter_decrypt_prepared['minus']
        # ret &= ((1 << self.int_bits) - 1)
        if 'minus' in self.next_iter_decrypt_prepared:
            del self.next_iter_decrypt_prepared['minus']
        return ret
    # 多线程操作-解密
    def _multiprocessing_decrypt(self, value):
        if self.masks is None:
            if self.index_prefix_for_minus or self.index_prefix_for_add:
                # LOGGER.info(f'decrypt {self.index_prefix_for_add} {self.index_prefix_for_minus}')
                pool_inputs = []
                # pool = Pool(N_JOBS)

                for begin, end in chunks_idx(range(len(value)), N_JOBS):
                    pool_inputs.append([begin, end, self.hash_algorithm, self.int_bits,
                                        self.index_prefix_for_add, self.index_prefix_for_minus])
                # pool_outputs = pool.starmap(_static_prepare_decrypt, pool_inputs)
                # pool.close()
                # pool.join()
                pool_outputs = Parallel(n_jobs=N_JOBS)(
                    delayed(_static_prepare_decrypt)(*pool_input) for pool_input in pool_inputs)

                temp_add = []
                temp_minus = []
                for ret in pool_outputs:  # ret is list
                    temp_add += ret[0]
                    temp_minus += ret[1]

                if 'add' not in self.next_iter_decrypt_prepared:
                    self.next_iter_decrypt_prepared['add'] = np.array(temp_add, dtype=object)
                    self.next_iter_decrypt_prepared['minus'] = np.array(temp_minus, dtype=object)
                else:
                    self.next_iter_decrypt_prepared['add'] += np.array(temp_add, dtype=object)
                    self.next_iter_decrypt_prepared['add'] &= ((1 << self.int_bits) - 1)
                    self.next_iter_decrypt_prepared['minus'] += np.array(temp_minus, dtype=object)
                    self.next_iter_decrypt_prepared['minus'] &= ((1 << self.int_bits) - 1)
        else:
            pass

        # LOGGER.info(f"decrypt add {len(self.next_iter_decrypt_prepared['add'])} {self.next_iter_decrypt_prepared['add'][0]}")
        # LOGGER.info(f"decrypt minus {len(self.next_iter_decrypt_prepared['minus'])} {self.next_iter_decrypt_prepared['minus'][0]}")
        ret = value + self.next_iter_decrypt_prepared['add'] - self.next_iter_decrypt_prepared['minus']
        ret &= ((1 << self.int_bits) - 1)

        if 'add' in self.next_iter_decrypt_prepared:
            del self.next_iter_decrypt_prepared['add']
        if 'minus' in self.next_iter_decrypt_prepared:
            del self.next_iter_decrypt_prepared['minus']
        if 'add' in self.next_iter_decrypt_prepared_idx:
            del self.next_iter_decrypt_prepared_idx['add']
        if 'minus' in self.next_iter_decrypt_prepared_idx:
            del self.next_iter_decrypt_prepared_idx['minus']

        return ret
    # 解密
    def decrypt(self, ciphertext):
        if self.hash_algorithm is not None:
            if not isinstance(ciphertext, np.ndarray):
                return None
            else:
                if self.masking_scheme == "double":
                    return self._multiprocessing_decrypt(ciphertext)
                else:
                    return self._multiprocessing_decrypt_single(ciphertext)
        else:
            return None


if __name__ == "__main__":
    iter_index = 0
    idx = 0
    index_prefix_for_add = iter_index.to_bytes(4, 'big') + idx.to_bytes(4, 'big')
    index_prefix_for_minus = iter_index.to_bytes(4, 'big') + idx.to_bytes(4, 'big')
    begin, end, hash_algorithm = 0,30,"sha256"
    int_bits = 16
    num_clients = 3
    
    # value = np.random.normal(size=(1, 3000))
    np.random.seed(0)
    value = np.random.randint(low=0,high=30000,size=(num_clients,2000))
    # print(value)
    print(f"聚合后数据：{reduce(lambda x, y:(x + y),value)}")
    flashe = FlasheCipher_Hash(int_bits)
    flashe.set_num_clients(num_clients)
    flashe.set_hash_algorithm(hash_algorithm)
    flashe.set_iter_index(iter_index)
    flashe.idx = idx
    start = time.time()
    encrypt = [flashe.encrypt(value[i]) for i in range(len(value))]
    print("加密时间：",time.time()-start)
    # print(encrypt)
    # reduce(lambda x, y: (x + y), ciphertext)
    agg = reduce(lambda x, y:(x + y),encrypt)
    # print(encrypt.)
    flashe.set_idx_list(raw_idx_list=[0] * num_clients, mode="decrypt")
    print("解密聚合后数据：",flashe.decrypt(agg))
    # print([flashe.decrypt(encrypt[i]) for i in range(len(value))])
    # print(_static_prepare_encrypt(begin, end, hash_algorithm, int_bits, index_prefix_for_add, index_prefix_for_minus))