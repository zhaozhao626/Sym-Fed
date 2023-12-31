import os

from joblib import Parallel, delayed
from encrypt.aes_prp import PsuedoRandomPermutation
import numpy as np
from multiprocessing import Pool, cpu_count

N_JOBS = 1
# LOGGER = log_utils.getLogger()
BITS_PER_BYTES = 8


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


def _static_prepare_encrypt_single(begin, end, prp_seed,
                            int_bits, index_prefix_for_add):
    local_prp = PsuedoRandomPermutation()
    local_prp.generate_key(assigned_key=prp_seed)

    len = end - begin
    merge_size = 128 // int_bits
    merge_num = (len - 1) // merge_size + 1
    add_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)

        # i + begin can guarantee to be unique globally
        index_for_add = index_prefix_for_add + (i + begin).to_bytes(8, 'big')

        add_term_bytes = local_prp.get_permutation(index_for_add)
        add_term_s = int.from_bytes(add_term_bytes, 'big')

        mask = (1 << int_bits) - 1
        for j in range(b, e):
            add_term = add_term_s & mask
            add_terms.append(add_term)
            add_term_s >>= int_bits

    return add_terms


def _static_prepare_encrypt(begin, end, prp_seed, int_bits,
                            index_prefix_for_add, index_prefix_for_minus):
    local_prp = PsuedoRandomPermutation()
    local_prp.generate_key(assigned_key=prp_seed)

    len = end - begin
    merge_size = 128 // int_bits
    merge_num = (len - 1) // merge_size + 1
    add_terms = []
    minus_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)

        # i + begin can guarantee to be unique globally
        index_for_add = index_prefix_for_add + (i + begin).to_bytes(8, 'big')
        index_for_minus = index_prefix_for_minus + (i + begin).to_bytes(8, 'big')

        add_term_bytes = local_prp.get_permutation(index_for_add)
        add_term_s = int.from_bytes(add_term_bytes, 'big')
        minus_term_bytes = local_prp.get_permutation(index_for_minus)
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


def _static_prepare_decrypt_single(begin, end, prp_seed, int_bits, index_prefix_for_minus_list):
    local_prp = PsuedoRandomPermutation()
    local_prp.generate_key(assigned_key=prp_seed)

    len = end - begin
    merge_size = 128 // int_bits
    merge_num = (len - 1) // merge_size + 1
    minus_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)

        # i + begin can guarantee to be unique globally
        index_for_minus_list = [k + (i + begin).to_bytes(8, 'big') for k in index_prefix_for_minus_list ]
        minus_term_s_list = [int.from_bytes(local_prp.get_permutation(k), 'big') for k in index_for_minus_list]

        mask = (1 << int_bits) - 1
        for j in range(b, e):
            minus_term = 0

            for k_idx, k in enumerate(minus_term_s_list):
                minus_term += k
                minus_term_s_list[k_idx] >>= int_bits

            minus_terms.append(minus_term & mask)

    return minus_terms


def _static_prepare_decrypt(begin, end, prp_seed, int_bits,
                            index_prefix_for_add_list, index_prefix_for_minus_list):
    local_prp = PsuedoRandomPermutation()
    local_prp.generate_key(assigned_key=prp_seed)

    len = end - begin
    merge_size = 128 // int_bits
    merge_num = (len - 1) // merge_size + 1
    add_terms = []
    minus_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, len)

        # i + begin can guarantee to be unique globally
        index_for_add_list = [k + (i + begin).to_bytes(8, 'big') for k in index_prefix_for_add_list ]
        index_for_minus_list = [k + (i + begin).to_bytes(8, 'big') for k in index_prefix_for_minus_list ]

        add_term_s_list = [int.from_bytes(local_prp.get_permutation(k), 'big') for k in index_for_add_list]
        minus_term_s_list = [int.from_bytes(local_prp.get_permutation(k), 'big') for k in index_for_minus_list]

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




class FlasheCipher(object):

    def __init__(self, int_bits, mask="double"):
        super(FlasheCipher, self).__init__()

        self.masking_scheme = mask
        self.masks = None
        self.total = None
        # 伪随机函数生成
        self.prp = PsuedoRandomPermutation()
        self.prp_seed = None
        self.prp_seed_len = 256

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
    # 生成伪随机函数种子，用于生成密钥
    def generate_prp_seed(self, assigned_seed=None):
        if assigned_seed is None:
            seed = os.urandom(
                self.prp_seed_len // BITS_PER_BYTES)
        else:
            if isinstance(assigned_seed, int):
                seed = int(assigned_seed & int(
                    2 ** self.prp_seed_len - 1)).to_bytes(
                    self.prp_seed_len, 'big')
            else:
                seed = int(int.from_bytes(assigned_seed, 'big') & int(
                    2 ** self.prp_seed_len - 1)).to_bytes(
                    self.prp_seed_len, 'big')

        self.prp_seed = seed
        self.prp.generate_key(assigned_key=seed)

    def get_prp_seed(self):
        return self.prp_seed
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
                        pool_inputs.append([begin, end, self.prp_seed, self.int_bits, index_prefix_for_minus])

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
            # else:
            #     one_hots = []
            #     for m in self.masks:
            #         a = np.zeros(self.total, dtype=object)
            #         a[m] = 1
            #         one_hots.append(a)

            #     minus = []
            #     add = []
            #     num_clients = len(self.masks)
            #     add.append(np.zeros(self.total, dtype=object))
            #     for client_idx, mask in enumerate(self.masks):
            #         if client_idx > 0:
            #             minus.append(one_hots[client_idx] & ~(one_hots[client_idx - 1]))
            #         else:
            #             minus.append(one_hots[client_idx])

            #         if client_idx < num_clients - 1:
            #             add.append(one_hots[client_idx] & ~(one_hots[client_idx + 1]))
            #         else:
            #             add.append(one_hots[client_idx])

            #     pool_inputs = []
            #     pool = Pool(N_JOBS)

            #     for begin, end in chunks_idx(range(self.total), N_JOBS):
            #         pool_inputs.append([begin, end, self.prp_seed, self.int_bits,
            #                             self.iter_index_bytes, add[begin:end], minus[begin:end]])
            #     pool_outputs = pool.starmap(_static_prepare_decrypt_spar, pool_inputs)
            #     pool.close()
            #     pool.join()

            #     temp_add = []
            #     temp_minus = []
            #     for ret in pool_outputs:  # ret is list
            #         temp_add += ret[0]
            #         temp_minus += ret[1]

            #     self.next_iter_decrypt_prepared['add'] = np.array(temp_add, dtype=object)
            #     self.next_iter_decrypt_prepared['minus'] = np.array(temp_minus, dtype=object)
    # 获取标识符
    def get_idx_list(self):
        return [self.idx]
    # 多线程加密-单掩码
    def _multiprocessing_encrypt_single(self, value):
        pool_inputs = []
        pool = Pool(N_JOBS)

        # LOGGER.info(f'encrypt {self.index_prefix_for_add} {self.index_prefix_for_minus}')
        for begin, end in chunks_idx(range(len(value)), N_JOBS):
            pool_inputs.append([begin, end, self.prp_seed, self.int_bits,
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
                pool_inputs.append([begin, end, self.prp_seed, self.int_bits,
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

            self.next_iter_encrypt_prepared['add'] = np.array(temp_add, dtype=object)
            self.next_iter_encrypt_prepared['minus'] = np.array(temp_minus, dtype=object)

        # LOGGER.info(f"encrypt add {len(self.next_iter_encrypt_prepared['add'])} {self.next_iter_encrypt_prepared['add'][0]}")
        # LOGGER.info(f"encrypt minus {len(self.next_iter_encrypt_prepared['minus'])} {self.next_iter_encrypt_prepared['minus'][0]}")
        ret = value + self.next_iter_encrypt_prepared['add'] - self.next_iter_encrypt_prepared['minus']
        # test = value + self.next_iter_encrypt_prepared['add'] - self.next_iter_encrypt_prepared['minus']
        ret &= ((1 << self.int_bits) - 1)

        if 'add' in self.next_iter_encrypt_prepared:
            del self.next_iter_encrypt_prepared['add']
        if 'minus' in self.next_iter_encrypt_prepared:
            del self.next_iter_encrypt_prepared['minus']

        return ret
    # 加密操作
    def encrypt(self, plaintext):
        if self.prp_seed is not None:
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
                    pool_inputs.append([begin, end, self.prp_seed, self.int_bits, self.index_prefix_for_minus])
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
                    pool_inputs.append([begin, end, self.prp_seed, self.int_bits,
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
        if self.prp_seed is not None:
            if not isinstance(ciphertext, np.ndarray):
                return None
            else:
                if self.masking_scheme == "double":
                    return self._multiprocessing_decrypt(ciphertext)
                else:
                    return self._multiprocessing_decrypt_single(ciphertext)
        else:
            return None
