import numpy as np
from encrypt.encryption import gmpy_math
import random
from encrypt.encryption.fixedpoint import FixedPointNumber
# from multiprocessing import Pool
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
N_JOBS = cpu_count()

def _static_encrypt(value, n):
    pubkey = PaillierPublicKey(n)
    return pubkey.encrypt(value)


def _static_decrypt(value, n, p, q):
    pubkey = PaillierPublicKey(n)
    prikey = PaillierPrivateKey(pubkey, p, q)
    return prikey.decrypt(value)


class PaillierKeypair(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_keypair(n_length=1024):
        p = q = n = None
        n_len = 0

        while n_len != n_length:
            p = gmpy_math.getprimeover(n_length // 2)
            q = p
            while q == p:
                q = gmpy_math.getprimeover(n_length // 2)
            n = p * q
            n_len = n.bit_length()

        public_key = PaillierPublicKey(n)
        private_key = PaillierPrivateKey(public_key, p, q)

        return public_key, private_key


class PaillierPublicKey(object):
    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def get_n(self):
        return self.n

    def __repr__(self):
        hashcode = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def apply_obfuscator(self, ciphertext):
        r = random.SystemRandom().randrange(1, self.n)
        obfuscator = gmpy_math.powmod(r, self.n, self.nsquare)

        return (ciphertext * obfuscator) % self.nsquare

    def encrypt(self, plaintext):
        if plaintext >= (self.n - self.max_int) and plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            ciphertext = gmpy_math.invert(neg_ciphertext, self.nsquare)
        else:
            ciphertext = (self.n * plaintext + 1) % self.nsquare

        ciphertext = self.apply_obfuscator(ciphertext)
        return ciphertext


class PaillierPrivateKey(object):
    def __init__(self, public_key, p, q):
        if not p * q == public_key.n:
            raise ValueError("given public key does not match the given p and q")
        if p == q:
            raise ValueError("p and q have to be different")
        self.public_key = public_key
        if q < p:
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p
        self.qsquare = self.q * self.q
        self.q_inverse = gmpy_math.invert(self.q, self.p)
        self.hp = self.h_func(self.p, self.psquare)
        self.hq = self.h_func(self.q, self.qsquare)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))

    def __repr__(self):
        hashcode = hex(hash(self))[2:]

        return "<PaillierPrivateKey {}>".format(hashcode[:10])

    def h_func(self, x, xsquare):
        return gmpy_math.invert(self.l_func(gmpy_math.powmod(self.public_key.g,
                                                                 x - 1, xsquare), x), x)

    def l_func(self, x, p):
        return (x - 1) // p

    def crt(self, mp, mq):
        u = (mp - mq) * self.q_inverse % self.p
        x = (mq + (u * self.q)) % self.public_key.n

        return x

    def decrypt(self, ciphertext):
        mp = self.l_func(gmpy_math.powmod(ciphertext,
                                              self.p-1, self.psquare),
                                              self.p) * self.hp % self.p
        mq = self.l_func(gmpy_math.powmod(ciphertext,
                                              self.q-1, self.qsquare),
                                              self.q) * self.hq % self.q

        plaintext = self.crt(mp, mq)
        return plaintext

    def get_p_q(self):
        return self.p, self.q


class PaillierEncryptedNumber(object):
    """Represents the Paillier encryption of a float or int.
    """

    def __init__(self, public_key, ciphertext, exponent=0):
        self.public_key = public_key
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscator = False

        if not isinstance(self.__ciphertext, int):
            raise TypeError("ciphertext should be an int, not: %s" % type(self.__ciphertext))

        if not isinstance(self.public_key, PaillierPublicKey):
            raise TypeError("public_key should be a PaillierPublicKey, not: %s" % type(self.public_key))

    def ciphertext(self, be_secure=True):
        """return the ciphertext of the PaillierEncryptedNumber.
        """
        if be_secure and not self.__is_obfuscator:
            self.apply_obfuscator()

        return self.__ciphertext

    def apply_obfuscator(self):
        """ciphertext by multiplying by r ** n with random r
        """
        self.__ciphertext = self.public_key.apply_obfuscator(self.__ciphertext)
        self.__is_obfuscator = True

    def __add__(self, other):
        if isinstance(other, PaillierEncryptedNumber):
            return self.__add_encryptednumber(other)
        else:
            return self.__add_scalar(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def __mul__(self, scalar):
        """return Multiply by an scalar(such as int, float)
        """

        encode = FixedPointNumber.encode(scalar, self.public_key.n, self.public_key.max_int)
        plaintext = encode.encoding

        if plaintext < 0 or plaintext >= self.public_key.n:
            raise ValueError("Scalar out of bounds: %i" % plaintext)

        if plaintext >= self.public_key.n - self.public_key.max_int:
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = gmpy_math.invert(self.ciphertext(False), self.public_key.nsquare)
            neg_scalar = self.public_key.n - plaintext
            ciphertext = gmpy_math.powmod(neg_c, neg_scalar, self.public_key.nsquare)
        else:
            ciphertext = gmpy_math.powmod(self.ciphertext(False), plaintext, self.public_key.nsquare)

        exponent = self.exponent + encode.exponent

        return PaillierEncryptedNumber(self.public_key, ciphertext, exponent)

    def increase_exponent_to(self, new_exponent):
        """return PaillierEncryptedNumber:
           new PaillierEncryptedNumber with same value but having great exponent.
        """
        if new_exponent < self.exponent:
            raise ValueError("New exponent %i should be great than old exponent %i" % (new_exponent, self.exponent))

        factor = pow(FixedPointNumber.BASE, new_exponent - self.exponent)
        new_encryptednumber = self.__mul__(factor)
        new_encryptednumber.exponent = new_exponent

        return new_encryptednumber

    def decrease_exponent_to(self, new_exponent):
        """return PaillierEncryptedNumber:
           new PaillierEncryptedNumber with same value but having great exponent.
        """
        if new_exponent > self.exponent:
            raise ValueError("New exponent %i should be smaller than old exponent %i" % (new_exponent, self.exponent))

        factor = pow(FixedPointNumber.BASE, self.exponent - new_exponent)
        new_encryptednumber = self.__mul__(factor)
        new_encryptednumber.exponent = new_exponent

        return new_encryptednumber

    def __align_exponent(self, x, y):
        """return x,y with same exponet
        """
        if x.exponent < y.exponent:
            x = x.increase_exponent_to(y.exponent)
        elif x.exponent > y.exponent:
            y = y.increase_exponent_to(x.exponent)

        return x, y

    def __add_scalar(self, scalar):
        """return PaillierEncryptedNumber: z = E(x) + y
        """
        encoded = FixedPointNumber.encode(scalar,
                                          self.public_key.n,
                                          self.public_key.max_int,
                                          max_exponent=self.exponent)

        return self.__add_fixpointnumber(encoded)

    def __add_fixpointnumber(self, encoded):
        """return PaillierEncryptedNumber: z = E(x) + FixedPointNumber(y)
        """
        if self.public_key.n != encoded.n:
            raise ValueError("Attempted to add numbers encoded against different public keys!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, encoded)

        encrypted_scalar = x.public_key.raw_encrypt(y.encoding, 1)
        encryptednumber = self.__raw_add(x.ciphertext(False), encrypted_scalar, x.exponent)

        return encryptednumber

    def __add_encryptednumber(self, other):
        """return PaillierEncryptedNumber: z = E(x) + E(y)
        """
        if self.public_key != other.public_key:
            raise ValueError("add two numbers have different public key!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, other)

        encryptednumber = self.__raw_add(x.ciphertext(False), y.ciphertext(False), x.exponent)

        return encryptednumber

    def __raw_add(self, e_x, e_y, exponent):
        """return the integer E(x + y) given ints E(x) and E(y).
        """
        ciphertext = e_x * e_y % self.public_key.nsquare

        return PaillierEncryptedNumber(self.public_key, ciphertext, exponent)




class PaillierCipher():
    def __init__(self):
        super(PaillierCipher, self).__init__()
        self.uuid = None
        self.exchanged_keys = None
        self.n = None
        self.key_length = None

    def set_n(self, n):  # for all (arbiter is necessary, while host and guest is optional since they dont add)
        self.n = n

    def get_n(self):
        return self.n

    def set_self_uuid(self, uuid):
        self.uuid = uuid

    def set_exchanged_keys(self, keys):
        self.exchanged_keys = keys

    def generate_key(self, n_length=2048):
        self.key_length = n_length
        self.public_key, self.privacy_key = \
            PaillierKeypair.generate_keypair(n_length=n_length)
        self.set_n(self.public_key.n)

    def get_key_pair(self):
        return self.public_key, self.privacy_key

    def set_public_key(self, public_key):
        self.public_key = public_key
        # for host
        self.set_n(public_key.n)

    def get_public_key(self):
        return self.public_key

    def set_privacy_key(self, privacy_key):
        self.privacy_key = privacy_key

    def get_privacy_key(self):
        return self.privacy_key

    def _dynamic_encrypt(self, value):
        return self.public_key.encrypt(value)

    def _multiprocessing_encrypt(self, value):
        shape = value.shape
        value_flatten = value.flatten()
        n = self.public_key.get_n()

        pool_inputs = []
        for i in range(len(value_flatten)):
            pool_inputs.append([value_flatten[i], n])

        # pool = Pool(N_JOBS)
        # ret = pool.starmap(_static_encrypt, pool_inputs)
        # pool.close()
        # pool.join()
        # print("test2")
        ret = Parallel(n_jobs=N_JOBS)(
            delayed(_static_encrypt)(value,n) for value in value_flatten)

        # print("test2")
        ret = np.array(ret)
        return ret.reshape(shape)

    def encrypt(self, value):
        if self.public_key is not None:
            if not isinstance(value, np.ndarray):
                return self._dynamic_encrypt(value)
            else:
                return self._multiprocessing_encrypt(value)
        else:
            return None

    def _dynamic_decrypt(self, value):
        return self.privacy_key.decrypt(value)

    def _multiprocessing_decrypt(self, value):
        shape = value.shape
        value_flatten = value.flatten()
        n = self.public_key.get_n()
        p, q = self.privacy_key.get_p_q()

        pool_inputs = []
        for i in range(len(value_flatten)):
            pool_inputs.append(
                [value_flatten[i], n, p, q]
            )

        # pool = Pool(N_JOBS)
        # ret = pool.starmap(_static_decrypt, pool_inputs)
        # pool.close()
        # pool.join()
        # ret = Parallel(n_jobs=N_JOBS)(
        #     delayed(_static_decrypt)(*pool_input) for pool_input in pool_inputs)
        ret = Parallel(n_jobs=N_JOBS)(
        delayed(_static_decrypt)(int(value), n, p, q) for value in value_flatten)
        ret = np.array(ret)
        return ret.reshape(shape)

    def decrypt(self, value):
        if self.privacy_key is not None:
            if not isinstance(value, np.ndarray):
                return self._dynamic_decrypt(value)
            else:
                return self._multiprocessing_decrypt(value)
        else:
            return None
