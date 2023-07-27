'''
Descripttion: 
Author: wjz
Date: 2023-02-14 20:44:49
LastEditors: wjz
LastEditTime: 2023-05-08 08:51:03
'''
from Crypto.Cipher import AES
from Crypto.Util import Counter
import os


class AESCipher(object):

    def __init__(self):
        super(AESCipher, self).__init__()
        self.key = None
        self.cipher = None
        self.key_len_bytes = None
    # AES256是一种分组密码。它将一个32字节的密钥和一个16字节的字符串作为输入，称为块并输出一个块
    def generate_key(self, key_length=256, assigned_key=None, mode="CTR"):
        # 32byte
        key_length_in_bytes = key_length // 8
        self.key_len_bytes = key_length_in_bytes

        if not assigned_key:
            key = os.urandom(key_length_in_bytes)
        else:
            if not isinstance(assigned_key, bytes):
                key = (int(assigned_key) & int(
                    256 ** key_length_in_bytes - 1)).to_bytes(
                    key_length_in_bytes, 'big')
            else:  # of type bytes
                key = (int.from_bytes(assigned_key, 'big') & int(
                    256 ** key_length_in_bytes - 1)).to_bytes(
                    key_length_in_bytes, 'big')

        self.key = key
        if mode == "CTR":
            self.cipher = AES.new(key, AES.MODE_CTR, counter = Counter.new(128, initial_value = 0))
        elif mode == "ECB":
            self.cipher = AES.new(key, AES.MODE_ECB)

    def encrypt(self, plaintext):
        if self.cipher is not None:
            ciphertext = self.cipher.encrypt(plaintext)
            return ciphertext
        else:
            return None

    def decrypt(self, ciphertext):
        if self.cipher is not None:
            plaintext = self.cipher.decrypt(ciphertext)
            return plaintext
        else:
            return None

    def get_key(self):
        return self.key
