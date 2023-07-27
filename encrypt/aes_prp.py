from encrypt.aes import AESCipher
# 伪随机数生成
class PsuedoRandomPermutation(object):

    def __init__(self):
        super(PsuedoRandomPermutation, self).__init__()
        self.key = None
        self.key_len = None
        self.mode = "ECB"
        # self.mode = "CTR"
        # aes操作
        self.aes = AESCipher()
    # 生成密钥操作
    def generate_key(self, key_length=256, assigned_key=None):
        self.key_len = key_length

        if assigned_key is not None:
            self.key = assigned_key
            self.aes.generate_key(key_length=key_length,
                                  assigned_key=assigned_key,
                                  mode=self.mode)
        else:
            self.aes.generate_key(key_length=key_length,
                                  mode=self.mode)
            self.key = self.aes.get_key()

    def get_permutation(self, index):
        # the input is usually int
        if isinstance(index, int):
            index = index.to_bytes(16, 'big')  # the PRP is 16 bytes

        permutation = self.aes.encrypt(index)
        return permutation
