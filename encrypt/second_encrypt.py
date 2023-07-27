import numpy as np
def generate_random(value,bit_size):
    clients_num = len(value)
    layer_num = len(value[0])
    # 使用列表推导式创建大小为 (a, b) 的列表，初始化为 None
    res  = [[None for _ in range(layer_num)] for _ in range(clients_num)]
    for i in range(layer_num):
        og_shape = value[0][i].shape
        # print(og_shape)
        model_params_size = np.prod(og_shape)
        random_values = generate_random_values(clients_num,model_params_size,bit_size)
        shifted_matrix = shift_and_normalize(random_values)
        # shifted_matrix = shifted_matrix.reshape(og_shape)
        for j in range(clients_num):
            res[j][i] = shifted_matrix[j].reshape(og_shape)
        # res.append(shifted_matrix)
    return res
    pass
def generate_random_values(clients_num, model_params_size,bit_size):
    # 生成范围在0到1之间的随机数矩阵
    random_matrix = np.random.rand(clients_num, model_params_size)

    # 缩放到0到2的bit_size次方之间
    scale_factor = 2 ** bit_size
    scaled_matrix = random_matrix * scale_factor

    return scaled_matrix

def shift_and_normalize(matrix):
    # 求每列的平均值
    column_means = np.mean(matrix, axis=0)
    # print(column_means.shape)
    # 偏移处理，每个值减去对应列的平均值
    shifted_matrix = matrix - column_means
    
    return shifted_matrix.astype(int)

# def main():
#     clients_num = 5  # 假设有5个客户端
#     model_params_size = 10  # 每个模型参数大小为10
#     bit_size = 12
#     # 生成随机数矩阵在0到2的12次方之间
#     random_values = generate_random_power_values(clients_num, model_params_size,bit_size)
#     print("随机数矩阵（0到2的12次方之间）：")
#     print(random_power_values)

#     # 偏移处理并归一化
#     shifted_matrix = shift_and_normalize(random_power_values)
#     print("\n偏移后的矩阵：")
#     print(shifted_matrix.shape)
#     print(np.round(np.sum(shifted_matrix,axis=0)))
#     # 检查偏移后每列的平均值是否接近于0
#     # column_means_after_shift = np.mean(shifted_matrix, axis=0)
#     print("\n偏移后每列的平均值：")
#     # print(column_means_after_shift)
def main():
    bit_size = 12
    value = [[np.random.rand(3, 4), np.random.rand(2, 5), np.random.rand(6, 2)]]*3
    res = generate_random(value,bit_size)
    print()

if __name__ == "__main__":
    main()
