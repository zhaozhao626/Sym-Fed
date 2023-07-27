import sys
sys.path.append(".")
# print(sys.path)
import numpy as np
from encrypt.aciq import ACIQ
# from encrypt.util import consts
# from multiprocessing import cpu_count, Pool
from encrypt.twocomplement import TwoComplement
import tensorflow as tf
import os
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count  
from concurrent.futures import ThreadPoolExecutor
import queue
import concurrent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# N_JOBS = cpu_count()
N_JOBS = 2


def _static_quantize_padding(value, alpha, int_bits, num_clients):
    # # 首先裁剪
    # value = np.clip(value, -alpha, alpha)

    # max_value = 2**(int_bits) - 1
    # scaled_value = (value + alpha) / (2 * alpha)  # 将输入值缩放到[0, 1]范围内
    # # print(scaled_value)
    # size = value.shape
    # value = np.floor(scaled_value * max_value + np.random.random(size)).astype(int) # float to int
    # # print(value)
    # # mapped_value = int(round(scaled_value * max_value))  # 将缩放后的值映射到[0, max_value]范围内
    # return value
    
     # 首先裁剪
    value = np.clip(value, -alpha, alpha)

    # 然后量化
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (pow(2, int_bits) - 1.0) / alpha
    value = unsigned_value * sign

    # then stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int) # float to int
    value = value.astype(object)  # np.int to int (to avoid being tranferred to float later)
    
    
    # padding_bits = int(np.ceil(np.log2(num_clients)))
    
    
    return value



def _static_unquantize_padding(value, alpha, int_bits, num_clients):
    factor = int(np.ceil(np.log2(num_clients)))
    int_bits = factor + int_bits
    alpha *= 2 ** factor
    # factor = num_clients
    # alpha *= factor
    # alpha *= 10
    value &= ((1 << int_bits) - 1)
    sign = tf.where(value > pow(2, int_bits - 1), -1, 1)
    
    # max_value = (2**(int_bits) - 1)*factor
    
    # scaled_value = value / max_value  # 将映射后的值缩放到[0, 1]范围内
    # value = scaled_value * (2 * alpha) - alpha  # 将缩放后的值还原到[-a, a]范围内
    
    return value


if  __name__=="__main__":
    a = _static_quantize_padding(0.5,2,5,10)
    print(a)
    print(_static_unquantize_padding(a*10,2,5,10))

# def _static_quantize_padding(value, alpha, int_bits, num_clients):
#     # 首先裁剪
#     value = np.clip(value, -alpha, alpha)

#     # 然后量化
#     sign = np.sign(value)
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * (pow(2, int_bits - 1) - 1.0) / alpha

#     value = unsigned_value + tf.nn.relu(-sign)*(pow(2, int_bits - 1))

#     # then stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int) # float to int
#     value = value.astype(object)  # np.int to int (to avoid being tranferred to float later)
#     # 将正负小数映射到整数区间 
#     # padding_bits = int(np.ceil(np.log2(num_clients)))
#     # int_bits = int_bits+padding_bits
#     # mod = 2 ** int_bits
#     # value = value % mod
    
#     return value


# def _static_unquantize_padding(value, alpha, int_bits, num_clients):
#     factor = int(np.ceil(np.log2(num_clients)))
#     # int_bits = factor + int_bits
#     # alpha *= 2 ** factor

#     value = -value+(pow(2, int_bits - 1) - 1.0)
#     # then unquantize
#     sign = np.sign(value).astype(np.float32)
#     value += tf.nn.relu(-sign)
    
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * alpha / (pow(2, int_bits - 1) - 1.0)
#     value = unsigned_value * sign

#     return value


def _static_batch(array, int_bits, element_bits, factor):
    
    element_bits += factor 
    # 一个batch多大
    batch_size = int_bits // element_bits
    # print(batch_size)
    if  len(array) % batch_size == 0  :
        pass
    else:
        pad_zero_nums = batch_size - len(array) % batch_size
        pad_zeros = [0] * pad_zero_nums
        array = np.append(array, pad_zeros)
    # 多少batch 
    batch_nums = len(array) // batch_size

    ret = []
    mod = 2 ** element_bits
    for b in range(batch_nums):
        temp = 0
        for i in range(batch_size):
            temp *= mod
            temp += array[i + b * batch_size]

        ret.append(temp)

    return np.array(ret).astype(object)


def _static_unbatch(array, int_bits, element_bits, factor): 
    true_element_bits = element_bits + factor
    element_bits += factor 
    batch_size = int_bits // element_bits

    ret = []
    mask = 2 ** element_bits - 1
    for item in array:
        temp = []
        for i in range(batch_size):
            num = item & mask
            temp.append(num)
            item >>= element_bits

        temp.reverse()
        ret += temp
    ret = np.array(ret)

    mod = 2 ** true_element_bits
    ret = ret % mod
    return ret



def get_alpha_r_max(trainable_list, element_bits,num_clients):
    local_min_list = []
    local_max_list = []
    size_list = []
    for idx, trainable in enumerate(trainable_list):
        local_min = []
        local_max = []
        for layer in trainable:
            local_min.append(np.amin(layer))
            local_max.append(np.amax(layer))

            if idx == 0:
                size_list.append(layer.size)

        local_min_list.append(np.array(local_min))
        local_max_list.append(np.array(local_max))

    local_min_list = np.array(local_min_list)
    local_max_list = np.array(local_max_list)

    min_list = np.amin(local_min_list, 0)
    max_list = np.amax(local_max_list, 0)

    n = len(trainable_list)
    aciq = ACIQ(element_bits)

    alpha_list = []
    r_max_list = []
    layer_cnt = 0
    for min, max in zip(min_list, max_list):
        if size_list[layer_cnt] == 1:
            alpha = max
        else:
            alpha = aciq.get_alpha_gaus(min, max, size_list[layer_cnt])

        r_max = alpha * num_clients

        alpha_list.append(alpha)
        r_max_list.append(r_max)
        layer_cnt += 1

    return alpha_list, r_max_list

# def quantize(trainable_list, element_bits,clipping_thresholds):
def quantize(trainable_list, element_bits,alpha_list):
    n = len(trainable_list)

    quantized = []
    og_shape = []
    for _, trainable in enumerate(trainable_list):
        quantized_layers = []
        # shape = [] 
        for idx, layer in enumerate(trainable):
            # origin shape
            shape_ = layer.shape
            # flatten the matrix
            layer_flatten = layer.flatten()
            # quantize the gradient
            ret = _static_quantize_padding(layer_flatten,
                                           alpha_list[idx],
                                           element_bits,
                                           n)
            # convert the type to array
            ret = np.array(ret).reshape(shape_)
            # ret = np.array(ret)
            # shape.append(shape_)
            quantized_layers.append(ret)
        quantized.append(np.array(quantized_layers,dtype=object))
        # og_shape.append(shape)
    return np.array(quantized),np.array(alpha_list)


def flatten_quantize(trainable_list, element_bits,alpha_list):
    n = len(trainable_list)

    quantized = []
    og_shape = []
    for _, trainable in enumerate(trainable_list):
        quantized_layers = []
        shape = [] 
        for idx, layer in enumerate(trainable):
            # origin shape
            shape_ = layer.shape
            # flatten the matrix
            layer_flatten = layer.flatten()
            # quantize the gradient
            ret = _static_quantize_padding(layer_flatten,
                                           alpha_list[idx],
                                           element_bits,
                                           n)
            # convert the type to array
            # ret = np.array(ret).reshape(shape_)
            ret = np.array(ret)
            shape.append(shape_)
            quantized_layers.append(ret)
        quantized.append(np.array(quantized_layers,dtype=object))
        og_shape.append(shape)
    return np.array(quantized),np.array(alpha_list),og_shape

def unquantize(trainable, alpha_list, element_bits, num_clients):
    layers = []
    for idx, layer in enumerate(trainable):
        shape = layer.shape
        layer_flatten = layer.flatten()
        
        ret = _static_unquantize_padding(layer_flatten,
                                         alpha_list[idx],
                                         element_bits,
                                         num_clients)
        ret = tf.convert_to_tensor(ret, dtype=tf.float32)
        ret = np.array(ret).reshape(shape)
        layers.append(ret)

    layers = np.array(layers,dtype=object)
    return layers
def flatten_unquantize(trainable, alpha_list, element_bits, num_clients, og_shape):
    layers = []
    for idx, layer in enumerate(trainable):
        # shape = layer.shape
        # layer_flatten = layer.flatten()
        
        ret = _static_unquantize_padding(layer,
                                         alpha_list[idx],
                                         element_bits,
                                         num_clients)
        ret = tf.convert_to_tensor(ret, dtype=tf.float32)
        ret = np.array(ret).reshape(og_shape[idx])
        layers.append(ret)

    layers = np.array(layers,dtype=object)
    return layers



# 池化
def batch_pool(trainable_list, int_bits, element_bits, factor, pool_size=4):
    n = len(trainable_list)

    batch_list = []
    og_shapes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:  # 最大工作进程数为 pool_size
        futures = []
        for _, trainable in enumerate(trainable_list):
            batch_layers = []
            shape_ = []
            for idx, layer in enumerate(trainable):
                # 原来的形状
                shape = layer.shape
                layer_flatten = layer.flatten()
                # 异步提交任务
                futures.append(executor.submit(_static_batch, layer_flatten, int_bits, element_bits, factor))
                shape_.append(shape)
            og_shapes.append(shape_)
            # 每当任务数等于池的大小时，等待所有任务完成
            if len(futures) == pool_size:
                batch_layers = [f.result() for f in futures]
                batch_list.append(np.array(batch_layers, dtype=object))
                futures = []
        # 处理剩余任务
        if len(futures) > 0:
            batch_layers = [f.result() for f in futures]
            batch_list.append(np.array(batch_layers, dtype=object))

    return np.array(batch_list), og_shapes



# 队列

def batch2(trainable_list, int_bits, element_bits, factor,mode="queue"):
    n = len(trainable_list)
    q = queue.Queue()  # 创建队列
    batch_list = []
    og_shapes = []
    for _, trainable in enumerate(trainable_list):
        batch_layers = []
        shape_ = []
        for idx, layer in enumerate(trainable):
            # 原来的形状
            shape = layer.shape
            layer_flatten = layer.flatten()
            # 将任务加入队列
            q.put((layer_flatten, int_bits, element_bits, factor))
            shape_.append(shape)
        og_shapes.append(shape_)
        while not q.empty():  # 遍历队列中的任务
            layer_flatten, int_bits, element_bits, factor = q.get()
            ret = _static_batch(layer_flatten, int_bits, element_bits, factor)
            batch_layers.append(ret)
        batch_list.append(np.array(batch_layers, dtype=object))

    return np.array(batch_list), og_shapes

# 线程池并行2

def batch3(trainable_list, int_bits, element_bits, factor):
    n = 4

    batch_list = []
    og_shapes = []
    with ThreadPoolExecutor(max_workers=n) as executor:  # 最大工作线程数为 n
        futures = []
        for _, trainable in enumerate(trainable_list):
            batch_layers = []
            shape_ = []
            for idx, layer in enumerate(trainable):
                # 原来的形状
                shape = layer.shape
                layer_flatten = layer.flatten()
                # 异步提交任务
                futures.append(executor.submit(_static_batch, layer_flatten, int_bits, element_bits, factor))
                shape_.append(shape)
            og_shapes.append(shape_)
            # 将所有任务提交到线程池后，等待所有任务完成
            batch_layers = [f.result() for f in futures]
            batch_list.append(np.array(batch_layers, dtype=object))

    return np.array(batch_list), og_shapes

# 并行1
# def batch(trainable_list, int_bits, element_bits, factor):
#     n = len(trainable_list)

#     batch_list = []
#     og_shapes = []
#     for _, trainable in enumerate(trainable_list):
#         batch_layers = []
#         shape_ = []
#         pool_inputs = []
#         # for begin, end in chunks_idx(range(len(value)), N_JOBS):
            
#         for idx, layer in enumerate(trainable):
#             # 原来的形状
#             shape = layer.shape
#             layer_flatten = layer.flatten()
#             pool_inputs.append([layer_flatten, int_bits,element_bits,factor])
#             shape_.append(shape)
            
#         pool_outputs = Parallel(n_jobs=N_JOBS)(
#             delayed(_static_batch)(*pool_input) for pool_input in pool_inputs)
#         for pool_output in pool_outputs:
#             batch_layers.append(pool_output)       
            
#         og_shapes.append(shape_)
#         # print(shape_)
#         batch_list.append(np.array(batch_layers,dtype=object))
#     return np.array(batch_list),og_shapes

# 未并行
def batch_(trainable_list, int_bits, element_bits, factor):
    n = len(trainable_list)

    batch_list = []
    og_shapes = []
    for _, trainable in enumerate(trainable_list):
        batch_layers = []
        shape_ = []
        for idx, layer in enumerate(trainable):
            # 原来的形状
            shape = layer.shape
            layer_flatten = layer.flatten()
            ret = _static_batch(layer_flatten,
                                int_bits,
                                element_bits,
                                factor)
            # ret = np.array(ret).reshape(shape)
            shape_.append(shape)
            batch_layers.append(ret)
        og_shapes.append(shape_)
        # print(shape_)
        batch_list.append(np.array(batch_layers,dtype=object))
    return np.array(batch_list),og_shapes
    


def batch(trainable_list, int_bits, element_bits, factor, mode = "none"):
    if mode=="pool":
        n = len(trainable_list)
        pool_size = 4
        batch_list = []
        og_shapes = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:  # 最大工作进程数为 pool_size
            futures = []
            for _, trainable in enumerate(trainable_list):
                batch_layers = []
                shape_ = []
                for idx, layer in enumerate(trainable):
                    # 原来的形状
                    shape = layer.shape
                    layer_flatten = layer.flatten()
                    # 异步提交任务
                    futures.append(executor.submit(_static_batch, layer_flatten, int_bits, element_bits, factor))
                    shape_.append(shape)
                og_shapes.append(shape_)
                # 每当任务数等于池的大小时，等待所有任务完成
                if len(futures) == pool_size:
                    batch_layers = [f.result() for f in futures]
                    batch_list.append(np.array(batch_layers, dtype=object))
                    futures = []
            # 处理剩余任务
            if len(futures) > 0:
                batch_layers = [f.result() for f in futures]
                batch_list.append(np.array(batch_layers, dtype=object))

        return np.array(batch_list), og_shapes
        pass
    elif mode == "queue":
        n = len(trainable_list)
        q = queue.Queue()  # 创建队列
        batch_list = []
        og_shapes = []
        for _, trainable in enumerate(trainable_list):
            batch_layers = []
            shape_ = []
            for idx, layer in enumerate(trainable):
                # 原来的形状
                shape = layer.shape
                layer_flatten = layer.flatten()
                # 将任务加入队列
                q.put((layer_flatten, int_bits, element_bits, factor))
                shape_.append(shape)
            og_shapes.append(shape_)
            while not q.empty():  # 遍历队列中的任务
                layer_flatten, int_bits, element_bits, factor = q.get()
                ret = _static_batch(layer_flatten, int_bits, element_bits, factor)
                batch_layers.append(ret)
            batch_list.append(np.array(batch_layers, dtype=object))

        return np.array(batch_list), og_shapes
        pass
    
    elif mode == "pool_":
        n = 4
        batch_list = []
        og_shapes = []
        with ThreadPoolExecutor(max_workers=n) as executor:  # 最大工作线程数为 n
            futures = []
            for _, trainable in enumerate(trainable_list):
                batch_layers = []
                shape_ = []
                for idx, layer in enumerate(trainable):
                    # 原来的形状
                    shape = layer.shape
                    layer_flatten = layer.flatten()
                    # 异步提交任务
                    futures.append(executor.submit(_static_batch, layer_flatten, int_bits, element_bits, factor))
                    shape_.append(shape)
                og_shapes.append(shape_)
                # 将所有任务提交到线程池后，等待所有任务完成
                batch_layers = [f.result() for f in futures]
                batch_list.append(np.array(batch_layers, dtype=object))

        return np.array(batch_list), og_shapes
        pass
    
    elif mode == "none":
        n = len(trainable_list)

        batch_list = []
        og_shapes = []
        for _, trainable in enumerate(trainable_list):
            batch_layers = []
            shape_ = []
            for idx, layer in enumerate(trainable):
                # 原来的形状
                shape = layer.shape
                layer_flatten = layer.flatten()
                ret = _static_batch(layer_flatten,
                                    int_bits,
                                    element_bits,
                                    factor)
                # ret = np.array(ret).reshape(shape)
                shape_.append(shape)
                batch_layers.append(ret)
            og_shapes.append(shape_)
            # print(shape_)
            batch_list.append(np.array(batch_layers,dtype=object))
        return np.array(batch_list),og_shapes


def unbatch(trainable,og_shapes, int_bits, element_bits, factor, mode = "none"): 
    if mode=="pool":
        layers = []
        
        def process_layer(layer, og_shape):
            layer_flatten = layer.flatten()
            ret = _static_unbatch(layer_flatten, int_bits, element_bits, factor)
            num_ele = np.prod(og_shape)
            ret = ret[:num_ele]
            ret = np.reshape(ret, og_shape)
            return ret
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for layer, og_shape in zip(trainable, og_shapes):
                results.append(executor.submit(process_layer, layer, og_shape))
            
            for result in concurrent.futures.as_completed(results):
                layers.append(result.result())
        
        layers = np.array(layers, dtype=object)
        return layers

        """
            引入队列的作用是在多线程或多进程环境中实现线程/进程间的安全数据传递和同步。

            在给定的代码中，引入队列可以用于以下目的：

            并发处理：队列可以用于将每个层的处理任务放入队列中，然后使用多个线程或进程同时处理队列中的任务。这样可以提高处理速度，尤其是当处理每个层的操作比较耗时时。

            结果收集：处理每个层的任务完成后，可以将结果放入队列中，然后按照完成的顺序从队列中取出结果。这样可以确保结果的顺序性，并且可以在结果可用时立即进行后续操作。

            数据传递：队列可以用于将每个层的处理结果传递给其他线程或进程进行后续处理。通过队列，不同的线程或进程之间可以安全地共享数据，避免了数据竞争和同步问题。
        """
    elif mode == "queue":
        layers = []
        result_queue = queue.Queue()
        
        def process_layer(layer, og_shape):
            layer_flatten = layer.flatten()
            ret = _static_unbatch(layer_flatten, int_bits, element_bits, factor)
            num_ele = np.prod(og_shape)
            ret = ret[:num_ele]
            ret = np.reshape(ret, og_shape)
            result_queue.put(ret)
        
        for layer, og_shape in zip(trainable, og_shapes):
            process_layer(layer, og_shape)
        
        while not result_queue.empty():
            result = result_queue.get()
            layers.append(result)
        
        layers = np.array(layers, dtype=object)
        return layers
    elif mode == "pool_":
        layers = []
        for layer, og_shape in zip(trainable, og_shapes):
            layer_flatten = layer.flatten()
            ret = _static_unbatch(layer_flatten, int_bits, element_bits, factor)
            num_ele = np.prod(og_shape)
            ret = ret[:num_ele]
            # print(ret)
            ret = np.reshape(ret, og_shape)
            
            # ret = tf.convert_to_tensor(ret, dtype=tf.float32)
            
            # ret = np.array(ret).reshape(og_shape)
            layers.append(ret)

        layers = np.array(layers,dtype=object)
        return layers
    elif mode == "none":
        layers = []
        for layer, og_shape in zip(trainable, og_shapes):
            layer_flatten = layer.flatten()
            ret = _static_unbatch(layer_flatten, int_bits, element_bits, factor)
            num_ele = np.prod(og_shape)
            ret = ret[:num_ele]
            # print(ret)
            ret = np.reshape(ret, og_shape)
            
            # ret = tf.convert_to_tensor(ret, dtype=tf.float32)
            
            # ret = np.array(ret).reshape(og_shape)
            layers.append(ret)

        layers = np.array(layers,dtype=object)
        return layers

def unbatch_(trainable,og_shapes, int_bits, element_bits, factor): 
    layers = []
    for layer, og_shape in zip(trainable, og_shapes):
        layer_flatten = layer.flatten()
        ret = _static_unbatch(layer_flatten, int_bits, element_bits, factor)
        num_ele = np.prod(og_shape)
        ret = ret[:num_ele]
        # print(ret)
        ret = np.reshape(ret, og_shape)
        
        # ret = tf.convert_to_tensor(ret, dtype=tf.float32)
        
        # ret = np.array(ret).reshape(og_shape)
        layers.append(ret)

    layers = np.array(layers,dtype=object)
    return layers