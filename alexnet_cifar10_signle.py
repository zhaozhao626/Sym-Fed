# import shutil
import datetime
import time
import csv
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from functools import reduce
from encrypt.ASHE_hash_signle import FlasheCipher_Hash
# 数据增强模块  
from util import augmentation
# from joblib import Parallel, delayed
# import multiprocessing
# from encrypt.wjz_flashe import FlasheCipher
# from encrypt.second_encrypt import generate_random
from encrypt.ASHE import FlasheCipher
from encrypt.paillier import PaillierCipher
from util.genernal import eval_traffic, increment_path
# from encrypt import encryption
# from encrypt.wjz_aciq import ACIQ
from encrypt.quantize import \
    batch, flatten_quantize, flatten_unquantize, unbatch,quantize,unquantize,get_alpha_r_max
# from encrypt.quantize_unsign import \
#     batch, flatten_quantize, flatten_unquantize, unbatch,quantize,unquantize,get_alpha_r_max
from multiprocessing import cpu_count
import logging
import os

import pickle 

N_JOBS = cpu_count()
# N_JOBS = 50

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = y_train.squeeze().astype(np.int32)
y_test = y_test.squeeze().astype(np.int32)

# 为每个client创建数据
def build_datasets(num_clients):
    avg_length = int(len(x_train) / num_clients)
    split_idx = [_ * avg_length for _ in range(1, num_clients)]

    x_train_clients = np.split(x_train, split_idx)
    y_train_clients = np.split(y_train, split_idx)

    logging.info("{} clients building datasets.".format(len(x_train_clients)))
    for idx, x_train_client in enumerate(x_train_clients):
        logging.info("{} client has {} data items.".format(idx, len(x_train_client)))

    train_dataset_clients = [tf.data.Dataset.from_tensor_slices(item) for item in zip(x_train_clients, y_train_clients)]

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 1000

    for i in range(len(train_dataset_clients)):
        # num_parallel_calls=N_JOBS加快速度
        train_dataset_clients[i] = train_dataset_clients[i].map(
            augmentation.augment_img,
            num_parallel_calls=N_JOBS)
        # train_dataset_clients[i] = train_dataset_clients[i].map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))
        train_dataset_clients[i] = train_dataset_clients[i].shuffle(SHUFFLE_BUFFER_SIZE,
                                                                    reshuffle_each_iteration=True).batch(BATCH_SIZE)

    return train_dataset_clients


model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                              input_shape=x_train.shape[1:]))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes))
model.add(keras.layers.Activation('softmax'))
# model.summary()

cce = tf.keras.losses.SparseCategoricalCrossentropy()


def loss(model, x, y):
    y_ = model(x)
    return cce(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
optimizer = tf.optimizers.RMSprop(learning_rate=0.0001)
global_step = tf.Variable(0)


# 裁剪梯度
def clip_gradients(grads, min_v, max_v):
    results = [tf.clip_by_value(t, min_v, max_v).numpy() for t in grads]
    return results


def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        results.append(x1[i] + x2[i])
    return results


# 聚合梯度
def aggregate_gradients(gradient_list, weight=0.5):
    results = reduce(do_sum, gradient_list)
    return np.array(results,dtype=object)


def aggregate_losses(loss_list):
    return np.sum(loss_list)


def flashe_encrypt(value):
    return flashe.encrypt(value)


def flashe_decrypt(value):
    flashe.set_idx_list(raw_idx_list=[0] * num_clients, mode="decrypt")
    return flashe.decrypt(value)

def paillier_encrypt(value):
    # print(value.shape)
    layers = []
    for idx, layer in enumerate(value):
        shape = layer.shape
        layer_flatten = layer.flatten()
        ret = paillier.encrypt(layer_flatten)
        ret = np.array(ret).reshape(shape)
        layers.append(ret)
    return layers

def paillier_decrypt(value):
    # print(value.shape)
    layers = []
    for idx, layer in enumerate(value):
        shape = layer.shape
        layer_flatten = layer.flatten()
        ret = paillier.decrypt(layer_flatten)
        ret = np.array(ret).reshape(shape)
        layers.append(ret)
    return layers




def setup_logger(filename='test.log'):
    ## setup logger
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w',encoding='utf-8')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)
    
if __name__ == '__main__':
    
    seed = 123
    np.random.seed(seed)
    resume = None
    debug = False
    # resume = "./checkpoint\\encrypt_alexnet_cifar10\\best_accuracy.h5"
    # num_epochs = 200
    num_epochs = 400
    num_clients = 10
    # experiment = "plain"
    # experiment = "encrypt"
    # experiment = "batch"
    # experiment = "quantize"
    experiment = "hash_encrypt"
    # experiment = "flatten_hash_encrypt"
    # experiment = "batch_hash_encrypt"
    parallel_mode = "pool" # pool queue pool_ "" 
    
    element_bits = 16  # number of bits per element in a client's data
    # To account for the worst-case overflow encountered during aggregation
    # additional_bits = int(np.ceil(np.log2(num_clients + 1)))
    additional_bits = int(np.ceil(np.log2(num_clients + 1)))
    # actual_element_bits = element_bits 
    init_bits = element_bits + additional_bits
    # 批处理加密时用 
    batch_size = 10  # 5个
    
    if  experiment == "encrypt":
        flashe = FlasheCipher(init_bits)
        flashe.set_num_clients(num_clients)
        flashe.generate_prp_seed()
        flashe.set_iter_index(0)
        flashe.idx = 0      
    elif experiment == "batch":
        paillier = PaillierCipher() 
        batch_bits = init_bits * batch_size
        paillier.generate_key(n_length=batch_bits)
    elif experiment == "hash_encrypt" or experiment=="flatten_hash_encrypt":
        flashe = FlasheCipher_Hash(init_bits)
        flashe.set_num_clients(num_clients)
        flashe.set_hash_algorithm("sha256")
        flashe.set_iter_index(0)
        flashe.idx = 0   
    elif experiment == "batch_hash_encrypt":
        batch_bits = init_bits * batch_size
        flashe = FlasheCipher_Hash(batch_bits)
        flashe.set_num_clients(num_clients)
        flashe.set_hash_algorithm("sha256")
        flashe.set_iter_index(0)
        flashe.idx = 0   
        
    # print('%s_test.log'%(experiment))
    if not debug:
        FILE_NAME = os.path.basename(__file__).split(".")[0]
        ROOT_PATH = os.getcwd()
        SAVE_NAME = f"%s_%s_b{init_bits}_e{num_epochs}"%(experiment,FILE_NAME)
        
        dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
        CHECKPOINT_PATH = "./checkpoint"+f"/{dayTime}" 
        RESULT_PATH = "./run"+f"/{dayTime}"
        LOG_PATH = "./log"+f"/{dayTime}"
        
        # dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
        
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        # print(Path(CHECKPOINT_PATH)/SAVE_NAME)
        mkdir = True
        CHECKPOINT_PATH = increment_path(path=Path(CHECKPOINT_PATH)/SAVE_NAME,mkdir=mkdir)
        SAVE_RESULT_PATH = increment_path(Path(RESULT_PATH)/SAVE_NAME,mkdir=mkdir)
        LOG_PATH = increment_path(Path(LOG_PATH)/SAVE_NAME,mkdir=mkdir)
        # print(CHECKPOINT_PATH)
        SAVE_LOG_PATH = f'{LOG_PATH}/%s.log'%(SAVE_NAME)
        SAVE_RESULT_TIME_PATH = f'{SAVE_RESULT_PATH}/time.csv'
        SAVE_RESULT_PER_PATH = f'{SAVE_RESULT_PATH}/performance.csv'

        setup_logger(filename=SAVE_LOG_PATH)
        # 裁剪-->量化-->加密
        effiency_col_name = ["step","train_loss","train_accuracy","test_loss","test_accuracy"]
        # if experiment=="encrypt" or experiment=="batch" or experiment=="hash_encrypt" or experiment=="flatten_hash_encrypt" or:
            
        if experiment=="plain":
            time_col_name = ["step","train_time","aggregate_time","elapsed time"]
            # effiency_col_name = ["step","train_loss","train_accuracy","test_loss","test_accuracy"]
        elif experiment=="quantize":
            time_col_name = ["step","train_time","quantize_time","aggregate_time","dequantize_time","elasp_time"]
            # effiency_col_name = ["step","train_loss","train_accuracy","test_loss","test_accuracy"]
        else:
            time_col_name = ["step","train_time","quantize_time","encrypt_time","aggregate_time","decrypt_time","dequantize_time","elasp_time"]
        with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
            writer = csv.writer(f)
            writer.writerow(time_col_name)
        with open(SAVE_RESULT_PER_PATH,'a',encoding='utf8',newline='') as f :
            writer = csv.writer(f)
            writer.writerow(effiency_col_name)
    if resume is not None:
        model.load_weights(resume)
        pass 
    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    test_loss_results = []
    test_accuracy_results = []
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        epoch_train_loss_avg = tf.keras.metrics.Mean()
        epoch_train_accuracy = tf.keras.metrics.Accuracy()
        # flashe.set_iter_index(epoch)
        # 为每个客户端分配数据
        train_dataset_clients = build_datasets(num_clients)
        count = 0
        # 分批进入
        total_cipher_size = 0
        total_plain_size = 0
        
        total_size = {}
        total_size['Plaintext'] = 0
        for data_clients in zip(*train_dataset_clients):
            count += 1
            logging.info("\n---------epoch {} batch{},{} clients are in federated training\n".format(epoch,count,len(data_clients)))
            logging.info("mode = {} parallel_mode = {}".format(experiment,parallel_mode))
            loss_batch_clients = []
            grads_batch_clients = []
            start_train = time.time()
            # 求解梯度
            for x, y in data_clients:
                loss_temp, grads_temp = grad(model, x, y)
                loss_batch_clients.append(loss_temp.numpy())
                grads_batch_clients.append([x.numpy() for x in grads_temp])
            end_train = time.time() - start_train
            logging.info("train_time:%.5f"%(end_train))
            

            
            grads_plain = aggregate_gradients(grads_batch_clients)
            plaintxt_byte = pickle.dumps(grads_batch_clients)
            plain_size = len(plaintxt_byte)
            total_size['Plaintext'] += plain_size
            
            
            # grads_plain = aggregate_gradients(grads_batch_clients)
            start = time.time()
            if experiment == "plain":
                start_aggr = time.time()
                # 每个client的批梯度
                grads = aggregate_gradients(grads_batch_clients)
               
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start
                logging.info("aggregation finished in %f" % (end_aggr))
                
            elif experiment == "encrypt":
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start
                logging.info("量化时间：%f" % (end_quan))
                
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                grads_batch_clients_enc = [flashe_encrypt(item) for item in grads_batch_clients_quan]
                end_enc = time.time() - start_enc
                logging.info("加密梯度时间：%f" % (end_enc))

                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_enc)
                client_weight = 1.0 / num_clients
                
                # 聚集损失
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start_aggr
                logging.info("聚集时间：%f" % (end_aggr))
                          
                start_de = time.time()
                grads_de = flashe_decrypt(grads_aggre)
                end_de = time.time() - start_de
                logging.info("解密时间：%f" % (end_de))

                start_unqu = time.time()
                # 解量化
                grads = unquantize(grads_de, r_maxs, element_bits, num_clients)
                end_unqu = time.time() - start_unqu
                logging.info("去量化时间：%f" % (end_unqu))

                # get the cipher size
                ciphertxt_byte = pickle.dumps(grads_batch_clients_enc)
                cipher_size = len(ciphertxt_byte)
                total_cipher_size += cipher_size 
            elif experiment == "batch":
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start
                logging.info("量化时间：%f" % (end_quan ))
                grads_batch_clients_quan,og_shapes = batch(grads_batch_clients_quan,batch_bits,element_bits,additional_bits,parallel_mode)
                logging.info("批处理时间：%f" % (time.time() - end_quan - start))
                
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                grads_batch_clients_enc = [paillier_encrypt(item) for item in grads_batch_clients_quan]
                end_enc = time.time() - start_enc
                logging.info("加密梯度时间：%f" % (end_enc))

                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_enc)
                client_weight = 1.0 / num_clients
                
                
                
                # 聚集损失
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start_aggr
                logging.info("聚集时间：%f" % (end_aggr))
                
                   
                start_de = time.time()
                grads_de = paillier_decrypt(grads_aggre)
                end_de = time.time() - start_de
                logging.info("解密时间：%f" % (end_de))

                start_unqu = time.time()
                # 解量化
                batch_grads = unbatch(grads_de,og_shapes[0],batch_bits,element_bits,additional_bits)
                grads = unquantize(batch_grads, r_maxs, element_bits, num_clients)
                
                end_unqu = time.time() - start_unqu
                logging.info("去量化时间：%f" % (end_unqu))

                
                # get the cipher size
                ciphertxt_byte = pickle.dumps(grads_batch_clients_enc)
                cipher_size = len(ciphertxt_byte)
                total_cipher_size += cipher_size 
            elif experiment == "quantize":
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start
                logging.info("量化时间：%f" % (end_quan))
                
                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_quan)
                client_weight = 1.0 / num_clients
                
                # 聚集损失
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start_aggr
                logging.info("聚集时间：%f" % (end_aggr))
                          
                start_unqu = time.time()
                # 解量化
                grads = unquantize(grads_aggre, r_maxs, element_bits, num_clients)
                end_unqu = time.time() - start_unqu
                logging.info("去量化时间：%f" % (end_unqu))
            elif experiment == "hash_encrypt":
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start
                logging.info("量化时间：%f" % (end_quan))
                print(grads_batch_clients_quan.shape)
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                # grads_batch_clients_enc = [flashe_encrypt(item) for index,item in enumerate(grads_batch_clients_quan)]
                grads_batch_clients_enc = [flashe_encrypt(item) for item in grads_batch_clients_quan]
                # grads_batch_clients_enc = [flashe_encrypt(item,first_random_value[index]) for index,item in enumerate(grads_batch_clients_quan)]
                end_enc = time.time() - start_enc
                logging.info("加密梯度时间：%f" % (end_enc))

                
                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_enc)
                client_weight = 1.0 / num_clients
                
                eval_traffic("全局梯度大小:",grads_aggre)
                
                # 聚集损失
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start_aggr
                logging.info("聚集时间：%f" % (end_aggr))
                          
                start_de = time.time()
                grads_de = flashe_decrypt(grads_aggre)
                end_de = time.time() - start_de
                logging.info("解密时间：%f" % (end_de))

                start_unqu = time.time()
                # 解量化
                grads = unquantize(grads_de, r_maxs, element_bits, num_clients)
                end_unqu = time.time() - start_unqu
                logging.info("去量化时间：%f" % (end_unqu))

                # get the cipher size
                ciphertxt_byte = pickle.dumps(grads_batch_clients_enc)
                cipher_size = len(ciphertxt_byte)
                total_cipher_size += cipher_size 
                
            elif experiment == "flatten_hash_encrypt":
                start_qu = time.time()
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs,og_shapes = flatten_quantize(grads_batch_clients,element_bits,alpha_list)
                
                # grads_batch_clients_quan,r_maxs,og_shapes = flatten_quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start_qu
                logging.info("量化时间：%f" % (end_quan))
                # batch_quantime.append(end_quan)
                
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                grads_batch_clients_enc = [flashe_encrypt(item) for item in grads_batch_clients_quan]
                end_enc = time.time() - start_enc
                logging.info("加密梯度时间：%f" % (end_enc))
                # batch_enctime.append(end_enc)

                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_enc)
                # grads_aggre = aggregate_gradients(grads_batch_clients_quan)
                client_weight = 1.0 / num_clients
                
                # 聚集损失
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start_aggr
                logging.info("聚集时间：%f" % (end_aggr))
                # batch_aggrtime.append(end_aggr)
                
                
                start_de = time.time()
                grads_de = flashe_decrypt(grads_aggre)
                end_de = time.time() - start_de
                logging.info("解密时间：%f" % (end_de))
                # batch_detime.append(end_de)

                start_unqu = time.time()
                # 解量化
                # grads_unqu = unquantize(grads_de, r_maxs, actual_element_bits, num_clients)
                # grads = unquantize(grads_aggre, r_maxs, actual_element_bits, num_clients)
                # print(grads_de)
                grads = flatten_unquantize(grads_de, r_maxs, element_bits, num_clients,og_shapes[0])
                # grads = unquantize(grads_de, r_maxs, element_bits, num_clients)
                end_unqu = time.time() - start_unqu
                logging.info("去量化时间：%f" % (end_unqu))
                # batch_unqutime.append(end_unqu)

                # get the cipher size
                ciphertxt_byte = pickle.dumps(grads_batch_clients_enc)
                cipher_size = len(ciphertxt_byte)
                total_cipher_size += cipher_size              
            elif experiment == "batch_hash_encrypt":
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() 
                logging.info("量化时间：%f" % (end_quan- start))
                grads_batch_clients_quan,og_shapes = batch(grads_batch_clients_quan,batch_bits,element_bits,additional_bits,parallel_mode)
                logging.info("批处理时间：%f" % (time.time()-end_quan))

                
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                grads_batch_clients_enc = [flashe_encrypt(item) for item in grads_batch_clients_quan]
                # 
                end_enc = time.time() - start_enc
                logging.info("加密梯度时间：%f" % (end_enc))

                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_enc)
                client_weight = 1.0 / num_clients
                
                # 聚集损失
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start_aggr
                logging.info("聚集时间：%f" % (end_aggr))
                          
                start_de = time.time()
                grads_de = flashe_decrypt(grads_aggre)
                end_de = time.time() - start_de
                logging.info("解密时间：%f" % (end_de))

                start_unbatch = time.time()
                # 解量化
                grads_de = unbatch(grads_de,og_shapes[0],batch_bits,element_bits,additional_bits)
                end_unbatch = time.time()-start_unbatch
                logging.info("去批处理时间：%f" % (end_unbatch))
                start_unqu = time.time()
                grads = unquantize(grads_de, r_maxs, element_bits, num_clients)
                end_unqu = time.time() - start_unqu
                logging.info("去量化时间：%f" % (end_unqu))
            elapsed_time = time.time() - start
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            
            logging.info('loss: {} \t elapsed time: {}'.format(loss_value, elapsed_time))
            
            # Track progress
            epoch_train_loss_avg.update_state(loss_value)
            epoch_train_accuracy.update_state(
                tf.concat([data_item[1] for data_item in data_clients], axis=0),
                tf.argmax(model(tf.concat([data_item[0] for data_item in data_clients], axis=0)), axis=1,
                          output_type=tf.int32)
            )
            if not debug:
                # if experiment=="encrypt" or experiment=="batch" or experiment=="hash_encrypt" or experiment=="flatten_hash_encrypt":
                    # time_writer.writerow([f"{epoch}_{count}",end_train,end_quan, end_enc, end_aggr,end_de,end_unqu,elapsed_time])

                if experiment == "plain":
                    # time_writer.writerow([f"{epoch}_{count}",end_train,end_aggr,elapsed_time])
                    with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
                        writer = csv.writer(f)
                        writer.writerow([f"{epoch}_{count}",end_train,end_aggr,elapsed_time])
                elif experiment == "quantize":
                    with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
                        writer = csv.writer(f)
                        writer.writerow([f"{epoch}_{count}",end_train,end_quan,end_aggr,end_unqu,elapsed_time])
                else:
                    with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
                        writer = csv.writer(f)
                        writer.writerow([f"{epoch}_{count}",end_train,end_quan, end_enc, end_aggr,end_de,end_unqu,elapsed_time])
        
            if experiment=="encrypt" or experiment=="batch" or experiment=="hash_encrypt" or experiment=="flatten_hash_encrypt" or experiment=="batch_hash_encrypt":
                total_size['Ciphertext'] = total_cipher_size
                for text in ['Plaintext', 'Ciphertext']:
                    kb = total_size[text] / 1024
                    if kb >= 1024:
                        mb = kb / 1024
                        if mb > 1024:
                            gb = mb / 1024
                            logging.info('{}_size {:.2f} GB'.format(text,gb))
                        else:
                            logging.info('{}_size {:.2f} MB'.format(text,mb))
                    else:
                            logging.info('{}_size {:.2f} KB'.format(text,kb))
            
        # end epoch
        # 记录训练时
        train_loss_v = epoch_train_loss_avg.result()
        train_accuracy_v = epoch_train_accuracy.result()
        # 测试
        test_loss_v = loss(model, x_test, y_test)
        test_accuracy_v = accuracy_score(y_test, tf.argmax(model(x_test), axis=1, output_type=tf.int32))
        # per_writer.writerow([epoch,train_loss_v,train_accuracy_v,test_loss_v,test_accuracy_v])
        if not debug:
            with open(SAVE_RESULT_PER_PATH,'a',encoding='utf8',newline='') as f :
                writer = csv.writer(f)
                writer.writerow([epoch,float(train_loss_v),float(train_accuracy_v),float(test_loss_v),float(test_accuracy_v)])
        # 记录
        logging.info("========Epoch {:03d}: train_loss: {:.3f}, train_accuracy: {:.3%}, test_loss: {:.3f}, test_accuracy: {:.3%}".format(
                epoch,
                train_loss_v,
                train_accuracy_v,
                test_loss_v,
                test_accuracy_v))
                # 记录
        train_loss_results.append(train_loss_v)
        train_accuracy_results.append(train_accuracy_v)
        test_loss_results.append(test_loss_v)
        test_accuracy_results.append(test_accuracy_v)
        # serialize weights to HDF5
        # 保存中间文件
        if not debug:
            save_epoch_model_path = os.path.join(CHECKPOINT_PATH,SAVE_NAME)
            if not os.path.exists(save_epoch_model_path):
                os.makedirs(save_epoch_model_path)
            if test_accuracy_v > best_accuracy:
                best_accuracy = test_accuracy_v
                save_model_path = os.path.join(save_epoch_model_path,"best_accuracy.h5")
                model.save_weights(save_model_path)
                save_model_message = os.path.join(save_epoch_model_path,"best_model_message.txt")
                with open(save_model_message,'w') as f:
                        f.write("best_epoch{}_accuracy{}_loss{}".format(epoch,best_accuracy,test_loss_v))
                # np.savetxt(save_model_message,np.array([]))
            else:
                save_model_path = os.path.join(save_epoch_model_path,"{}_b{:02d}_e{:03d}.h5".format(SAVE_NAME,element_bits,epoch))
                model.save_weights(save_model_path)
            logging.info("Saved epoch_model to disk")

    if not debug:
        save_result_path = os.path.join(SAVE_RESULT_PATH,SAVE_NAME) 
        np.save(save_result_path,
                {"train_loss":train_loss_results,
                "train_accuracy":train_accuracy_results,
                "test_loss":test_loss_results,
                "test_accuracy":test_accuracy_results
            })
        # 保存最终模型结构
        model_json = model.to_json()
        save_model_arc_path = os.path.join(SAVE_RESULT_PATH,"model_arc_{}.json".format(SAVE_NAME))
        with open(save_model_arc_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        save_model_weight_path = os.path.join(SAVE_RESULT_PATH,"model_weight_{}.h5".format(SAVE_NAME))
        model.save_weights(save_model_weight_path)
        
        save_all_model_path = os.path.join(SAVE_RESULT_PATH,"model_{}.h5".format(SAVE_NAME))
        model.save(save_all_model_path)
        logging.info("Saved model to disk")
# if not debug:
#     res = np.load(save_result_path+".npy",allow_pickle=True).item()
#     train_loss_results = res["train_loss"]
#     train_accuracy_results = res["train_accuracy"]
#     fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))
#     fig.suptitle('Training Metrics')

#     axes[0].set_ylabel("Train_Loss", fontsize=14)
#     axes[0].plot(train_loss_results)

#     axes[1].set_ylabel("Train_Accuracy", fontsize=14)
#     # axes[1].set_xlabel("Epoch", fontsize=14)
#     axes[1].plot(train_accuracy_results)

#     axes[2].set_ylabel("Test_Loss", fontsize=14)
#     axes[2].plot(test_loss_results)

#     axes[3].set_ylabel("Test_Accuracy", fontsize=14)
#     axes[3].set_xlabel("Epoch", fontsize=14)
#     axes[3].plot(test_accuracy_results)
#     plt.show()