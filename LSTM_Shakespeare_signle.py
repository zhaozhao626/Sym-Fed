import csv
from pathlib import Path
import time
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import copy 
from functools import reduce, partial
# tf.enable_eager_execution()
# from tensorflow import contrib
import multiprocessing
import sys
from encrypt.ASHE_hash_signle import FlasheCipher_Hash
from encrypt.paillier import PaillierCipher
# from encrypt.ASHE_hash import FlasheCipher_Hash
from encrypt.quantize import \
    batch, unbatch,quantize,unquantize,get_alpha_r_max
from util.genernal import eval_traffic, increment_path, setup_logger
sys.path.append("..")
import logging
import os
import pickle 
# experiment = "encrypt"
num_epochs = 50
num_clients = 10
# experiment = "plain"
# experiment = "encrypt"
# experiment = "batch"
# experiment = "quantize"
experiment = "hash_encrypt"
# experiment = "flatten_hash_encrypt"
# experiment = "batch_hash_encrypt"
element_bits = 16 
additional_bits = int(np.ceil(np.log2(num_clients + 1)))
# actual_element_bits = element_bits 
init_bits = element_bits + additional_bits
logging.info(f"num_epochs:{num_epochs}  init_bits:{init_bits}")

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

FILE_NAME = os.path.basename(__file__).split(".")[0]
ROOT_PATH = os.getcwd()  

SAVE_NAME = f"%s_%s_b{init_bits}_e{num_epochs}"%(experiment,FILE_NAME)

CHECKPOINT_PATH = "./checkpoint" 
RESULT_PATH = "./run"
LOG_PATH = "./log"

mkdir = True
CHECKPOINT_PATH = increment_path(path=Path(CHECKPOINT_PATH)/SAVE_NAME,mkdir=mkdir)
SAVE_RESULT_PATH = increment_path(Path(RESULT_PATH)/SAVE_NAME,mkdir=mkdir)
LOG_PATH = increment_path(Path(LOG_PATH)/SAVE_NAME,mkdir=mkdir)

SAVE_RESULT_PER_PATH = f'{SAVE_RESULT_PATH}/performance.csv'
SAVE_RESULT_TIME_PATH = f'{SAVE_RESULT_PATH}/time.csv'
SAVE_LOG_PATH = f'{LOG_PATH}/%s.log'%(SAVE_NAME)
setup_logger(filename=SAVE_LOG_PATH)


N_JOBS = multiprocessing.cpu_count()



path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
logging.info('Length of text: {} characters'.format(len(text)))



# The unique characters in the file
# 统计字频
vocab = sorted(set(text))
logging.info('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
# 为字符编码
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# 文本转化为编码
text_as_int = np.array([char2idx[c] for c in text])

# 打印
logging.info('{')
for char, _ in zip(char2idx, range(20)):
    logging.info('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
logging.info('  ...\n}')

logging.info('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text) // seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    logging.info(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

for item in sequences.take(5):
    logging.info(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset_0 = sequences.map(split_input_target)

for input_example, target_example in dataset_0.take(1):
    logging.info('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    logging.info('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    logging.info("Step {:4d}".format(i))
    logging.info("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    logging.info("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Batch size
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset_0 = dataset_0.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

if tf.test.is_gpu_available():
    rnn = tf.compat.v1.keras.layers.CuDNNGRU
else:
    rnn = partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
model.summary()

for input_example_batch, target_example_batch in dataset_0.take(1):
    example_batch_predictions = model(input_example_batch)
    logging.info(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

logging.info("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
logging.info("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


def build_datasets(num_clients):
    
    dataset_raw = sequences.map(split_input_target)
    train_dataset_clients = [dataset_raw.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
                                for _ in range(num_clients)]

    return train_dataset_clients

def loss(model, x, y):
    y_ = model(x)
    return tf.keras.losses.sparse_categorical_crossentropy(y, y_, from_logits=True)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.optimizers.Adam()
global_step = tf.Variable(0)


def clip_gradients(grads, min_v, max_v):
    results = [tf.clip_by_value(t, min_v, max_v).numpy() for t in grads]
    return results


def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        if isinstance(x1[i], tf.IndexedSlices) and isinstance(x2[i], tf.IndexedSlices):
            results.append(tf.IndexedSlices(values=tf.concat([x1[i].values, x2[i].values], axis=0),
                                            indices=tf.concat([x1[i].indices, x2[i].indices], axis=0),
                                            dense_shape=x1[i].dense_shape))
        else:
            results.append(x1[i] + x2[i])
    return results


def aggregate_gradients(gradient_list, weight=0.5):
    results = reduce(do_sum, gradient_list)
    return np.array(results,dtype=object)


def aggregate_losses(loss_list):
    return np.mean(loss_list)




def sparse_to_dense(gradients):
    result = []
    for layer in gradients:
        if isinstance(layer, tf.IndexedSlices):
            result.append(tf.convert_to_tensor(layer).numpy())
        else:
            result.append(layer.numpy())
    return result

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

if __name__ == '__main__':
    seed = 123
    # tf.random.set_random_seed(seed)
    np.random.seed(seed)
    resume = None
    # 批处理加密时用 
    batch_size = 20  # 5个
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
    elif experiment == "hash_encrypt":
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
    # 模型checkpoint文件 
    save_epoch_model_path = os.path.join(CHECKPOINT_PATH,SAVE_NAME)
    # 保留结果文件
    save_result_path = os.path.join(SAVE_RESULT_PATH,SAVE_NAME) 
    # 保留模型架构
    save_model_arc_path = os.path.join(SAVE_RESULT_PATH,"model_arc_{}.json".format(SAVE_NAME))
    # 保存模型权重文件
    save_model_weight_path = os.path.join(SAVE_RESULT_PATH,"model_weight_{}.h5".format(SAVE_NAME))
    # 保存模型全部内容
    save_all_model_path = os.path.join(SAVE_RESULT_PATH,"model_{}.h5".format(SAVE_NAME))

    effiency_col_name = ["step","train_loss"]
    if experiment=="encrypt" or experiment=="batch" or experiment=="hash_encrypt" or experiment == "batch_hash_encrypt":
        time_col_name = ["step","train_time","quantize_time","encrypt_time","aggregate_time","decrypt_time","dequantize_time","elasp_time","elapsed time"]
    elif experiment=="plain":
        time_col_name = ["step","train_time","aggregate_time","elapsed time"]
    elif experiment=="quantize":
        time_col_name = ["step","train_time","quantize_time","aggregate_time","dequantize_time","elasp_time"]
        

    with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
        writer = csv.writer(f)
        writer.writerow(time_col_name)
    with open(SAVE_RESULT_PER_PATH,'a',encoding='utf8',newline='') as f :
        writer = csv.writer(f)
        writer.writerow(effiency_col_name)
        
    if resume is not None:
        model.load_weights(resume)
        pass 
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()

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
            logging.info("mode = {}".format(experiment))
            logging.info("{} clients are in federated training".format(len(data_clients)))
            
            loss_batch_clients = []
            grads_batch_clients = []
            or_grads_batch_clients = []
            start_train = time.time()
            for x, y in data_clients:
                loss_temp, grads_temps = grad(model, x, y)
                loss_batch_clients.append(loss_temp.numpy())
                or_grads_batch_clients.append(grads_temps)
            end_train = time.time() - start_train
            
            logging.info("train_time:%.5f"%(end_train))
            grads_batch_clients = [sparse_to_dense(item) for item in or_grads_batch_clients]
            
            plaintxt_byte = pickle.dumps(grads_batch_clients)
            plain_size = len(plaintxt_byte)
            total_size['Plaintext'] += plain_size 
            
            start = time.time()
            if experiment == "plain":
                # grads_batch_clients = or_grads_batch_clients
                start_aggr = time.time()
                # 每个client的批梯度
                grads = aggregate_gradients(grads_batch_clients)
               
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])
                end_aggr = time.time() - start_aggr
                logging.info("aggregation finished in %f" % (end_aggr))
            elif experiment == "encrypt":
                start_qu = time.time()
                # grads_batch_clients = [sparse_to_dense(item) for item in or_grads_batch_clients]
                
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start_qu
                logging.info("量化时间：%f" % (end_quan))

                
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                grads_batch_clients_enc = [flashe_encrypt(item) for item in grads_batch_clients_quan]
                end_enc = time.time() - start_enc
                logging.info("加密梯度时间：%f" % (end_enc))


                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_enc)
                # grads_aggre = aggregate_gradients(grads_batch_clients_quan)
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
                start_qu = time.time()                               
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time()
                logging.info("量化时间：%f" % (end_quan - start_qu))
                grads_batch_clients_quan,og_shapes = batch(grads_batch_clients,batch_bits,element_bits,additional_bits)
                
                logging.info("批处理时间：%f" % (time.time()-end_quan))
                
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                grads_batch_clients_enc = [paillier_encrypt(item) for item in grads_batch_clients_quan]
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
                start_qu = time.time()
                # grads_batch_clients = [sparse_to_dense(item) for item in or_grads_batch_clients]
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start_qu
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
                start_qu = time.time()
                # grads_batch_clients = [sparse_to_dense(item) for item in or_grads_batch_clients]
                
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() - start_qu
                logging.info("量化时间：%f" % (end_quan))

                
                start_enc = time.time()
                # 加密梯度 外层选择client 内层选择layer
                grads_batch_clients_enc = [flashe_encrypt(item) for item in grads_batch_clients_quan]
                end_enc = time.time() - start_enc
                logging.info("加密梯度时间：%f" % (end_enc))


                start_aggr = time.time()
                # 聚合梯度
                grads_aggre = aggregate_gradients(grads_batch_clients_enc)
                # grads_aggre = aggregate_gradients(grads_batch_clients_quan)
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
                
            elif experiment == "batch_hash_encrypt":
                # grads_batch_clients = [sparse_to_dense(item) for item in or_grads_batch_clients]
                alpha_list, r_max_list = get_alpha_r_max(grads_batch_clients,element_bits,num_clients)
                grads_batch_clients_quan,r_maxs = quantize(grads_batch_clients,element_bits,alpha_list)
                end_quan = time.time() 
                logging.info("量化时间：%f" % (end_quan- start))
                grads_batch_clients_quan,og_shapes = batch(grads_batch_clients_quan,batch_bits,element_bits,additional_bits)
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

                start_unqu = time.time()
                # 解量化
                grads_de = unbatch(grads_de,og_shapes[0],batch_bits,element_bits,additional_bits)
                grads = unquantize(grads_de, r_maxs, element_bits, num_clients)
                end_unqu = time.time() - start_unqu
                logging.info("去量化时间：%f" % (end_unqu))
                
                # get the cipher size
                ciphertxt_byte = pickle.dumps(grads_batch_clients_enc)
                cipher_size = len(ciphertxt_byte)
                total_cipher_size += cipher_size 
                
            elapsed_time = time.time() - start
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            
            logging.info( "loss: {} \telapsed time: {}".format(loss_value, elapsed_time) )
            if experiment == "encrypt" or experiment == "batch" or experiment=="hash_encrypt" or experiment == "batch_hash_encrypt":
                # time_writer.writerow([f"{epoch}_{count}",end_train,end_quan, end_enc, end_aggr,end_de,end_unqu,elapsed_time])
                with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
                    writer = csv.writer(f)
                    writer.writerow([f"{epoch}_{count}",end_train,end_quan, end_enc, end_aggr,end_de,end_unqu,elapsed_time])
            elif experiment == "plain":
                # time_writer.writerow([f"{epoch}_{count}",end_train,end_aggr,elapsed_time])
                with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
                    writer = csv.writer(f)
                    writer.writerow([f"{epoch}_{count}",end_train,end_aggr,elapsed_time])
            elif experiment == "quantize":
                with open(SAVE_RESULT_TIME_PATH,'a',encoding='utf8',newline='') as f :
                    writer = csv.writer(f)
                    writer.writerow([f"{epoch}_{count}",end_train,end_quan,end_aggr,end_unqu,elapsed_time])
            if experiment == "encrypt" or experiment == "batch" or experiment=="hash_encrypt" or experiment == "batch_hash_encrypt":
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
        
        train_loss_results.append(epoch_loss_avg.result())
        logging.info("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
        print("---"*20)
        # 保存性能指标
        with open(SAVE_RESULT_PER_PATH,'a',encoding='utf8',newline='') as f :
            writer = csv.writer(f)
            writer.writerow([epoch,float(epoch_loss_avg.result())])
            
        if not os.path.exists(save_epoch_model_path):
            os.makedirs(save_epoch_model_path)
        save_model_path = os.path.join(save_epoch_model_path,"{}_b{:02d}_e{:03d}.h5".format(SAVE_NAME,element_bits,epoch))
        model.save_weights(save_model_path)
        logging.info("Saved epoch_model to disk")
   
        np.save(save_result_path,
            {"train_loss":train_loss_results,
        })
        # 保存最终模型结构
        model_json = model.to_json()

        with open(save_model_arc_path, "w") as json_file:
            json_file.write(model_json)
            
        model.save_weights(save_model_weight_path)
        model.save(save_all_model_path)
        logging.info("Saved model to disk")

    # diskfig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    # fig.suptitle('Training Metrics')
    
    # axes[0].set_ylabel("Loss", fontsize=14)
    # axes[0].plot(loss_array)
    
    # axes[1].set_ylabel("Accuracy", fontsize=14)
    # axes[1].set_xlabel("Batch", fontsize=14)
    # axes[1].plot(accuracy_array)
    # plt.show()
