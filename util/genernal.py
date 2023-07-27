import logging
from pathlib import Path
import glob
import pickle
import re
from scipy import ndimage
from skimage import measure
import numpy as np
import tensorflow as tf

def increment_path(path, exist_ok=False, sep='_', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        print(suffix)
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        # print("test")
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path
def convert_binary_image(image,threshold=0.5):
    image = np.array(image)
    image[image >= threshold ] = 1
    image[image < threshold ] = 0
    return tf.convert_to_tensor(image)
    
# JA = TP / (TP + FP + FN)  
# DI = 2 * TP / (2 * TP + FP + FN)
# AC = (TP + TN) / (TP + FP + TN + FN)
# SE = TP / (TP + FN)
# SP = TN / (FP + TN)

def evaluate(y_pre, y):
    # x 和 target 都是二值化的图像
    JA_sum, AC_sum, DI_sum, SE_sum, SP_sum = [], [], [], [], []
    size = y_pre.shape[0]
    if not isinstance(y_pre,np.ndarray):
        y_pre = np.squeeze(y_pre.numpy(),axis=3)
    if not isinstance(y,np.ndarray):    
        y= np.squeeze(y.numpy(),axis=3)
    y_pre[y_pre >= 0.5] = 1
    y_pre[y_pre < 0.5] = 0
    for i in range(y_pre.shape[0]):
        y_true = y[i]
        y_pred = y_pre[i]
        JA = jaccard_index(y_true, y_pred)
        AC = accuracy(y_true, y_pred)
        DI = dice_coefficient(y_true, y_pred)
        SE = sensitivity(y_true, y_pred)
        SP = specificity(y_true, y_pred)
        JA_sum.append(JA); AC_sum.append(AC); DI_sum.append(DI); SE_sum.append(SE); SP_sum.append(SP)
    # print(sum(SP_sum))
    # print(size)
    return sum(JA_sum)/size, sum(AC_sum)/size, sum(DI_sum)/size, sum(SE_sum)/size, sum(SP_sum)/size

def eval_traffic(text,value):
    kb = pickle.dumps(value)
    kb = len(kb)
    if kb >= 1024:
        mb = kb / 1024
        if mb > 1024:
            gb = mb / 1024
            logging.info('{}_size {:.2f} GB'.format(text,gb))
        else:
            logging.info('{}_size {:.2f} MB'.format(text,mb))
    else:
            logging.info('{}_size {:.2f} KB'.format(text,kb))     
# def evaluate(x, target):
#     # get data batch size
#     size = x.shape[0]
#     # print(size)
#     if not isinstance(x,np.ndarray):
#         x = np.squeeze(x.numpy(),axis=3)
#     if not isinstance(target,np.ndarray):  
#         target =np.squeeze(target.numpy(),axis=3)  
         
#     # x, target = 预测值和真实值
#     JA_sum, AC_sum, DI_sum, SE_sum, SP_sum = [], [], [], [], []
#     x_tmp = x
#     target_tmp = target

    
#     x_tmp[x_tmp >= 0.5] = 1
#     x_tmp[x_tmp < 0.5] = 0
#     x_tmp = np.array(x_tmp, dtype='uint8')
#     x_tmp = ndimage.binary_fill_holes(x_tmp).astype(int)

#     # only reserve largest connected component.
#     box = []
#     [lesion, num] = measure.label(x_tmp, return_num=True)
#     if num == 0:
#         JA_sum.append(0)
#         AC_sum.append(0)
#         DI_sum.append(0)
#         SE_sum.append(0)
#         SP_sum.append(0)
#         logging.info("无连通图")
#     else:
#         \region = measure.regionprops(lesion)
#         for i in range(num):
#             box.append(region[i].area)
#         label_num = box.index(max(box)) + 1
#         lesion[lesion != label_num] = 0
#         lesion[lesion == label_num] = 1

#         #  calculate TP,TN,FP,FN
#         TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 1)))
#         # True Negative (TN): we predict1 a label of 0 (negative), and the true label is 0.
#         TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

#         # False Positive (FP): we predict1 a label of 1 (positive), but the true label is 0.
#         FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

#         # False Negative (FN): we predict1 a label of 0 (negative), but the true label is 1.
#         FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 1)))

#         #  calculate JA, Dice, SE, SP
#         JA = TP / ((TP + FN + FP + 1e-7))
#         AC = (TP + TN) / ((TP + FP + TN + FN + 1e-7))
#         DI = 2 * TP / ((2 * TP + FN + FP + 1e-7))
#         SE = TP / (TP + FN+1e-7)
#         SP = TN / ((TN + FP+1e-7))

#         JA_sum.append(JA); AC_sum.append(AC); DI_sum.append(DI); SE_sum.append(SE); SP_sum.append(SP)

#     return sum(JA_sum)/size, sum(AC_sum)/size, sum(DI_sum)/size, sum(SE_sum)/size, sum(SP_sum)/size

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
    
    

def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def sensitivity(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true, y_pred))
    false_negatives = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    return true_positives / (true_positives + false_negatives)

def specificity(y_true, y_pred):
    true_negatives = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
    false_positives = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    return true_negatives / (true_negatives + false_positives)


# # 定义评估指标
# def jaccard_index(y_true, y_pred):
#     intersection = tf.reduce_sum(y_true * y_pred)
#     union = tf.reduce_sum(y_true + y_pred) - intersection
#     return tf.reduce_mean(intersection / union).numpy()

# def dice_coefficient(y_true, y_pred):
#     numerator = 2 * tf.reduce_sum(y_true * y_pred)
#     denominator = tf.reduce_sum(y_true + y_pred)
#     return tf.reduce_mean(numerator / denominator).numpy()

# def accuracy(y_true, y_pred):
#     threshold = 0.5
#     y_pred = tf.cast(y_pred > threshold, tf.float32)
#     return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32)).numpy()

# def sensitivity(y_true, y_pred):
#     true_positive = tf.reduce_sum(y_true * y_pred)
#     actual_positive = tf.reduce_sum(y_true)
#     return tf.reduce_mean(true_positive / actual_positive).numpy()

# def specificity(y_true, y_pred):
#     true_negative = tf.reduce_sum((1 - y_true) * (1 - y_pred))
#     actual_negative = tf.reduce_sum(1 - y_true)
#     return tf.reduce_mean(true_negative / actual_negative).numpy()