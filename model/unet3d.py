import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization,Conv3D,MaxPooling3D,Conv3DTranspose,UpSampling3D
import nibabel as nib
import numpy as np	
class unet_encoder(tf.keras.Model):
    def __init__(self):
        super(unet_encoder,self).__init__()
        self.b1 = BatchNormalization()
        self.conv1 = Conv3D(8,3,activation='relu',padding='same')

        self.b2 = BatchNormalization()
        self.conv2 = Conv3D(16,3,activation='relu',padding='same')

        self.b3 = BatchNormalization()
        self.conv3 = Conv3D(16,(3,3,2),activation='relu')

        self.b4 = BatchNormalization()
        self.conv4 = Conv3D(32,(3,3,1),activation='relu',strides=2)

        self.b5 = BatchNormalization()
        self.conv5 = Conv3D(64,(3,3,1),activation='relu',strides=2)

        self.b6 = BatchNormalization()
        self.maxpool1 = MaxPooling3D((2,2,1))
    
    def call(self,x,features):
        x = self.b1(x)
        x = self.conv1(x)

        x = self.b2(x)
        x = self.conv2(x)

        x = self.b3(x)
        # 第一个连接特征图
        x = self.conv3(x)
        x = self.b4(x)
        features.append(x)
        # 第二个连接特征图
        x = self.conv4(x)
        x = self.b5(x)
        features.append(x)
        # 第三个连接特征图
        x = self.conv5(x)
        x = self.b6(x)
        features.append(x)
        # 输出变量
        outputs = self.maxpool1(x)

        return outputs

class unet_decoder(tf.keras.Model):
    def __init__(self):
        super(unet_decoder,self).__init__()
        self.b1 = BatchNormalization()
        self.up1 = UpSampling3D((2,2,1))
        self.conv1tp = Conv3DTranspose(64,(3,3,1),activation='relu',padding='same')

        self.b2 = BatchNormalization()
        self.up2 = UpSampling3D((2,2,2))
        self.conv2tp = Conv3DTranspose(32,(3,3,1),activation='relu')
        self.conv2 = Conv3D(32,3,activation='relu')

        self.b3 = BatchNormalization()
        self.conv3tp = Conv3DTranspose(16,(3,3,2),activation='relu')

        self.b4 = BatchNormalization()
        self.up4 = UpSampling3D((2,2,2))
        self.conv4tp = Conv3DTranspose(16,(3,3,1),activation='relu')

        self.b5 = BatchNormalization()
        self.conv5tp = Conv3DTranspose(8,(3,3,1),activation='relu')

        self.conv6tp = Conv3DTranspose(4,(1,1,5),activation='relu')
        self.conv_out = Conv3D(1,(1,1,4),activation='relu')

    def call(self,x,features):
        
        x = self.b1(x)
        x = self.up1(x)
        x = self.conv1tp(x)
        x = tf.concat((features[-1],x),axis=-1)

        x = self.b2(x)
        x = self.up2(x)
        x = self.conv2tp(x)
        x = self.conv2(x)

        x = self.b3(x)
        x = self.conv3tp(x)
        x = tf.concat((features[-2],x),axis=-1)

        x = self.b4(x)
        x = self.up4(x)
        x = self.conv4tp(x)
        x = tf.concat((features[-3],x),axis=-1)
        
        x = self.b5(x)
        x = self.conv5tp(x)
        x = self.conv6tp(x)

        x = self.conv_out(x)
        outputs = x
        
        return outputs


class Unet3D(tf.keras.Model):
    def __init__(self,encoder,decoder):
        super(Unet3D,self).__init__()
        self.features = []
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self,x):
        x = self.encoder(x,self.features)
        outputs = self.decoder(x,self.features)
        return outputs

def nearest_4d(img,size):
    res = np.zeros(size)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            for k in range(res.shape[2]):
                idx = i*img.shape[0] // res.shape[0]
                idy = j*img.shape[1] // res.shape[1]
                idz = k*img.shape[2] // res.shape[2]
                res[i,j,k,:] = img[idx,idy,idz,:]
    return res

# 按照数据文件路径以迭代器的方式读取数据
class DataIterator:
    def __init__(self,image_paths,label_paths,size=None,transp_shape=[0,1,2,3],mode='nib'):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.size = size
        self.transp = transp_shape
        self.mode=mode

    def read_and_resize(self,img_path,lbl_path):
        if self.mode=='nib':
            img = nib.load(img_path)
            lbl = nib.load(lbl_path)

            img = img.get_fdata(caching='fill', dtype='float32')
            lbl = lbl.get_fdata(caching='fill', dtype='float32')
            
        elif self.mode == 'np':
            img = np.load(img_path)
            lbl = np.load(lbl_path)
        else:
            return None,None
        
        img /= np.max(img)
        lbl /= np.max(lbl)

        img = img.transpose(self.transp)
        if len(lbl.shape)<len(img.shape):
            lbl = np.expand_dims(lbl,axis=-1)
        lbl = lbl.transpose(self.transp)

        if self.size != None:
            if len(self.size) == 3:
                img = nearest_3d(img,self.size)
                lbl = nearest_3d(lbl,self.size)
            else:
                img = nearest_4d(img,self.size)
                lbl = nearest_4d(lbl,self.size)
        return img,lbl
    
    def __iter__(self):
        for img_path,lbl_path in zip(self.image_paths,self.label_paths):
            img,lbl = self.read_and_resize(img_path,lbl_path)
            if isinstance(img,np.ndarray) and isinstance(lbl,np.ndarray):
                yield (img,lbl)
            else:
                return
# 数据生成器，因为训练用的标签数据少了一个维度，所以在返回数据对象之前给数据对象扩充维度
class DataGenerator:
    def __init__(self,image_paths,label_paths,size=None,batch_size=32,transp_shape=[0,1,2,3],mode='nib'):
        dataiter = DataIterator(image_paths,label_paths,size,transp_shape,mode)
        self.batch_size = batch_size
        self.dataiter = iter(dataiter)
    
    def __iter__(self):
        while 1:
            i = 0
            imgs = []
            lbls = []
            for img,lbl in self.dataiter:
                imgs.append(img)
                lbls.append(lbl)
                i += 1
                if i >= self.batch_size:
                    break
            
            if i == 0:
                break
            imgs = np.stack(imgs)
            lbls = np.stack(lbls)
            if len(imgs.shape) < 5:
                imgs = np.expand_dims(imgs,axis=-1)
                lbls = np.expand_dims(lbls,axis=-1)
            
            yield (imgs,lbls)
