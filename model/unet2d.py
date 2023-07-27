from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
# 定义u-Net网络模型
def Unet():
    # contraction path
    # 输入层数据为256*256的三通道图像
    inputs = Input(shape=[256, 256, 3])
    # 第一个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # 最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 第二个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 第三个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 第四个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # 将部分隐藏层神经元丢弃，防止过于细化而引起的过拟合情况
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # 将部分隐藏层神经元丢弃，防止过于细化而引起的过拟合情况
    drop5 = Dropout(0.5)(conv5)

    # expansive path
    # 上采样
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    # copy and crop(和contraction path 的feature map合并拼接)
    merge6 = concatenate([drop4, up6], axis=3)
    # 两个有效卷积层
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    # 上采样
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # 上采样
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # 上采样
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    
    # # 优化器为 Adam,损失函数为 binary_crossentropy，评价函数为 accuracy
    # model.compile(optimizer=Adam(lr=1e-4),
    #              loss='binary_crossentropy',
    #              metrics=['accuracy'])
    return model
