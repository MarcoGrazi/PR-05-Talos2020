import tensorflow as tf
import pandas as pd
import keras
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

'''
this script is the one I used to try different configurations of the Neural Network 
which will tdrive the Talos2020 robot in autonomous mode. The main structure of the network
was blandly inspired by the paper "Neural circuit policies enabling auditable autonomy" published by 
Nature Machine Intelligence. With a Dataset of more than 38000 images and associated data (collected by 
driving the robot in REC mode), the results were not good enough. Still needs research, maybe trying a two network 
(one to classify the right command and one to calculate the right value for that command) approach would reap
better results  
'''


TRAIN_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
              '/Talos 2020/Software/Dataset/Train/'
TEST1_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
            '/Talos 2020/Software/Dataset/Test1/'
TEST2_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
            '/Talos 2020/Software/Dataset/Test2/'
MODEL_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots/Talos 2020/Software/ANIMA/AI Models/'
SEQUENCE_LENGTH = 16


class CustomGenerator(keras.utils.Sequence):
    def __init__(self, sensorsfile, statusfile, controlsfile,
                 imagedir, batchsize, targetsize):
        self.imagedir = imagedir
        self.sensorsfile = sensorsfile
        self.statusfile = statusfile
        self.controlsfile = controlsfile
        self.batchsize = batchsize
        self.targetsize = targetsize

    def __len__(self):
        cont = 0
        for file in os.listdir(self.imagedir):
            if 'image' in file:
                cont += 1
        return int(np.ceil(cont/float(self.batchsize)))

    def __getitem__(self, item):
        names = []
        for file in os.listdir(self.imagedir):
            if 'image' in file:
                names.append(file)
        if (item - 1) <= 0:
            item += 1
        samples = names[(item-1)*self.batchsize: item*self.batchsize]
        x, y = self.Generator(samples, item)
        return x, y

    def Generator(self, samples, item):
        y = np.array(pd.read_csv(self.controlsfile, quotechar='"', header=None))
        xst = np.array(pd.read_csv(self.statusfile, quotechar='"', header=None))
        xsn = np.array(pd.read_csv(self.sensorsfile, quotechar='"', header=None))
        X = []
        if (item-1) <= 0:
            item += 1
        for i in range((item-1)*self.batchsize, item*self.batchsize):
            X.append(np.concatenate((xsn[i], xst[i]), axis=-1))
        X = np.array(X)
        X = X.reshape(1, self.batchsize, 19)
        Y = y[(item-1)*self.batchsize: item * self.batchsize]
        Y = Y.reshape(1, self.batchsize, 16)
        xi = []
        for file in samples:
            im = Image.open(os.path.join(self.imagedir, file))
            im = im.resize(self.targetsize, Image.NEAREST)
            im = np.array(im)
            im = im/float(255)
            im = im.reshape((self.targetsize[0], self.targetsize[1], 3))
            xi.append(im)
        XI = np.array(xi)
        XI = XI.reshape((1, self.batchsize, self.targetsize[0], self.targetsize[1], 3))
        return [np.array(XI), np.array(X)], np.array(Y)

def ActiveRatio(y_true, y_pred):
    zero = tf.constant(0, dtype=tf.float32)
    y_true = tf.reshape(y_true, (1, 16, 16))
    logic_t = tf.not_equal(y_true, zero)
    temp_t = tf.where(logic_t, 1.0, 0.0)
    right = tf.multiply(y_pred, y_true)
    logic_bm = tf.not_equal(right, zero)
    right = tf.where(logic_bm, 1.0, 0.0)
    right = tf.reduce_sum(right)
    total = tf.reduce_sum(temp_t)
    if total == 0:
        if right == 0:
            return 1.0
        else:
            return 0.0
    else:
        return right/total

def LCHCrossentropy(y_true, y_pred):
    zero = tf.constant(0, dtype=tf.float32)
    y_true = tf.reshape(y_true, (16, 16, 1))
    y_pred = tf.reshape(y_pred, (16, 16, 1))
    logic_t = tf.not_equal(y_true, zero)
    T = tf.where(logic_t, 1.0, 0.0)
    Anti_T = tf.where(logic_t, 0.0, 1.0)
    BC = keras.losses.BinaryCrossentropy()
    LCH = keras.losses.LogCosh()
    loss1 = BC(T, y_pred)
    loss2 = LCH(y_true, y_pred)
    loss = loss1*0.5 + loss2*1
    return loss


TrainGen = CustomGenerator(controlsfile=TRAIN_DIR + 'Controls.csv', statusfile=TRAIN_DIR + 'Status.csv',
                           sensorsfile=TRAIN_DIR + 'Sensors.csv', imagedir=TRAIN_DIR, batchsize=SEQUENCE_LENGTH,
                           targetsize=[256, 192])


input1 = keras.layers.Input(shape=[SEQUENCE_LENGTH, 256, 192, 3])
Z = keras.layers.TimeDistributed(
    keras.layers.Conv2D(64, 5, strides=2, padding='same', activation='relu'))(input1)
Z = keras.layers.TimeDistributed(keras.layers.MaxPool2D(4))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Conv2D(32, 5, padding='same', activation='relu'))(Z)
Z = keras.layers.TimeDistributed(keras.layers.MaxPool2D(2))(Z)
Z = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(Z)
Z = keras.layers.TimeDistributed(keras.layers.Dropout(0.4))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Conv2D(32, 3, padding='same', activation='relu'))(Z)
Z = keras.layers.TimeDistributed(keras.layers.MaxPool2D(2))(Z)
Z = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(Z)
Z = keras.layers.TimeDistributed(keras.layers.Dropout(0.4))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Flatten())(Z)
input2 = keras.layers.Input(shape=[SEQUENCE_LENGTH, 19])
Z = keras.layers.Concatenate(axis=2)([Z, input2])
Z = keras.layers.TimeDistributed(keras.layers.Dense(100, 'relu'))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(Z)
Z = keras.layers.SimpleRNN(100, 'relu', return_sequences=True)(Z)
output = keras.layers.TimeDistributed(keras.layers.Dense(16, 'linear'))(Z)


model = keras.Model(inputs=[input1, input2], outputs=[output])
model.compile(optimizer="Nadam", loss=LCHCrossentropy, metrics=['mse', 'mae', 'accuracy', ActiveRatio])

print(model.summary())
train = int(input('train?'))
if train:
    history = model.fit(TrainGen, epochs=1, batch_size=1,
              callbacks=[keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.2, monitor="loss"),
                                               keras.callbacks.EarlyStopping(patience=3, monitor="loss")])

    pd.DataFrame(history.history).plot(figsize=(12, 7))
    plt.grid(True)
    plt.gca().set_ylim(0, 0.05)
    plt.show()
    save = int(input('train save?'))
    if save:
        model.save(MODEL_DIR + 'TrainedModel')

model = keras.models.load_model(MODEL_DIR + 'TrainedModel', custom_objects={'LCHCrossentropy': LCHCrossentropy,
                                                                            'ActiveRatio': ActiveRatio})

Test1Gen = CustomGenerator(controlsfile=TEST1_DIR + 'Controls.csv', statusfile=TEST1_DIR + 'Status.csv',
                          sensorsfile=TEST1_DIR + 'Sensors.csv', imagedir=TEST1_DIR, batchsize=SEQUENCE_LENGTH,
                          targetsize=[256, 192])

Test2Gen = CustomGenerator(controlsfile=TEST2_DIR + 'Controls.csv', statusfile=TEST2_DIR + 'Status.csv',
                          sensorsfile=TEST2_DIR + 'Sensors.csv', imagedir=TEST2_DIR, batchsize=SEQUENCE_LENGTH,
                          targetsize=[256, 192])

score1 = model.evaluate(Test1Gen)
score2 = model.evaluate(Test2Gen)
print('Test1: '+str(score1) + ', Test2: '+str(score2))
for i in range(27):
    input1, y = Test1Gen.__getitem__(i)
    print('prediction: ', model.predict(input1)[0][15], ',\nlabel: ', y[0][15])

save = int(input('save?'))
cont = 0
if save:
    for file in os.listdir(MODEL_DIR):
        if 'model' in file:
            cont += 1
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.target_spec.supported_types = [tf.float16]
    model = converter.convert()
    open(MODEL_DIR+'model'+str(cont)+'.tflite', 'wb').write(model)

model = tf.lite.Interpreter(MODEL_DIR+'model'+str(cont)+'.tflite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
for i in range(0, 27):
    X, _ = Test2Gen.__getitem__(i)
    input_1 = np.array(X[0], dtype=np.float32)
    input_2 = np.array(X[1], dtype=np.float32)
    model.set_tensor(input_details[0]['index'], input_1)
    model.set_tensor(input_details[1]['index'], input_2)
    model.invoke()
    print(model.get_tensor(output_details[0]['index'])[0][15])
