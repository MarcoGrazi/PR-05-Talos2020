import tensorflow
import pandas as pd
import keras
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
              '/Talos 2020/Software/Dataset/Train/'
TEST1_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
            '/Talos 2020/Software/Dataset/Test1'
TEST2_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
            '/Talos 2020/Software/Dataset/Test2'
MODEL_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots/Talos 2020/Software/ANIMA/AI Models'
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
        y = pd.read_csv(self.controlsfile, quotechar='"', header=None)
        y = np.array(y)
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
        Y = Y.reshape(1, self.batchsize, 14)
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

TrainGen = CustomGenerator(controlsfile=TRAIN_DIR + 'Controls.csv', statusfile=TRAIN_DIR + 'Status.csv',
                           sensorsfile=TRAIN_DIR + 'Sensors.csv', imagedir=TRAIN_DIR, batchsize=SEQUENCE_LENGTH,
                           targetsize=[512, 384])

input1 = keras.layers.Input(shape=[SEQUENCE_LENGTH, 512, 384, 3])
Z = keras.layers.TimeDistributed(
    keras.layers.Conv2D(20, 5, strides=2, padding='same', activation='relu'))(input1)
Z = keras.layers.TimeDistributed(keras.layers.MaxPool2D(4))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Conv2D(26, 5, padding='same', activation='relu'))(Z)
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
Z = keras.layers.TimeDistributed(keras.layers.Dense(20, 'relu'))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(Z)
Z = keras.layers.SimpleRNN(25, 'relu', return_sequences=True)(Z)
output = keras.layers.TimeDistributed(keras.layers.Dense(14, 'relu'))(Z)


model = keras.Model(inputs=[input1, input2], outputs=[output])
model.compile(optimizer="Nadam", loss='mae', metrics=["accuracy"])

print(model.summary())
history = model.fit(TrainGen, epochs=15, batch_size=1,
          callbacks=[keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.2, monitor="loss"),
                                           keras.callbacks.EarlyStopping(patience=2, monitor="loss")])

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3)
plt.show()

Test1Gen = CustomGenerator(controlsfile=TEST1_DIR + 'Controls.csv', statusfile=TEST1_DIR + 'Status.csv',
                          sensorsfile=TEST1_DIR + 'Sensors.csv', imagedir=TEST1_DIR, batchsize=SEQUENCE_LENGTH,
                          targetsize=[512, 384])

Test2Gen = CustomGenerator(controlsfile=TEST2_DIR + 'Controls.csv', statusfile=TEST2_DIR + 'Status.csv',
                          sensorsfile=TEST2_DIR + 'Sensors.csv', imagedir=TEST2_DIR, batchsize=SEQUENCE_LENGTH,
                          targetsize=[512, 384])

score1 = model.evaluate(Test1Gen, batch_size=1)
score2 = model.evaluate(Test2Gen, batch_size=1)
print('Test1: '+score1 + ', Test2: '+score2)

save = input('save?')
if save:
    cont=0
    for file in os.listdir(MODEL_DIR):
        if 'model' in file:
            cont+=1
    model.save(MODEL_DIR+'model'+str(cont))
