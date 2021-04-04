import tensorflow as tf
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
            '/Talos 2020/Software/Dataset/Test1/'
TEST2_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
            '/Talos 2020/Software/Dataset/Test2/'
MODEL_DIR = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots/Talos 2020/Software/ANIMA/AI Models/'
SEQUENCE_LENGTH = 32


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
        y = y[:, 14:]
        Y = y[(item-1)*self.batchsize: item * self.batchsize]
        Y = Y.reshape(1, self.batchsize, 2)
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
        return np.array(XI), np.array(Y)


TrainGen = CustomGenerator(controlsfile=TRAIN_DIR + 'Controls.csv', statusfile=TRAIN_DIR + 'Status.csv',
                           sensorsfile=TRAIN_DIR + 'Sensors.csv', imagedir=TRAIN_DIR, batchsize=SEQUENCE_LENGTH,
                           targetsize=[256, 192])


input1 = keras.layers.Input(shape=[SEQUENCE_LENGTH, 256, 192, 3])
Z = keras.layers.TimeDistributed(
    keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='relu'))(input1)
Z = keras.layers.TimeDistributed(keras.layers.MaxPool2D(2))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Conv2D(16, 5, padding='same', activation='relu'))(Z)
Z = keras.layers.TimeDistributed(keras.layers.MaxPool2D(2))(Z)
Z = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(Z)
Z = keras.layers.TimeDistributed(keras.layers.Dropout(0.4))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Flatten())(Z)
Z = keras.layers.TimeDistributed(keras.layers.Dense(20, 'relu'))(Z)
Z = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(Z)
Z = keras.layers.SimpleRNN(15, 'relu', return_sequences=True)(Z)
output = keras.layers.TimeDistributed(keras.layers.Dense(2, 'sigmoid'))(Z)


model = keras.Model(inputs=[input1], outputs=[output])
model.compile(optimizer=keras.optimizers.SGD(momentum=0.4), loss='mse', metrics=['mse', 'mae'])

print(model.summary())
train = int(input('train?'))
if train:
    history = model.fit(TrainGen, epochs=100, batch_size=1,
              callbacks=[keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2, monitor="loss"),
                                               keras.callbacks.EarlyStopping(patience=3, monitor="loss")])

    pd.DataFrame(history.history).plot(figsize=(12, 7))
    plt.grid(True)
    plt.gca().set_ylim(0, 0.05)
    plt.show()
    save = int(input('train save?'))
    if save:
        model.save(MODEL_DIR + 'TrainedModel')

model = keras.models.load_model(MODEL_DIR + 'TrainedModel')

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
