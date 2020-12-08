from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import adafruit_mpu6050
import adafruit_gps
import serial
from evdev import InputDevice, categorize, ecodes
import multiprocessing
import math
import RPi.GPIO as GPIO
import busio
import board
import time
from picamera import PiCamera
import numpy as np
import os
import csv


class CAM:
    cam = PiCamera()
    cam.image_denoise = True
    cam.led = False
    EFFECTS = ['none']
    SERVOZ = None
    SERVOY = None
    servochannel = [10, 8] #[y, z]
    servopos = [90, 80]

    def __init__(self, pca):
        self.SERVOY = servo.Servo(pca.channels[self.servochannel[0]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOZ = servo.Servo(pca.channels[self.servochannel[1]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOY.angle = self.servopos[0]
        self.SERVOZ.angle = self.servopos[1]

    def Update_Camera_Commands(self):
        if(self.servopos[0] < 170 and self.servopos[0]>50):
            self.servopos[0] += CONTROLS.STICKDX[0] / 16384
        else:
            if self.servopos[0]>90:
                self.servopos[0] = 169
            else:
                self.servopos[0] = 51
        if (self.servopos[1] < 150 and self.servopos[1] > 30):
            self.servopos[1] -= CONTROLS.STICKDX[1] / 16384
        else:
            if self.servopos[1]>90:
                self.servopos[1] = 149
            else:
                self.servopos[1] = 31
        if CONTROLS.SBUTTONDX:
            self.Shoot()
            CONTROLS.SBUTTONDX = 0
        if CONTROLS.B:
            self.Record()
            CONTROLS.B = 0
        self.SERVOY.angle = self.servopos[0]
        self.SERVOZ.angle = self.servopos[1]

    def Shoot(self):
        for effect in self.EFFECTS:
            self.cam.image_effect = effect
            self.cam.capture("/home/pi/Desktop/ANIMA/Immagini/image" + time.strftime("%M%S")+effect+".jpg")

    def RecSave(self, ID):
        self.cam.capture("/home/pi/Desktop/ANIMA/Data/image" + str(ID) + ".jpg")

    def Record(self):
        if self.cam.recording:
            self.cam.stop_recording()
            print('Stopped Recording')
        else:
            self.cam.start_recording("/home/pi/Desktop/ANIMA/Video/video" + time.strftime("%M%S")+".h264")
            print('Started Recording')

class MPU:
    GYRO = [0, 0, 0]  # [x, y, z]
    ACCEL = [0, 0, 0]  # [x, y, z]
    TEMPERATURE = 0
    OFFSETGYRO = [0, 0, 0]  #[x, y, z]
    OFFSETACCEL = [0, 0, 0]  #[x, y, z]
    I2C = busio.I2C(board.SCL, board.SDA)
    mpu = adafruit_mpu6050.MPU6050(I2C)
    mpu.gyro_range = adafruit_mpu6050.GyroRange.RANGE_250_DPS
    mpu.accelerometer_range = adafruit_mpu6050.Range.RANGE_2_G
    prev = 0

    def Calibration(self):
        for _ in range(0, 2000):
            self.OFFSETGYRO = np.array(self.OFFSETGYRO) + np.array(self.mpu.gyro)
            self.OFFSETACCEL = np.array(self.OFFSETACCEL) + np.array(self.mpu.acceleration)
        for i in range(len(self.OFFSETGYRO)):
            self.OFFSETGYRO[i] /= 2000
        self.OFFSETACCEL[2] = self.OFFSETACCEL[2] - 2000
        for i in range(len(self.OFFSETACCEL)):
            self.OFFSETACCEL[i] /= 2000


    def UpdateData(self):
        deltag=[0, 0, 0]
        deltaa=[0, 0, 0]
        for _ in range(0, 50):
            deltag += (np.array(self.mpu.gyro) -
                           np.array(self.OFFSETGYRO))
            deltaa += (np.array(self.mpu.acceleration) -
                           np.array(self.OFFSETACCEL))
        now = time.time()
        if self.prev:
            for i in range(len(deltag)):
                deltag[i] /= 50
            self.GYRO += (np.array(deltag))*(now-self.prev)
            for i in range(len(deltaa)):
                deltaa[i] /= 50
            self.ACCEL = np.array(deltaa)
        self.prev = time.time()
        self.TEMPERATURE = self.mpu.temperature


class GPS:
    uart = serial.Serial('/dev/serial0', baudrate=9600, timeout=10)
    gps = adafruit_gps.GPS(uart, debug=False)
    gps.send_command(b'PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,')
    gps.send_command(b'PMTK220, 1000')
    EARTH_RADIUS = 6371
    DIST_TRAVELLED = 0
    DIST_INITIAL = 0
    CURRENT_COORDINATES = [0, 0]
    PREV_COORDINATES = []
    COUNTER = 0

    def GPS_Update(self):
        self.gps.update()
        if self.gps.has_fix:
            self.CURRENT_COORDINATES[0] = self.gps.latitude
            self.CURRENT_COORDINATES[1] = self.gps.longitude
            if self.CURRENT_COORDINATES[0]!=0 and self.CURRENT_COORDINATES[1]!=0:
                self.PREV_COORDINATES.append(self.CURRENT_COORDINATES)
            self.COUNTER += 1

    def CalculateDistance(self, mode):
        if mode == "travelled":
            for i in range(1, len(self.PREV_COORDINATES)):
                dlat = math.radians(self.PREV_COORDINATES[i-1][0] - self.PREV_COORDINATES[i][0])
                dlong = math.radians(self.PREV_COORDINATES[i-1][1] - self.PREV_COORDINATES[i][1])
                a = (math.sin(dlat/2))**2 +math.cos(math.radians(self.PREV_COORDINATES[i-1][0])) *\
                    math.cos(self.PREV_COORDINATES[i][0]) * (math.sin(dlong/2))**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                self.DIST_TRAVELLED += c*self.EARTH_RADIUS
        elif mode == "initial":
            dlat = math.radians(self.PREV_COORDINATES[0][0] - self.PREV_COORDINATES[len(self.PREV_COORDINATES)][0])
            dlong = math.radians(self.PREV_COORDINATES[0][1] - self.PREV_COORDINATES[len(self.PREV_COORDINATES)][1])
            a = (math.sin(dlat / 2)) ** 2 + math.cos(math.radians(self.PREV_COORDINATES[0][0])) * \
                math.cos(self.PREV_COORDINATES[len(self.PREV_COORDINATES)][0]) * (math.sin(dlong / 2)) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            self.DIST_INITIAL = c*self.EARTH_RADIUS


class USENSOR:
    GPIO.setmode(GPIO.BCM)
    GPIO_TRIGGER = 24
    GPIO_ECHO = 25
    GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
    GPIO.setup(GPIO_ECHO, GPIO.IN)
    DISTANCE = 0

    def UpdateDistance(self):
        GPIO.output(self.GPIO_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.GPIO_TRIGGER, False)
        StartTime = time.time()
        StopTime = time.time()
        while GPIO.input(self.GPIO_ECHO) == 0:
            StartTime = time.time()
        while GPIO.input(self.GPIO_ECHO) == 1:
            StopTime = time.time()
        TimeElapsed = StopTime - StartTime
        self.DISTANCE = (TimeElapsed * 34300) / 2


class CONTROLS:
    gamepad = InputDevice('/dev/input/event0')
    MODE = "rc"
    RT = 0            # 0 -> 255
    LT = 0            # 0 -> 255
    RB = 0
    LB = 0
    SBUTTONSX = 0
    SBUTTONDX = 0
    STICKDX = [0, 0]  # [y, x] up-32768, center -1, down 32767,
                      # left-32767, center 0, right 32767
    STICKSX = [0, 0]  # [y, x] up-32768, center -1, down 32767,
                      # left-32767, center 0, right 32767
    CROSS = [0, 0]    # [y, x] up-1, down 1, left-1, right 1
    A = 0
    B = 0
    X = 0
    Y = 0
    a = 304
    b = 305
    x = 307
    y = 308
    cross = [17, 16]
    sbuttondx = 318
    sbuttonsx = 317
    lt = 2
    rt = 5
    rb = 311
    lb = 310
    stickdx = [4, 3]
    sticksx = [1, 0]


    def UpdateOrders(self):
        events = self.gamepad.read_loop()
        for event in events:
            if event.code == self.a:
                self.A = event.value
                if self.A == 1:
                    if self.MODE == 'rc':
                        self.MODE = 'auto'
                    elif self.MODE == 'auto':
                        self.MODE = 'rc'
            elif event.code == self.x:
                self.X = event.value
            elif event.code == self.y:
                self.Y = event.value
            elif event.code == self.b:
                self.B = event.value
            elif event.code == self.cross[0]:
                self.CROSS[0] = event.value
            elif event.code == self.cross[1]:
                self.CROSS[1] = event.value
            elif event.code == self.sbuttondx:
                self.SBUTTONDX = event.value
            elif event.code == self.sbuttonsx:
                self.SBUTTONSX = event.value
            elif event.code == self.lt:
                self.LT = event.value
            elif event.code == self.rt:
                self.RT = event.value
            elif event.code == self.rb:
                self.RB = event.value
            elif event.code == self.lb:
                self.LB = event.value
            elif event.code == self.stickdx[0]:
                self.STICKDX[0] = event.value+1
            elif event.code == self.stickdx[1]:
                self.STICKDX[1] = event.value
            elif event.code == self.sticksx[0]:
                self.STICKSX[0] = event.value+1
            elif event.code == self.sticksx[1]:
                self.STICKSX[1] = event.value
            events.close()


class LEGS:
    SERVOADX = None
    SERVOPDX = None
    SERVOASX = None
    SERVOPSX = None
    MOTORADX = None
    MOTORASX = None
    servopos = [88, 88, 40, 36] # ADX, ASX, PDX, PSX
    servochannel = [11, 6, 5, 4]
    motorthrottle = [0, 0] # ADX, ASX
    motorchannel = [0, 15]

    def __init__(self, PCA):
        self.SERVOADX = servo.Servo(PCA.channels[self.servochannel[0]],
                               min_pulse=700, max_pulse=2400)
        self.SERVOASX = servo.Servo(PCA.channels[self.servochannel[1]],
                               min_pulse=2400, max_pulse=700)
        self.SERVOPDX = servo.Servo(PCA.channels[self.servochannel[2]],
                               min_pulse=700, max_pulse=2400)
        self.SERVOPSX = servo.Servo(PCA.channels[self.servochannel[3]],
                               min_pulse=2400, max_pulse=700)
        self.MOTORADX = servo.ContinuousServo(PCA.channels[self.motorchannel[0]],
                                              min_pulse=1000, max_pulse=2400)
        self.MOTORASX = servo.ContinuousServo(PCA.channels[self.motorchannel[1]],
                                              min_pulse=1000, max_pulse=2400)
        self.MOTORADX.throttle = 0
        self.MOTORASX.throttle = 0
        self.SERVOADX.angle = self.servopos[0]
        self.SERVOASX.angle = self.servopos[1]
        self.SERVOPDX.angle = self.servopos[2]
        self.SERVOPSX.angle = self.servopos[3]

    def Update_Legs_Position(self):
        self.servopos[0] += CONTROLS.CROSS[0]
        self. servopos[1] += CONTROLS.CROSS[0]
        self.servopos[2] += CONTROLS.CROSS[1]
        self.servopos[3] += CONTROLS.CROSS[1]
        self.motorthrottle[0] = 0
        self.motorthrottle[1] = 0
        self.MOTORADX.throttle = self.motorthrottle[0]
        self.MOTORASX.throttle = self.motorthrottle[1]
        self.motorthrottle[0] = CONTROLS.RT / (255 * 1.5)
        self.motorthrottle[1] = CONTROLS.LT / (255 * 1.5)
        self.SERVOADX.angle = self.servopos[0]
        self.SERVOASX.angle = self.servopos[1]
        self.SERVOPDX.angle = self.servopos[2]
        self.SERVOPSX.angle = self.servopos[3]
        self.MOTORADX.throttle = self.motorthrottle[0]
        self.MOTORASX.throttle = self.motorthrottle[1]



class ARM():
    SERVOZ = None
    SERVOS = None
    SERVOE = None
    SERVOW = None
    SERVOH = None
    servopos = [90, 0, 0, 0, 0] #z, s, e, w, h
    servochannel = [9, 7, 12, 14, 13]
    def __init__(self, PCA):
        self.SERVOZ = servo.Servo(PCA.channels[self.servochannel[0]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOS = servo.Servo(PCA.channels[self.servochannel[1]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOE = servo.Servo(PCA.channels[self.servochannel[2]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOW = servo.Servo(PCA.channels[self.servochannel[3]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOH = servo.Servo(PCA.channels[self.servochannel[4]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOZ.angle = self.servopos[0]
        self.SERVOS.angle = self.servopos[1]
        self.SERVOE.angle = self.servopos[2]
        self.SERVOW.angle = self.servopos[3]
        self.SERVOH.angle = self.servopos[4]

    def Update_Arm_Position(self):
        if self.servopos[0] > 10 and self.servopos[0] < 170:
            self.servopos[0] -= CONTROLS.STICKSX[1]/16384
        else:
            if self.servopos[0] > 90:
                self.servopos[0] = 169
            else:
                self.servopos[0] = 11

        if self.servopos[1] > 10 and self.servopos[1]< 170:
            self. servopos[1] += CONTROLS.STICKSX[0]/16384
        else:
            if self.servopos[1] > 90:
                self.servopos[1] = 169
            else:
                self.servopos[1] = 11

        if self.servopos[1] > 110:
            self.servopos[2] += CONTROLS.STICKSX[0]/16384
        else:
            self.servopos[2] = 0
        self.servopos[3] = self.servopos[1]/1.8
        if CONTROLS.SBUTTONSX:
            if self.servopos[4]>10:
                self.servopos[4] = 0
            else:
                self.servopos[4] = 50
        self.SERVOZ.angle = self.servopos[0]
        self.SERVOS.angle = self.servopos[1]
        self.SERVOE.angle = self.servopos[2]
        self.SERVOW.angle = self.servopos[3]
        self.SERVOH.angle = self.servopos[4]


def AIprocess():
    MPU.GYRO = [0, 0, 0]
    MPU.Calibration()
    Model =
    while True:
        now = time.time()
        print('TIME: ' + str(now - init))
        MPU.UpdateData()
        GPS.GPS_Update()
        USENSOR.UpdateDistance()
        print("USENSOR: " + str(USENSOR.DISTANCE))
        print("GPS: " + str(GPS.CURRENT_COORDINATES[0]) + ", " + str(GPS.CURRENT_COORDINATES[1]))
        print("GYRO: " + str(MPU.GYRO[0]) + ", " + str(MPU.GYRO[1]) + ", " + str(MPU.GYRO[2]))
        print("ACCEL: " + str(MPU.ACCEL[0]) + ", " + str(MPU.ACCEL[1]) + ", " + str(MPU.ACCEL[2]))
        print("TEMPERATURE: " + str(MPU.TEMPERATURE))


def RECprocess():
    MPU.GYRO = [0, 0, 0]
    MPU.Calibration()
    pre = init
    cont = 0
    with open('"/home/pi/Desktop/ANIMA/Data/Controls.csv', 'w', newline='') as csvfile:
        controlswriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    with open('"/home/pi/Desktop/ANIMA/Data/Data.csv', 'w', newline='') as file:
        datawriter = csv.writer(file, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for name in os.listdir():
        if name[0:4] == 'image':
            cont += 1

    while True:
        MPU.UpdateData()
        GPS.GPS_Update()
        now = time.time()
        print(str(now-pre))
        if (now-pre) > 0.2:
            cont += 1
            CAM.RecSave(cont)
            controlswriter.writerow([CONTROLS.RB/255,  CONTROLS.LB/255, CONTROLS.STICKSX[0]/32768,
                                 CONTROLS.STICKSX[1]/32768,  CONTROLS.STICKDX[0]/32768,
                                 CONTROLS.STICKDX[1]/32768, CONTROLS.SBUTTONDX,  CONTROLS.SBUTTONSX,
                                 CONTROLS.CROSS[0], CONTROLS.CROSS[1]])
            datawriter.writerow([MPU.GYRO[0], MPU.GYRO[1], MPU.GYRO[2], MPU.ACCEL[0], MPU.ACCEL[1], MPU.ACCEL[2],
                                 GPS.CURRENT_COORDINATES[0], GPS.CURRENT_COORDINATES[1]])
            pre = now


i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 60
LEGS = LEGS(pca)
ARM = ARM(pca)
CAM = CAM(pca)
MPU = MPU()
MPU.Calibration()
GPS = GPS()
USENSOR = USENSOR()
CONTROLS = CONTROLS()
init = time.time()
prev = CONTROLS.MODE
REC = multiprocessing.Process(target=RECprocess)
REC.start()

while True:
    CONTROLS.UpdateOrders()
    LEGS.Update_Legs_Position()
    CAM.Update_Camera_Commands()
    ARM.Update_Arm_Position()
    if CONTROLS.MODE == "auto" and prev != 'auto':
        prev = 'auto'
        print(CONTROLS.MODE)
        REC.terminate()
        AI = multiprocessing.Process(target=AIprocess)
        AI.start()
    elif CONTROLS.MODE == "rc" and prev != 'rc':
        prev = 'rc'
        print(CONTROLS.MODE)
        AI.terminate()
        REC = multiprocessing.Process(target=RECprocess)
        REC.start()










