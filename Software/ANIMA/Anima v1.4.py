from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import adafruit_mpu6050
import adafruit_gps
import serial
from evdev import InputDevice, categorize, ecodes
import multiprocessing as mp
import tensorflow as tf
import busio
import board
import time
from picamera import PiCamera
import numpy as np
import os
import csv
from select import select

class CAM:
    cam = PiCamera()
    EFFECTS = ['none']
    SERVOZ = None
    SERVOY = None
    servochannel = [10, 8] #[y, z]
    servopos = [90, 90]
    ID = 0

    def __init__(self, pca):
        #initialize the servos on the camera mount 
        self.SERVOY = servo.Servo(pca.channels[self.servochannel[0]],
                                    min_pulse=700, max_pulse=2400)
        self.SERVOZ = servo.Servo(pca.channels[self.servochannel[1]],
                                    min_pulse=700, max_pulse=2400)
        #set the servos to the initial position
        self.SERVOY.angle = self.servopos[0]
        self.SERVOZ.angle = self.servopos[1]
        
        #camera settings
        self.cam.image_denoise = True
        self.cam.led = False
        self.cam.framerate = 60
        
        #the number of Images already present in the Data folder are counted so
        # that the next images taken will not overwrite the previous ones
        for name in os.listdir("/home/pi/Desktop/ANIMA/Data"):
            if 'image' in name:
                self.ID += 1
        print('IDRec: '+str(self.ID))

    #when called it responds to the commands given by the CONTROLS class
    def Update_Camera_Commands(self):
        #these if-else statements are to avoid angle out of range errors 
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
            self.Shoot('user')
            CONTROLS.SBUTTONDX = 0
        if CONTROLS.B:
            self.Record()
            CONTROLS.B = 0
        self.SERVOY.angle = self.servopos[0]
        self.SERVOZ.angle = self.servopos[1]

    #this method is necessary to release the hardware and memory resources used by the camera.
    #if they are ot released the PiCamera module will throw an exception as soon as the cam instance is
    #reinitialized on a new excecution of the program
    def Close(self):
        self.cam.close()

    #when called it shoots as many images as many different filters
    #are specified in the EFFECTS list and saves them to the Immagini folder
    def Shoot(self, mode):
        if mode == 'user':
            for effect in self.EFFECTS:
                self.cam.image_effect = effect
                self.cam.capture("/home/pi/Desktop/ANIMA/Immagini/image" + time.strftime("%M%S")+effect+".jpg")
        elif mode == 'ai':
            output = np.empty((1024, 768, 3))
            self.cam.capture(output, 'rgb')
            return output


    # when called it shoots one image and saves it to the data folder. at the same time
    #it records which commands were given in response to the input contained in the image
    # and in what position each servo of the robot was in that situation
    def RecSave(self):
        self.cam.capture("/home/pi/Desktop/ANIMA/Data/image" + str(self.ID) + ".jpg", use_video_port=True)
        with open('/home/pi/Desktop/ANIMA/Data/Controls.csv', 'a') as control:
            controlwriter = csv.writer(control, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            controlwriter.writerow([CONTROLS.A, CONTROLS.B, CONTROLS.X, CONTROLS.Y,
                                    CONTROLS.STICKDX[0], CONTROLS.STICKDX[1],
                                    CONTROLS.STICKSX[0], CONTROLS.STICKSX[1],
                                    CONTROLS.CROSS[0], CONTROLS.CROSS[1], CONTROLS.SBUTTONDX,
                                    CONTROLS.SBUTTONSX, CONTROLS.RB, CONTROLS.LB, CONTROLS.RT,
                                    CONTROLS.LT])
        with open('/home/pi/Desktop/ANIMA/Data/Status.csv', 'a') as status:
            statuswriter = csv.writer(status, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            statuswriter.writerow([LEGS.servopos[0],LEGS.servopos[1],
                                   LEGS.servopos[2],LEGS.servopos[3],
                                   ARM.servopos[0],ARM.servopos[1],ARM.servopos[2],
                                   ARM.servopos[3],ARM.servopos[4],self.servopos[0],
                                   self.servopos[1]])
        self.ID+=1
            
    # handles the start, end and save of video recordings, which are then stored in the
    # Video folder
    def Record(self):
        if self.cam.recording:
            self.cam.stop_recording()
            print('Stopped Recording')
        else:
            self.cam.start_recording("/home/pi/Desktop/ANIMA/Video/video" + time.strftime("%M%S")+".h264")
            print('Started Recording')


# stores the data, updates it and handles the calibration of the Inertial Measurement Unit
# in this case the robot uses a MPU6050
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
        print('Calibrating...')
        for _ in range(0, 2000):
            self.OFFSETGYRO = np.array(self.OFFSETGYRO) + np.array(self.mpu.gyro)
            self.OFFSETACCEL = np.array(self.OFFSETACCEL) + np.array(self.mpu.acceleration)
        for i in range(len(self.OFFSETGYRO)):
            self.OFFSETGYRO[i] /= 2000
        self.OFFSETACCEL[2] = self.OFFSETACCEL[2] - 2000
        for i in range(len(self.OFFSETACCEL)):
            self.OFFSETACCEL[i] /= 2000
        print('Calibration Finished')


    def UpdateData(self):
        deltag=[0, 0, 0]
        deltaa=[0, 0, 0]
        #takes the average rotation or acceleration values of 10 readings and
        #updates the data
        for _ in range(0, 10):
            deltag += (np.array(self.mpu.gyro) -
                           np.array(self.OFFSETGYRO))
            deltaa += (np.array(self.mpu.acceleration) -
                           np.array(self.OFFSETACCEL))
        now = time.time()
        if self.prev:
            for i in range(len(deltag)):
                deltag[i] /= 10
            self.GYRO += (np.array(deltag))*(now-self.prev)
            for i in range(len(deltaa)):
                deltaa[i] /= 10
            self.ACCEL = np.array(deltaa)
        self.prev = time.time()
        self.TEMPERATURE = self.mpu.temperature


# stores and updates the GPS coordinates of the robot
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


'''
#if an ultrasonic sensor is installed on the robot
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
'''


#this class is responsible for the update of the commands of all servos and motors
#on the rover. When in REC mode it takes as input an xbox controller, while when in
#autonomous mode the commands are given by an AI algorithm
class CONTROLS:
    gamepad = InputDevice('/dev/input/event0')
    MODE = "rc"
    RT = 0            # 0 -> 255 -> 0 to 1
    LT = 0            # 0 -> 255 -> 0 to 1
    RB = 0
    LB = 0
    SBUTTONSX = 0
    SBUTTONDX = 0
    STICKDX = [0, 0]  # [y, x] up-32768, center -1, down 32767,
                      # left-32767, center 0, right 32767 -> -1 to 1
    STICKSX = [0, 0]  # [y, x] up-32768, center -1, down 32767,
                      # left-32767, center 0, right 32767 -1 to 1
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
    pre = 0

    #this function handles the Controller inputs
    def UpdateOrders(self):
        r,w,d = select([self.gamepad], [], [], 1)
        if r:
            # during testing I noticed that the STICKSX[1] command was updated in
            #a way somewhat different than all the others. This meant that while all the
            #other controls were able to "get out" of this class to actuate LEGS and ARM,
            # the STICKSX[1] command was set to zero before it could reach its intended destination
            # by restricting its ability to reset itself I was able to make it work.
            # unfortunately the result is not as elegant as it should be, with all commands
            #treated equally.
            fssx = 1
            for event in self.gamepad.read():
                if event.code == self.a:
                    self.A = event.value
                    if self.A == 1:
                        if self.MODE == 'rc':
                            self.MODE = 'auto'
                        elif self.MODE == 'auto':
                            self.MODE = 'rc'
                elif event.code == self.sticksx[1] and fssx:
                    self.STICKSX[1] = event.value/32768
                    fssx=0
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
                    self.LT = event.value/255
                elif event.code == self.rt:
                    self.RT = event.value/255
                elif event.code == self.rb:
                    self.RB = event.value
                elif event.code == self.lb:
                    self.LB = event.value
                elif event.code == self.stickdx[0]:
                    self.STICKDX[0] = (event.value+1)/32768
                elif event.code == self.stickdx[1]:
                    self.STICKDX[1] = event.value/32768
                elif event.code == self.sticksx[0]:
                    self.STICKSX[0] = (event.value+1)/32768
                
            
                


class LEGS:
    SERVOADX = None
    SERVOPDX = None
    SERVOASX = None
    SERVOPSX = None
    MOTORADX = None
    MOTORASX = None
    servopos = [87, 92, 40, 36] # ADX, ASX, PDX, PSX
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
        
        # this sequence Calibrates the motors ESC. this calibration
        #is not always necessary
        print('MotorCalibration?')
        yes = input()
        if yes == '1':
            print('MotorCalibrating...')
            self.MOTORADX.throttle = 0.9
            self.MOTORASX.throttle = 0.9
            time.sleep(4)
        self.MOTORADX.throttle = 0
        self.MOTORASX.throttle = 0
        print('MotorCalibration Finished')
        
        self.SERVOADX.angle = self.servopos[0]
        self.SERVOASX.angle = self.servopos[1]
        self.SERVOPDX.angle = self.servopos[2]
        self.SERVOPSX.angle = self.servopos[3]

    def Update_Legs_Position(self):
        self.motorthrottle[0] = 0
        self.motorthrottle[1] = 0
        self.MOTORADX.throttle = self.motorthrottle[0]
        self.MOTORASX.throttle = self.motorthrottle[1]
        self.servopos[0] += CONTROLS.CROSS[0]
        self. servopos[1] += CONTROLS.CROSS[0]
        self.servopos[2] += CONTROLS.CROSS[1]
        self.servopos[3] += CONTROLS.CROSS[1]
        self.motorthrottle[0] = CONTROLS.RT / 1.5
        self.motorthrottle[1] = CONTROLS.LT / 1.5
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
            self.servopos[0] -= CONTROLS.STICKSX[1]*2
        else:
            if self.servopos[0] > 90:
                self.servopos[0] = 169
            else:
                self.servopos[0] = 11

        if self.servopos[1] > 10 and self.servopos[1]< 150:
            self.servopos[1] += CONTROLS.STICKSX[0]*2
        else:
            if self.servopos[1] > 90:
                self.servopos[1] = 149
            else:
                self.servopos[1] = 11
                
        # this if statement makes sure that the forearm only moves when the arm is
        #beyond 110 degrees forward, so that the grabber moves parallel to the ground
        #and at a certain height
        if self.servopos[1] > 110:
            self.servopos[2] += CONTROLS.STICKSX[0]*2
        elif self.servopos[2]>=1:
            self.servopos[2]-=1
            
        # the wrist servo is set to a position that makes it parallel to the ground
        # I decided the 1.8 ratio as a result of experimenting with the arm movement
        self.servopos[3] = self.servopos[1]/1.5
        
        # the grabber servos moves to open and close the fingers in a
        # range between 10 and 50 degrees
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


class AI():
    MODEL_PATH = ''
    SLIDING_WINDOW = []
    def __init__(self):
        # load the tflite model. This model was optimized for latency and quantized to float16.
        # The whole program is executed by a Raspberry pi4 with 4GB of RAM, so optimization is necessary
        self.model = tf.lite.Interpreter(self.MODEL_PATH)
        self.model.allocate_tensors()
        # since our model is a recurrent model that process 16 steps long sequences,
        # we initialize a list to contain 16 input steps.
        for i in range(16):
            MPU.UpdateData()
            GPS.GPS_Update()
            Sensors = np.array(np.concatenate((MPU.GYRO, MPU.ACCEL, GPS.CURRENT_COORDINATES)))
            image = np.array(CAM.Shoot('ai')).resize((256, 192, 3))
            self.SLIDING_WINDOW.append([image, Sensors])

    def AIupdate(self):
        MPU.UpdateData()
        GPS.GPS_Update()
        # to introduce a new input while maintaining the same 16 steps length,
        # we remove the first element of SLIDING_WINDOW and we introduce the new input
        # at the back of the list. We implemented a 16 blocks FIFO buffer
        self.SLIDING_WINDOW.pop(0)
        Sensors = np.array(np.concatenate((MPU.GYRO, MPU.ACCEL, GPS.CURRENT_COORDINATES), axis=-1))
        image = np.array(CAM.Shoot('ai')).resize((512, 384, 3))
        self.SLIDING_WINDOW.append([image, Sensors])

        # inference part
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        input_1 = np.array(self.SLIDING_WINDOW[0], dtype=np.float32)
        input_2 = np.array(self.SLIDING_WINDOW[1], dtype=np.float32)
        self.model.set_tensor(input_details[0]['index'], input_1)
        self.model.set_tensor(input_details[1]['index'], input_2)
        self.model.invoke()
        output = self.model.get_tensor(output_details[0]['index'])[0][15]

        # translation from model output to CONTROL commands
        CONTROLS.A = output[0]
        CONTROLS.B = output[1]
        CONTROLS.X = output[2]
        CONTROLS.Y = output[3]
        CONTROLS.STICKDX[0] = output[4]
        CONTROLS.STICKDX[1] = output[5]
        CONTROLS.STICKSX[0] = output[6]
        CONTROLS.STICKSX[1] = output[7]
        CONTROLS.CROSS[0] = output[8]
        CONTROLS.CROSS[1] = output[9]
        CONTROLS.SBUTTONDX = output[10]
        CONTROLS.SBUTTONSX = output[11]
        CONTROLS.RB = output[12]
        CONTROLS.LB = output[13]
        CONTROLS.RT = output[14]
        CONTROLS.LT = output[15]


        
# when in REC mode shis extra process is needed. Sensors data update delay is not
# really a problem, and the fact that the evdev controller blocks the execution of the main
# process could be fixed to a reasonable functioning order by putting a timeout of 0.5 in the
# controls select function. however that way the command response would have a bothering lag
# which in case of the motors could be dangerous for the robot. 
def SENProcess(CONsignal):
    print('Sensors Started')
    while True:
        MPU.UpdateData()
        GPS.GPS_Update()
        if CONsignal.is_set():
            CONsignal.clear()
            print("SEN")
            with open('/home/pi/Desktop/ANIMA/Data/Sensors.csv', 'a') as sensors:
                sensorswriter = csv.writer(sensors, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
                sensorswriter.writerow([MPU.GYRO[0], MPU.GYRO[1], MPU.GYRO[2],
                                        MPU.ACCEL[0], MPU.ACCEL[1], MPU.ACCEL[2],
                                        GPS.CURRENT_COORDINATES[0], GPS.CURRENT_COORDINATES[1]])
                
        
def RECProcess(SYNCTIME):
    CONTROLS.UpdateOrders()
    now = time.time()
    if now-SYNCTIME > 0.2:
        print('REC: '+str(now-SYNCTIME))
        CAM.RecSave()
        CONsignal.set()

                   
def cleanup():
    CAM.Close()
    SEN.terminate()
    SEN.join()
    print("exit procedure done")


try:
    SYNCTIME = time.time()
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 60
    LEGS = LEGS(pca)
    ARM = ARM(pca)
    CAM = CAM(pca)
    MPU = MPU()
    MPU.Calibration()
    GPS = GPS()
    CONTROLS = CONTROLS()
    CONsignal = mp.Event()
    SEN = mp.Process(target=SENProcess, args = (CONsignal,))
    SEN.start()
    PMODE = 'rc'
    print('Setup Finished')
    while True:
        SYNCTIME = time.time()
        if CONTROLS.MODE == 'rc':
            if PMODE != 'rc':
                print('Switching to RC MODE ...')
                MPU = MPU()
                MPU.Calibration()
                GPS = GPS()
                SEN = mp.Process(target=SENProcess, args=(CONsignal,))
                SEN.start()
                print('RC MODE started')
            RECProcess(SYNCTIME)

        elif CONTROLS.MODE == 'auto':
            if PMODE != 'auto':
                print('Switching to AUTONOMOUS MODE ...')
                SEN.terminate()
                MPU = MPU()
                MPU.Calibration()
                GPS = GPS()
                AI = AI()
                print('AUTONOMOUS MODE started')
            AI.AIupdate()

        LEGS.Update_Legs_Position()
        CAM.Update_Camera_Commands()
        ARM.Update_Arm_Position()
        PMODE = CONTROLS.MODE
except KeyboardInterrupt:
    cleanup()
    print('exit')
except:
    cleanup()
    print('exit')
