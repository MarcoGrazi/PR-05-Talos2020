import tflite
import Adafruit_PCA9685
import adafruit_mpu6050
import threading
import cv2
import math
import sys
import serial
from time import sleep
from picamera import PiCamera
import BlynkLib as B

BLYNK_AUTH = ""

CAM = PiCamera()
BLYNK = B.Blynk(BLYNK_AUTH)


class MPU:
    GYRO = [0, 0, 0]  # [x, y, z]
    ACCEL = [0, 0, 0]  # [x, y, z]
    TEMPERATURE = 0
    OFFSETGYRO = [0, 0, 0]  #[x, y, z]
    OFFSETACCEL = [0, 0, 0]  #[x, y, z]
    mpu = adafruit_mpu6050.MPU6050()
    mpu.gyro_range = adafruit_mpu6050.GyroRange.RANGE_250_DPS
    mpu.accelerometer_range = adafruit_mpu6050.Range.RANGE_2_G

    def Calibration(self):
        global OFFSETGYRO
        global OFFSETACCEL
        global mpu
        for _ in range(0, 300):
            OFFSETGYRO += mpu.gyro()
            OFFSETACCEL += mpu.acceleration()
        OFFSETGYRO /= 300
        OFFSETACCEL /= 300

    def UpdateData(self):
        global GYRO
        global OFFSETGYRO
        global ACCEL
        global OFFSETACCEL
        global TEMPERATURE
        for _ in range(0, 20):
            GYRO += (mpu.gyro - OFFSETGYRO)*0.004
            ACCEL += (mpu.acceleration - OFFSETACCEL)*0.004
            sleep(0.004)
        GYRO /= 20
        ACCEL /= 20
        TEMPERATURE = mpu.temperature


class GPS:
    gpgga_info = "$GPGGA,"
    ser = serial.Serial("/dev/ttyS0")
    GPGGA_buffer = 0
    NMEA_buff = 0
    EARTH_RADIUS = 6371
    DIST_TRAVELLED = 0
    DIST_INITIAL = 0
    CURRENT_COORDINATES = []
    PREV_COORDINATES = []
    COUNTER = 0

    # convert raw NMEA string into degree decimal format
    def convert_to_degrees(self, raw_value):
        decimal_value = raw_value / 100.00
        degrees = int(decimal_value)
        mm_mmmm = (decimal_value - int(decimal_value)) / 0.6
        position = degrees + mm_mmmm
        position = "%.4f" % (position)
        return position

    def GPS_Coordinates(self):
        global NMEA_buff
        global CURRENT_COORDINATES
        global PREV_COORDINATES
        global COUNTER
        nmea_time = []
        nmea_latitude = []
        nmea_longitude = []
        nmea_time = NMEA_buff[0]  # extract time from GPGGA string
        nmea_latitude = NMEA_buff[1]  # extract latitude from GPGGA string
        nmea_longitude = NMEA_buff[3]  # extract longitude from GPGGA string

        lat = float(nmea_latitude)  # convert string into float for calculation
        longi = float(nmea_longitude)  # convert string into float for calculation

        CURRENT_COORDINATES[0] = self.convert_to_degrees(lat)  # get latitude in degree decimal format
        CURRENT_COORDINATES[1] = self.convert_to_degrees(longi)  # get longitude in degree decimal format

        PREV_COORDINATES.append(CURRENT_COORDINATES)
        COUNTER += 1

    def CalculateDistance(self, mode):
        global DIST_TRAVELLED
        global DIST_INITIAL
        global PREV_COORDINATES
        global EARTH_RADIUS
        if mode=="travelled":
            for i in range(1, len(PREV_COORDINATES)):
                dlat = math.radians(PREV_COORDINATES[i-1][0] - PREV_COORDINATES[i][0])
                dlong = math.radians(PREV_COORDINATES[i-1][1] - PREV_COORDINATES[i][1])
                a = (math.sin(dlat/2))**2 +math.cos(math.radians(PREV_COORDINATES[i-1][0])) *\
                    math.cos(PREV_COORDINATES[i][0]) * (math.sin(dlong/2))**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                DIST_TRAVELLED += c*EARTH_RADIUS
        elif mode=="initial":
            dlat = math.radians(PREV_COORDINATES[0][0] - PREV_COORDINATES[len(PREV_COORDINATES)][0])
            dlong = math.radians(PREV_COORDINATES[0][1] - PREV_COORDINATES[len(PREV_COORDINATES)][1])
            a = (math.sin(dlat / 2)) ** 2 + math.cos(math.radians(PREV_COORDINATES[0][0])) * \
                math.cos(PREV_COORDINATES[len(PREV_COORDINATES)][0]) * (math.sin(dlong / 2)) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            DIST_INITIAL = c*EARTH_RADIUS

MPU = MPU()
MPU.Calibration()
GPS = GPS()
GPS.GPS_Coordinates()
print("GPS: " + str(GPS.CURRENT_COORDINATES[0]) + ", " + str(GPS.CURRENT_COORDINATES[1]))

while True:
    MPU.UpdateData()
    print("GYRO: " + str(MPU.GYRO[0]) + ", " + str(MPU.GYRO[1]) + ", " + str(MPU.GYRO[2]))
    print("ACCEL: " + str(MPU.ACCEL[0]) + ", " + str(MPU.ACCEL[1]) + ", " + str(MPU.ACCEL[2]))
    print("TEMPERATURE: " + str(MPU.TEMPERATURE))










