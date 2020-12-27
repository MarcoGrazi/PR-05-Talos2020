import os
import csv
import natsort
from PIL import Image


'''
the objective of this script is to take the data from each Data folder and concatenate it into
one Train dataset. Due to the undeterministic nature of parallel programming, and to occasional
hardware hiccups in the Talos2020 robot some csv files do not contain the same number of datalines as
the others, or at least not as many as the number of images recorded. If the difference is too large 
all the data has to be discarded because the misalignment in terms of time sequence would be too large.
However sometimes the difference in datalines is due to one process stop working properly, so it is not 
associated with a misalignment in the time sequence. Taking into account all these considerations I have
designed this script to patch the missing lines with 0 values, so that the data doesn't have to be discarded.
As a side effect this processing of the data could be beneficial to the reliability of the final AI algorithm
teaching it to account for noise and missing data (in case for example of an hardware failure)   
'''

gp = 0
gpc = 0

for folder in os.listdir('C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots'
          '/Talos 2020/Software/Dataset'):
    if 'Data' in folder:
        basename = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots' \
                   '/Talos 2020/Software/Dataset/' + folder
        length = 0
        for image in os.listdir(basename):
            if 'image' in image:
                length += 1

        with open(basename+'/Controls.csv', 'r') as c:
            con = csv.reader(c)
            clength = 0
            print(length)
            for row in con:
                clength += 1
            print(clength)
        with open(basename + '/Status.csv', 'r') as st:
            stat = csv.reader(st)
            stlength = 0
            for row in stat:
                stlength += 1
            print(stlength)
        with open(basename + '/Sensors.csv', 'r') as se:
            sen = csv.reader(se)
            selength = 0
            for row in sen:
                selength += 1
            print(selength)
        print(length - selength)
        print('-----------------')
        with open(basename + '/Controls.csv', 'a', newline='') as c:
            conwriter = csv.writer(c, delimiter=',',
                                       quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(length-clength):
                conwriter.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        with open(basename + '/Status.csv', 'a', newline='') as st:
            statwriter = csv.writer(st, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(length-stlength):
                statwriter.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        with open(basename + '/Sensors.csv', 'a', newline='') as se:
            senwriter = csv.writer(se, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(length-selength):
                senwriter.writerow([0, 0, 0, 0, 0, 0, 0, 0])

        Images = os.listdir(basename)
        Images = natsort.natsorted(Images)
        for image in Images:
            if 'image' in image:
                im = Image.open(basename+"/"+image)
                im.save('C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots'
                        '/Talos 2020/Software/Dataset/Train/image' +str(gp)+'.jpg')
                gp += 1

        with open('C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots'
                        '/Talos 2020/Software/Dataset/Train/Controls.csv', 'a', newline='') as control:
            controlwriter = csv.writer(control, delimiter=',',
                                       quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            with open(basename + '/Controls.csv', 'r') as c:
                con = csv.reader(c)
                for row in con:
                    controlwriter.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                                            row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15]])
                    gpc += 1

        with open('C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots'
                        '/Talos 2020/Software/Dataset/Train/Status.csv', 'a', newline='') as status:
            statuswriter = csv.writer(status, delimiter=',',
                                      quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            with open(basename + '/Status.csv', 'r') as st:
                stat = csv.reader(st)
                for row in stat:
                    statuswriter.writerow([float(row[0])/180, float(row[1])/180, float(row[2])/180,
                                           float(row[3])/180, float(row[4])/180, float(row[5])/180,
                                           float(row[6])/180, float(row[7])/180, float(row[8])/180,
                                           float(row[9])/180, float(row[10])/180])

        with open('C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/Robots'
                        '/Talos 2020/Software/Dataset/Train/Sensors.csv', 'a', newline='') as sensors:
            sensorswriter = csv.writer(sensors, delimiter=',',
                                       quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            with open(basename + '/Sensors.csv', 'r') as se:
                sen = csv.reader(se)
                for row in sen:
                    if float(row[2])/360 >= 1:
                        row[2] = str(float(row[2]) - 360 * (float(row[2]) // 360))
                    if float(row[2])/360 <= -1:
                        row[2] = str(float(row[2]) - 360 * (float(row[2]) // 360))
                    if float(row[2]) >= 180:
                        row[2] = str(float(row[2]) - 360)
                    if float(row[2]) <= -180:
                        row[2] = str(float(row[2]) + 360)
                    sensorswriter.writerow([float(row[0])/180, float(row[1])/180,
                                            float(row[2])/180, row[3], row[4], row[5],
                                            float(row[6])%1, float(row[7])%1])
print(gp)











