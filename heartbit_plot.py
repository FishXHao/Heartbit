import sys, serial
import numpy as np
from time import sleep
from collections import deque
from matplotlib import pyplot as plt
import threading as td
import multiprocessing as mp
from matplotlib.widgets import Button, RadioButtons, Slider
import pickle
import time
import copy
from scipy.signal import butter, lfilter

heartrate = None
whatpass = 'no'
cutfreq = 2
thread1 = None
thread2 = None
# class that holds analog data for N samples
data = []
def change_pass(label):
    global whatpass
    whatpass = label

def cutfreq_change(val):
    global cutfreq
    cutfreq = labar.val

def butter_pass(cutoff, fs, whatpass, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=whatpass, analog=False)
    return b, a

def butter_pass_filter(data, cutoff, fs, whatpass, order=5):
    b, a = butter_pass(cutoff, fs, whatpass, order=order)
    y = lfilter(b, a, data)
    return y

def store(event):
    global data
    my_file = open('my_file.pickle', 'wb')
    pickle.dump(data, my_file)
    my_file.close()

def loadfile(event):
    with open('my_file.pickle', 'rb') as file:
        data = pickle.load(file)
        plt.subplot(413)
        plt.ylim([np.min(list(data)[300:500]) - 10, np.max(list(data)[300:500]) + 10])
        plt.plot(data)

def clear(event):
    plt.subplot(413)
    plt.cla()

class AnalogData:
    # constr
    def __init__(self, maxLen):
        self.ax = deque([0.0] * maxLen)
        self.fft = ([0.0] * maxLen)
        self.ay = deque([0.0] * maxLen)
        self.maxLen = maxLen
        self.count = 0
        self.last = 0
        self.ratecurrent = 0
        self.ratelast = 0
        self.ifupper = 0
        self.smallrang = []
    # ring buffer
    def addToBuf(self, buf, val):
        global heartrate, data
        global thread1
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)
        self.smallrang = (np.array(buf))[0:50]
        self.fft = np.fft.fft(buf)/1000
        """取得心律"""
        if (val > (np.mean(self.smallrang) + 8))&(self.ifupper == 0):
            self.ifupper = 1
        if self.ifupper == 1:
            if val < (np.mean(self.smallrang) + 8):
                self.ratecurrent = time.time()
                if self.ratecurrent != self.ratelast:
                    heartrate = (int(100*(60/(self.ratecurrent-self.ratelast))))/100-20
                self.ifupper = 0
                self.ratelast = self.ratecurrent
        thread1 = None
        data = self.ax
        """fs
        if (self.count == 0):
            self.last = time.time()
            self.count += 1
        elif (self.count < 1000):
            self.count += 1
        else:
            current = time.time()
            print(1000 / (current - self.last))
            self.count = 0
        """

    # add data
    def add(self, data):
        assert (len(data) == 1)
        #thread = td.Thread(target=self.addToBuf, args=(self.ax, data[0]))
        #thread.start()
        self.addToBuf(self.ax, data[0])


# plot class
class AnalogPlot:
    count = 0
    # constr
    def __init__(self, analogData):
        # set plot to animated
        plt.ion()
        plt.subplot(411)
        self.axline, = plt.plot(analogData.ax, label="")
        plt.ylim([np.min(list(analogData.ax)[0:200]), np.max(list(analogData.ax)[0:200])])
        plt.subplot(412)
        self.fft, = plt.plot(analogData.fft)
        # plt.ylim([-500, 2023])

    # update plot
    def update(self, analogData):
        global heartrate, thread2, whatpass, cutfreq, data
        # print("min %s" % np.min(analogData.ax))
        self.count += 1
        if self.count % 5 == 0:
            plt.subplot(411)
            if whatpass == 'no':
                data = copy.deepcopy(np.array(analogData.ax))
                self.axline.set_ydata(analogData.ax)
                self.axline.set_label(heartrate)
                plt.ylim([np.min(list(analogData.ax)[0:500]) - 10, np.max(list(analogData.ax)[0:500]) + 10])
                plt.legend(loc='upper right')
                plt.subplot(412)
                fftdata = np.abs(np.fft.rfft(data))[0:1000]  # remove dc value  and high frequency value
                fftdata = fftdata / len(fftdata)
                frq = np.fft.rfftfreq(len(analogData.ax), d=1. / 93)[0:1000]  # remove dc value  and high frequency value
                self.fft.set_data(frq, fftdata)
                plt.xlim(0, 60)
                plt.ylim(0, 8)
            else:
                afterpass = butter_pass_filter(analogData.ax, cutfreq, 93, whatpass)
                data = copy.deepcopy(np.array(afterpass))
                self.axline.set_ydata(afterpass)
                self.axline.set_label(heartrate)
                #plt.xlim(0, 1000)
                plt.ylim([np.min(list(afterpass)[300:500]) - 10, np.max(list(afterpass)[300:500]) + 10])
                plt.legend(loc='upper right')
                """fourier"""
                plt.subplot(412)
                fftdata = np.abs(np.fft.rfft(afterpass))[0:1000]  # remove dc value  and high frequency value
                fftdata = fftdata/len(fftdata)
                frq = np.fft.rfftfreq(len(afterpass), d=1. / 93)[0:1000]  # remove dc value  and high frequency value
                self.fft.set_data(frq, fftdata)
                plt.ylim([np.min(list(fftdata)[10:500]) - 10, np.max(list(fftdata)[10:500]) + 10])
                plt.xlim(0, 60)
            plt.draw()
            plt.pause(0.001)
        thread2 = None
radioloc = plt.axes([0., 0.15, 0.15, 0.15])
radio = RadioButtons(radioloc, (['no', 'high', 'low']))
radio.on_clicked(change_pass)
labarloc = plt.axes([0.3, 0.2, 0.65, 0.03])
labar = Slider(labarloc,'freqlimit', 0.1, 4, valinit=2)
labar.on_changed(cutfreq_change)
btnloc = plt.axes([0.7, 0.05, 0.1, 0.075])
button = Button(btnloc, 'store')
btnloc2 = plt.axes([0.1, 0.05, 0.1, 0.075])
button2 = Button(btnloc2, 'plot')
btnloc3 = plt.axes([0.4, 0.05, 0.1, 0.075])
button3 = Button(btnloc3, 'clear')
button.on_clicked(store)
button2.on_clicked(loadfile)
button3.on_clicked(clear)

# main() function
def main():
    global thread1,thread2
    # expects 1 arg - serial port string
    #  if(len(sys.argv) != 2):
    #    print 'Example usage: python showdata.py "/dev/tty.usbmodem411"'
    #    exit(1)

    # strPort = '/dev/tty.usbserial-A7006Yqh'
    # strPort = sys.argv[1];
    strPort = 'com8'
    # plot parameters
    analogData = AnalogData(1000)
    analogPlot = AnalogPlot(analogData)

    print('plotting data...')

    # open serial port
    ser = serial.Serial(strPort, 115200)
    print(ser)
    thread1 = None
    thread2 = None
    last = time.time()
    while True:
        try:
            line = ser.readline()
            # print(line)
            '''
            hp = np.roll(hp, 1)
            hp[0] = float(line)
            print(np.mean(hp))
            '''
            for val in line.split():
                # print (float(val))

                data = [float(val)]

                #print (data)

            if (len(data) == 1):
                thread1 = td.Thread(target=analogData.add(data))
                thread1.start()
                #analogData.add(data)
                current = time.time()
                if (current-last)>0.01:
                    thread2 = td.Thread(target=analogPlot.update(analogData))
                    thread2.start()
                    last = current
                #analogPlot.update(analogData)
        except KeyboardInterrupt:
            print('exiting')
            break
    # close serial
    ser.flush()
    ser.close()


# call main
if __name__ == '__main__':
    thread = mp.Process(target=main)
    thread.start()
    #main()