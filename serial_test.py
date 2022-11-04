#!/usr/bin/env python3
import json
import serial
import time

def get_temp():
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    ser.reset_input_buffer()
    #while True:
    #ser.write(b"Hello from Raspberry Pi!\n")
    while True:
        line = ser.readline().decode('utf-8').rstrip()
        if line:
            #print('no')
            json_obj = json.loads(line)
            return json_obj["ObjectTemp"]
            #time.sleep(1)

print(get_temp())

#if __name__ == '__main__':
#    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
#    ser.reset_input_buffer()
#    while True:
#        #ser.write(b"Hello from Raspberry Pi!\n")
#        line = ser.readline().decode('utf-8').rstrip()
#        if line:
#            json_obj = json.loads(line)
#            print(json_obj["ObjectTemp"])
#            time.sleep(1)
