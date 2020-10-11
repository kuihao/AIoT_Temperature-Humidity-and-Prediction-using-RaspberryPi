import Adafruit_DHT
import csv
import datetime
import math
import requests
import sys
import time
'''
function: write [datetime, temperature, humidity] in csv
ipt: dhtwriter, temperature, humidity
opt: none
'''
def WriteInCSV(dhtwriter, temperature, humidity):
    dhtwriter.writerow([datetime.datetime.now().strftime('%y %m %d %H-%M-%S'),
        '{:0.1f}'.format(temperature),
        '{:0.1f}'.format(humidity)])

# set Adafruit_DHT's varibles 
sensor = Adafruit_DHT.DHT11
pin = 4

# write in csv with append mode
with open('dht11_record.csv', 'a') as dhtfile:
    # create a csv writer object
    dhtwriter = csv.writer(dhtfile, dialect='excel')
    # get sencor data from DHT11
    try:
        print('[INFO]press Ctrl-C to abort the process')
        while True:
            # Get sensor data.  Use the read_retry method which will retry up
            # to 15 times to get a sensor reading (waiting 2 seconds between each retry).
            humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
            # Checking sensor dedect correctly
            if humidity is not None and temperature is not None:
                # function: write [datetime, temperature, humidity] in csv
                WriteInCSV(dhtwriter, temperature, humidity)
                print('[INFO]Temperature={0:0.1f}*C Humidity={1:0.1f}%'.format(temperature, humidity))
            else:
                print('[ERROR]Failed to get reading. Try again!')
            time.sleep(5)
    except KeyboardInterrupt:
        print('[INFO]abort')