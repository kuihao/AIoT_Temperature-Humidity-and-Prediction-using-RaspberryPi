import sys
import Adafruit_DHT
import time
import requests
import math

# set Adafruit_DHT's varibles 
sensor = Adafruit_DHT.DHT11
pin = 4

# get sencor data from DHT11
try:
    print('press Ctrl-C to abort the process')
    while True:
        humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT11, GPIO_PIN)
        # Try to grab a sensor reading.  Use the read_retry method which will retry up
        # to 15 times to get a sensor reading (waiting 2 seconds between each retry).
        if humidity is not None and temperature is not None:
            print('Temperature={0:0.1f}*C Humidity={1:0.1f}%'.format(temperature, humidity))
        else:
            print('Failed to get reading. Try again!')
        time.sleep(10)
except KeyboardInterrupt:
    print('abort')