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
    print('按下 Ctrl-C 可停止程式')
    while True:
        humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT11, GPIO_PIN)
        # Try to grab a sensor reading.  Use the read_retry method which will retry up
        # to 15 times to get a sensor reading (waiting 2 seconds between each retry).
        if humidity is not None and temperature is not None:
            print('溫度={0:0.1f}度C 濕度={1:0.1f}%'.format(temperature, humidity))
        else:
            print('讀取失敗，重新讀取。')
        time.sleep(10)
except KeyboardInterrupt:
    print('關閉程式')