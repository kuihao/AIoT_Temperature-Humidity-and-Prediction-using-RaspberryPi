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

# set Ubidots uploading variables
TOKEN = "<ubidots-token>"  # Put your TOKEN here
DEVICE_LABEL = "Raspi_DHT11"  # Put your device label here 
VARIABLE_LABEL_1 = "temperature"  # Put your first variable label here
VARIABLE_LABEL_2 = "humidity"  # Put your second variable label here

'''
function: build payload(the essencial datas need upload to ubidots)
ipt: VARIABLE_LABEL_1, VARIABLE_LABEL_2, temperature, humidity
opt: payload object in dict
'''
def build_payload(variable_1, variable_2, var_temperature, var_humidity): 
    payload = {variable_1: var_temperature,
               variable_2: var_humidity}
    return payload

'''
function: exec post_request (to ubidots)  
ipt: payload
opt: none
'''
def post_request(payload):
    # Creates the headers for the HTTP requests
    url = "http://industrial.api.ubidots.com"
    url = "{}/api/v1.6/devices/{}".format(url, DEVICE_LABEL)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    # Makes the HTTP requests
    status = 400
    attempts = 0
    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)

    # Processes results
    if status >= 400:
        print("[ERROR] Could not send data after 5 attempts, please check \
            your token credentials and internet connection")
        return False

    print("[INFO] request made properly, your device is updated")
    return True

# set Adafruit_DHT's varibles 
sensor = Adafruit_DHT.DHT11
pin = 4

'''
Main Execting codes
'''
# write in csv with append mode
with open('dht11_record.csv', 'a') as dhtfile:
    # create a csv writer object
    dhtwriter = csv.writer(dhtfile, dialect='excel')
    # get sencor data from DHT11
    try:
        print('[INFO]press Ctrl-C to abort the process.')
        while True:
            # Get sensor data.  Use the read_retry method which will retry up
            # to 15 times to get a sensor reading (waiting 2 seconds between each retry).
            humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
            # Checking sensor dedect correctly
            if humidity is not None and temperature is not None:
                print('[SENSOR]Temperature={0:0.1f}*C Humidity={1:0.1f}%'.format(temperature, humidity))
                # function: write [datetime, temperature, humidity] in csv
                WriteInCSV(dhtwriter, temperature, humidity)
                # function: build ubidots payload
                payload = build_payload(
                    VARIABLE_LABEL_1, VARIABLE_LABEL_2, temperature, humidity)
                print("[INFO] Attemping to send data.")
                # function: exec post_request
                post_request(payload)
                print("[INFO] Post finished.")
            else:
                print('[ERROR]Failed to get reading. Try again!')
            time.sleep(3600)
    except KeyboardInterrupt:
        print('[INFO]Abort!')