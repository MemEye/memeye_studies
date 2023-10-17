import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
from threading import Thread
import readchar
import datetime
import time
import csv
import pandas as pd
import os

latest_osc_data = None
last_collect_time = time.time()
current_annotation = ''
collect_data = False
test_output = False

sensor_buffers = {
    "/EmotiBit/0/PPG:RED": [],
    "/EmotiBit/0/PPG:IR": [], 
    "/EmotiBit/0/PPG:GRN": [], 
    "/EmotiBit/0/EDA": [],
    "/EmotiBit/0/HUMIDITY": [], 
    "/EmotiBit/0/ACC:X": [], 
    "/EmotiBit/0/ACC:Y": [], 
    "/EmotiBit/0/ACC:Z": [], 
    "/EmotiBit/0/GYRO:X": [], 
    "/EmotiBit/0/GYRO:Y": [], 
    "/EmotiBit/0/GYRO:Z": [], 
    "/EmotiBit/0/MAG:X": [], 
    "/EmotiBit/0/MAG:Y": [], 
    "/EmotiBit/0/MAG:Z": [], 
    "/EmotiBit/0/THERM": [], 
    "/EmotiBit/0/TEMP0": [],
    "/EmotiBit/0/TEMP1": [],
    "/EmotiBit/0/EDL": [],
    "/EmotiBit/0/ER": [],
    "/EmotiBit/0/SA": [],
    "/EmotiBit/0/SR": [],
    "/EmotiBit/0/SF": [],
    "/EmotiBit/0/HR": [],
    "/EmotiBit/0/BI": [],
    "/EmotiBit/0/HUMID": [],
}

data_log = []

def setup_dispatcher():
    dispatch = dispatcher.Dispatcher()

    dispatch.map("/EmotiBit/0/PPG:RED", filter_handler, 'PPG_RED')
    dispatch.map("/EmotiBit/0/PPG:IR", filter_handler, 'PPG_IR')
    dispatch.map("/EmotiBit/0/PPG:GRN", filter_handler, 'PPG_GRN')
    dispatch.map("/EmotiBit/0/EDA", filter_handler, 'EDA')
    dispatch.map("/EmotiBit/0/HUMIDITY", filter_handler, 'HUMIDITY')
    dispatch.map("/EmotiBit/0/ACC:X", filter_handler, 'ACC_X')
    dispatch.map("/EmotiBit/0/ACC:Y", filter_handler, 'ACC_Y')
    dispatch.map("/EmotiBit/0/ACC:Z", filter_handler, 'ACC_Z')
    dispatch.map("/EmotiBit/0/GYRO:X", filter_handler, 'GYRO_X')
    dispatch.map("/EmotiBit/0/GYRO:Y", filter_handler, 'GRYO_Y')
    dispatch.map("/EmotiBit/0/GYRO:Z", filter_handler, 'GYRO_Z')
    dispatch.map("/EmotiBit/0/MAG:X", filter_handler, 'MAG_X')
    dispatch.map("/EmotiBit/0/MAG:Y", filter_handler, 'MAG_Y')
    dispatch.map("/EmotiBit/0/MAG:Z", filter_handler, 'MAG_Z')
    dispatch.map("/EmotiBit/0/THERM", filter_handler, 'THERM')
    dispatch.map("/EmotiBit/0/TEMP0", filter_handler, 'TEMP0')
    dispatch.map("EmotiBit/0/TEMP1", filter_handler, 'TEMP0')
    dispatch.map("/EmotiBit/0/EDL", filter_handler, 'TEMP0')
    dispatch.map("/EmotiBit/0/ER", filter_handler, 'ER')
    dispatch.map("/EmotiBit/0/SA", filter_handler, 'SA')
    dispatch.map("/EmotiBit/0/SR", filter_handler, 'SR')
    dispatch.map("/EmotiBit/0/SF", filter_handler, 'SF')
    dispatch.map("/EmotiBit/0/HR", filter_handler, 'HR')
    dispatch.map("/EmotiBit/0/BI", filter_handler, 'BI')
    dispatch.map("/EmotiBit/0/HUMID", filter_handler, 'HUMID')

    return dispatch


def filter_handler(unused_addr, *args):
    """ Handle incoming OSC messages """

    global latest_osc_data
    global last_collect_time
    global data_log
    global current_annotation
    global sensor_buffers
    global collect_data
    global test_output

    current_time = time.time()

    # Append to the relevant buffer
    sensor_buffers[unused_addr].append(args)

    # If the interval has passed, process the buffers
    if current_time - last_collect_time > BUFFER_INTERVAL and collect_data:
        group_data = []
        timestamp = datetime.datetime.now().isoformat()
        for address, buffer in sensor_buffers.items():
            # Take the latest value (or None if no data)
            value = buffer[-1] if buffer else None
            group_data.append(value)
            # print(group_data)
            buffer.clear()  # Clear buffer for next interval
        # Append the grouped data with a timestamp to data_log
        group_data.append((['LABEL'], current_annotation))
        group_data.append((['TIMESTAMP'], timestamp))
        data_log.append(group_data)
        last_collect_time = current_time
        latest_osc_data = f"data: {group_data}"
    elif test_output:
        print(f'data test: {args}')
        test_output = False

def format_and_save_data(data_log):
    global save_location
    global save_name

    dict_data = []
    for row in data_log:
        row_dict = {}
        for item in row:
            if item is None:
                continue
            label = item[0][0]  # Extract the label
            values = item[1:]   # Get the data values
            for i, value in enumerate(values):
                col_name = f"{label}_{i+1}"  # Construct the column name
                row_dict[col_name] = value
        dict_data.append(row_dict)
    df = pd.DataFrame(dict_data)
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    # Create the full path to the file
    full_path = os.path.join(save_location, save_name)

    # Save the DataFrame to a CSV file
    df.to_csv(full_path, index=False)


def main(ip, port):
    global collect_data
    global current_annotation
    global test_output
    global data_log

    dispatch = setup_dispatcher()
    # Start OSC server
    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatch)
    print("Serving on {}".format(server.server_address))
    # server.serve_forever()

    server_thread = Thread(target=server.serve_forever)
    server_thread.start()

    while True:
        key = readchar.readkey()
        key = key.lower()

        if key == 't':
            test_output = True
            print("server output")

        if key == 'r':
            print('starting recording session')
            collect_data = True
        
        if key == 's':
            print('stopping recording session and saving')
            if current_annotation != '':
                current_annotation = ''
            
            collect_data = False
            format_and_save_data(data_log)
            data_log = []

        if key == 'p':
            print('ending this annotation label')
            current_annotation = ''

        if key == 'a':
            pass
            print('input the annotation label:')
            current_annotation = str(input())

        if key == 'e':
            print('confirm exit: (Y/N)')
            confirm = str(input())
            if confirm.lower() == 'y':
                print("exiting")
                current_annotation = ''
                server.shutdown()
                format_and_save_data(data_log)
                break
            elif confirm.lower() == 'n':
                continue
            else:
                print('invalid entry, please try again (Y/N):')
                confirm = str(input())

if __name__=='__main__':
    BUFFER_INTERVAL = 0.01  # 100hz
    save_location = './data_test'
    save_name = 'record_1.csv'

    PORT_NUMBER = 12345
    IP_DEFAULT = "127.0.0.1"
    main(IP_DEFAULT, PORT_NUMBER)

