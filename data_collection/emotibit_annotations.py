from pythonosc import dispatcher
from pythonosc import osc_server
from threading import Thread
import readchar
import datetime
import time
import csv
import pandas as pd
import os
import zmq
import msgpack as serializer
import time
import socket
import sys


emotibit_latest_osc_data = None
emotibit_last_collect_time = time.time()
current_annotation = ''
emotibit_collect_data = False
emotibit_test_output = False
emotibit_data_log = []
#TODO: add in pupillabs timestamps vars

emotibit_sensor_buffers = {
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

    global emotibit_latest_osc_data
    global emotibit_last_collect_time
    global emotibit_data_log
    global current_annotation
    global emotibit_sensor_buffers
    global emotibit_collect_data
    global emotibit_test_output
    global EMOTIBIT_BUFFER_INTERVAL

    current_time = time.time()

    # Append to the relevant buffer
    emotibit_sensor_buffers[unused_addr].append(args)

    # If the interval has passed, process the buffers
    if current_time - emotibit_last_collect_time > EMOTIBIT_BUFFER_INTERVAL and emotibit_collect_data:
        group_data = []
        timestamp = datetime.datetime.now().isoformat()
        for address, buffer in emotibit_sensor_buffers.items():
            # Take the latest value (or None if no data)
            value = buffer[-1] if buffer else None
            group_data.append(value)
            # print(group_data)
            buffer.clear()  # Clear buffer for next interval
        # Append the grouped data with a timestamp to data_log
        group_data.append((['LABEL'], current_annotation))
        group_data.append((['TIMESTAMP'], timestamp))
        emotibit_data_log.append(group_data)
        emotibit_last_collect_time = current_time
        emotibit_latest_osc_data = f"data: {group_data}"
    elif emotibit_test_output:
        print(f'data test: {args}')
        emotibit_test_output = False

def emotibit_save_data(data_log):
    global emotibit_save_location
    global record_num

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
    if not os.path.exists(emotibit_save_location):
        os.makedirs(emotibit_save_location)

    # Create the full path to the file
    full_path = os.path.join(emotibit_save_location, f'record_{record_num}.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(full_path, index=False)
    record_num += 1

def emotibit_server_thread(emotibit_ip, emotibit_port):
    dispatch = setup_dispatcher()
    # Start OSC server
    server = osc_server.ThreadingOSCUDPServer((emotibit_ip, emotibit_port), dispatch)
    print("Emotibit serving on {}".format(server.server_address))
    return server, dispatch


def main(emotibit_ip, emotibit_port):
    global emotibit_collect_data
    global current_annotation
    global emotibit_test_output
    global emotibit_data_log
    
    emotibit_server, emotibit_dispatch = emotibit_server_thread(emotibit_ip, emotibit_port)
    emotibit_thread = Thread(target=emotibit_server.serve_forever)
    emotibit_thread.start()

    print("Enter 'R' to start recording")
    print("Enter 'S' to stop recording")
    print("Enter 'A' to start an annotation")
    print("Enter 'P' to stop an annotation")
    print("Enter 'E' to exit the tool")
    print("Enter 'T' to test the emotibit output")

    while True:
        key = readchar.readkey()
        key = key.lower()

        if key == 't':
            emotibit_test_output = True
            print("server output")

        if key == 'r':
            print('starting recording session')
            emotibit_collect_data = True
        
        if key == 's':
            print('stopping recording session and saving')
            if current_annotation != '':
                current_annotation = ''
            
            emotibit_collect_data = False
            emotibit_save_data(emotibit_data_log)
            emotibit_data_log = []

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
                emotibit_server.shutdown()
                emotibit_save_data(emotibit_data_log)
                break
            elif confirm.lower() == 'n':
                continue
            else:
                print('invalid entry, please try again (Y/N):')
                confirm = str(input())


if __name__=='__main__':
    EMOTIBIT_BUFFER_INTERVAL = 0.01  # 100hz
    emotibit_save_location = './data_test'
    record_num = 0
    # emotibit_save_name = f'record_{record_num}.csv'

    EMOTIBIT_PORT_NUMBER = 12345
    EMOTIBIT_IP_DEFAULT = "127.0.0.1"
    main(EMOTIBIT_IP_DEFAULT, EMOTIBIT_PORT_NUMBER)

