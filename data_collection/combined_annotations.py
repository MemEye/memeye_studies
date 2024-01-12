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
from psychopy import visual, core, event
from threading import Thread
import os

EMOTIBIT_BUFFER_INTERVAL = 0.01  # 100hz
emotibit_save_location = './data_test'
record_num = 0
emotibit_latest_osc_data = None
emotibit_last_collect_time = time.time()
current_annotation = ''
emotibit_collect_data = False
emotibit_test_output = False
emotibit_data_log = []
pupil_time_align_val = None


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
    global pupil_time_align_val

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
        if pupil_time_align_val != None:
            group_data.append((['PUPIL_TIME'], pupil_time_align_val))
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


def check_capture_exists(ip_address, port):
    """check pupil capture instance exists"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if not sock.connect_ex((ip_address, port)):
            print("Found Pupil Capture")
        else:
            print("Cannot find Pupil Capture")
            sys.exit()


def setup_pupil_remote_connection(ip_address, port):
    """Creates a zmq-REQ socket and connects it to Pupil Capture or Service
    to send and receive notifications.

    We also set up a PUB socket to send the annotations. This is necessary to write
    messages to the IPC Backbone other than notifications

    See https://docs.pupil-labs.com/developer/core/network-api/ for details.
    """
    # zmq-REQ socket
    ctx = zmq.Context.instance()
    pupil_remote = ctx.socket(zmq.REQ)
    pupil_remote.connect(f"tcp://{ip_address}:{port}")

    # PUB socket
    pupil_remote.send_string("PUB_PORT")
    pub_port = pupil_remote.recv_string()
    pub_socket = zmq.Socket(ctx, zmq.PUB)
    pub_socket.connect("tcp://127.0.0.1:{}".format(pub_port))

    return pupil_remote, pub_socket


def request_pupil_time(pupil_remote):
    """Uses an existing Pupil Core software connection to request the remote time.
    Returns the current "pupil time" at the timepoint of reception.
    See https://docs.pupil-labs.com/core/terminology/#pupil-time for more information
    about "pupil time".
    """
    pupil_remote.send_string("t")
    pupil_time = pupil_remote.recv()
    return float(pupil_time)


def measure_clock_offset(pupil_remote, clock_function):
    """Calculates the offset between the Pupil Core software clock and a local clock.
    Requesting the remote pupil time takes time. This delay needs to be considered
    when calculating the clock offset. We measure the local time before (A) and
    after (B) the request and assume that the remote pupil time was measured at (A+B)/2,
    i.e. the midpoint between A and B.

    As a result, we have two measurements from two different clocks that were taken
    assumingly at the same point in time. The difference between them ("clock offset")
    allows us, given a new local clock measurement, to infer the corresponding time on
    the remote clock.
    """
    local_time_before = clock_function()
    pupil_time = request_pupil_time(pupil_remote)
    local_time_after = clock_function()

    local_time = (local_time_before + local_time_after) / 2.0
    clock_offset = pupil_time - local_time
    return clock_offset


def measure_clock_offset_stable(pupil_remote, clock_function, n_samples=10):
    """Returns the mean clock offset after multiple measurements to reduce the effect
    of varying network delay.

    Since the network connection to Pupil Capture/Service is not necessarily stable,
    one has to assume that the delays to send and receive commands are not symmetrical
    and might vary. To reduce the possible clock-offset estimation error, this function
    repeats the measurement multiple times and returns the mean clock offset.

    The variance of these measurements is expected to be higher for remote connections
    (two different computers) than for local connections (script and Core software
    running on the same computer). You can easily extend this function to perform
    further statistical analysis on your clock-offset measurements to examine the
    accuracy of the time sync.
    """
    assert n_samples > 0, "Requires at least one sample"
    offsets = [
        measure_clock_offset(pupil_remote, clock_function) for x in range(n_samples)
    ]
    return sum(offsets) / len(offsets)  # mean offset


def notify(pupil_remote, notification):
    """Sends ``notification`` to Pupil Remote"""
    topic = "notify." + notification["subject"]
    payload = serializer.dumps(notification, use_bin_type=True)
    pupil_remote.send_string(topic, flags=zmq.SNDMORE)
    pupil_remote.send(payload)
    return pupil_remote.recv_string()


def send_trigger(pub_socket, trigger):
    """Sends annotation via PUB port"""
    payload = serializer.dumps(trigger, use_bin_type=True)
    pub_socket.send_string(trigger["topic"], flags=zmq.SNDMORE)
    pub_socket.send(payload)


def new_trigger(label, duration, timestamp):
    """Creates a new trigger/annotation to send to Pupil Capture"""
    return {
        "topic": "annotation",
        "label": label,
        "timestamp": timestamp,
        "duration": duration,
    }

def collect_sensor_data(emotibit_ip, emotibit_port):
    global emotibit_collect_data
    global current_annotation
    global emotibit_test_output
    global emotibit_data_log
    global pupil_time_align_val
    
    emotibit_server, emotibit_dispatch = emotibit_server_thread(emotibit_ip, emotibit_port)

    pupil_ip, pupil_port = "127.0.0.1", 50020
    emotibit_thread = Thread(target=emotibit_server.serve_forever)
    emotibit_thread.start()

        # 1. Setup network connection
    check_capture_exists(pupil_ip, pupil_port)
    pupil_remote, pub_socket = setup_pupil_remote_connection(pupil_ip, pupil_port)

    # 2. Setup local clock function
    local_clock = time.perf_counter

    # 3. Measure clock offset accounting for network latency
    stable_offset_mean = measure_clock_offset_stable(
        pupil_remote, clock_function=local_clock, n_samples=10
    )

    pupil_time_actual = request_pupil_time(pupil_remote)
    local_time_actual = local_clock()
    pupil_time_calculated_locally = local_time_actual + stable_offset_mean
    print(f"Pupil time actual: {pupil_time_actual}")
    print(f"Local time actual: {local_time_actual}")
    print(f"Stable offset: {stable_offset_mean}")
    print(f"Pupil time (calculated locally): {pupil_time_calculated_locally}")

    duration = 0.0

    # 4. Prepare and send annotations
    # Start the annotations plugin
    notify(
        pupil_remote,
        {"subject": "start_plugin", "name": "Annotation_Capture", "args": {}},
    )

    print("Enter 'R' to start recording")
    print("Enter 'S' to stop recording")
    print("Enter 'A' to start a recognition annotation")
    print("Enter 'B' to start a recall annotation")
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
            pupil_time_align_val = request_pupil_time(pupil_remote)
            emotibit_collect_data = True
            pupil_remote.send_string("R")
            pupil_remote.recv_string()
        
        if key == 's':
            print('stopping recording session and saving')
            if current_annotation != '':
                local_time = local_clock()
                minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
                send_trigger(pub_socket, minimal_trigger)

                # Add custom keys to your annotation
                minimal_trigger["custom_key"] = "custom value"
                send_trigger(pub_socket, minimal_trigger)
                current_annotation = ''
            
            pupil_time_align_val = request_pupil_time(pupil_remote)
            emotibit_collect_data = False
            emotibit_save_data(emotibit_data_log)
            emotibit_data_log = []
            pupil_remote.send_string("r")
            pupil_remote.recv_string()

        if key == 'p':
            print('ending this annotation label')
            pupil_time_align_val = request_pupil_time(pupil_remote)
            minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            current_annotation = ''
            

        if key == 'a':
            local_time = local_clock()
            # print('input the annotation label:')
            # current_annotation = str(input())
            current_annotation = 'recognition'
            duration = 0.0 #TODO: look into this value
            minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            pupil_time_align_val = request_pupil_time(pupil_remote)

        if key == 'b':
            local_time = local_clock()
            current_annotation = 'recall'
            duration = 0.0 #TODO: look into this value
            minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            pupil_time_align_val = request_pupil_time(pupil_remote)

        if key == 'e':
            print('confirm exit: (Y/N)')
            confirm = str(input())
            if confirm.lower() == 'y':
                print("exiting")
                current_annotation = ''
                emotibit_server.shutdown()

                pupil_time_align_val = request_pupil_time(pupil_remote)
                emotibit_save_data(emotibit_data_log)
                local_time = local_clock()
                minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
                send_trigger(pub_socket, minimal_trigger)
                # Add custom keys to your annotation
                minimal_trigger["custom_key"] = "custom value"
                send_trigger(pub_socket, minimal_trigger)

                # stop recording
                pupil_remote.send_string("r")
                pupil_remote.recv_string()

                break
            elif confirm.lower() == 'n':
                continue
            else:
                print('invalid entry, please try again (Y/N):')
                confirm = str(input())

# if __name__=='__main__':
#     EMOTIBIT_BUFFER_INTERVAL = 0.01  # 100hz
#     emotibit_save_location = './data_test'
#     record_num = 0
#     # emotibit_save_name = f'record_{record_num}.csv'

#     EMOTIBIT_PORT_NUMBER = 12345
#     EMOTIBIT_IP_DEFAULT = "127.0.0.1"
#     sensor_thread = Thread(target=collect_sensor_data, args = (EMOTIBIT_IP_DEFAULT, EMOTIBIT_PORT_NUMBER))
#     sensor_thread.start()
#     # collect_sensor_data(EMOTIBIT_IP_DEFAULT, EMOTIBIT_PORT_NUMBER)

