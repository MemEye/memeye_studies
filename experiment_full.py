from pythonosc import dispatcher, osc_server
from threading import Thread
import datetime
import time
import pandas as pd
import os
import zmq
import msgpack as serializer
import socket
import sys
from psychopy import visual, core, event, monitors
import os
import random
import json
import numpy as np
from PIL import Image

# VARIABLES THAT CAN CHANGE - ADJUST THESE TO CHANGE THE EXPERIMENT

#TODO: play around with sensors more

#TODO: adjust psychopy window on lab comp

#TODO: 48 choose 32 for each particpate 16 choose 12

#TODO: remove exp 2, remove facts phase

#TODO: 1 min game, 1 min chill/move

#TODO: see where to reduce the number of screens

EMOTIBIT_BUFFER_INTERVAL = 0.02  # 50hz, fastest datastream is 25Hz, can probably do 0.04
data_save_location = './data'
subject_id = 'test'
experiment_num = 1
subject_save_location = os.path.join(data_save_location, subject_id)
#TODO: have date/time included in save location to prevent overwritting


learning_time = 7  # Time each image is shown during learning phase (in seconds)
recognition_time = 5
break_time = 3  # Time between images during learning phase (in seconds)
recall_time = 10  # Max time for each image during remembering phase (in seconds)

exp_1_shown_images_dir = './experiment_1_images/people/shown/'
exp_1_extra_images_dir = './experiment_1_images/people/extra/'
exp_1_practice_images_dir = './experiment_1_images/people/practice/'

exp_2_shown_images_dir = './experiment_2_images/people/shown/'
exp_2_extra_images_dir = './experiment_2_images/people/extra/'
exp_2_practice_images_dir = './experiment_2_images/people/practice/'


mon = monitors.Monitor('testMonitor')  # Replace 'testMonitor' with the name of your monitor
screen_width, screen_height = mon.getSizePix()
print(mon.getSizePix())
window_height = screen_height - 50  # Adjust this value to leave space for the taskbar/dock
win = visual.Window(
    size=(1350, 740), 
    pos=(0, 25),  # This centers the window vertically. Adjust as needed.
    fullscr=False,  # Fullscreen is set to False
    screen=0,
    color=[0, 0, 0]
)
print(win.size)
# win = visual.Window(fullscr=False, color=[0, 0, 0])
noise_texture = np.random.normal(loc=0.5, scale=0.3, size=(win.size[1], win.size[0])) # loc is the mean, scale is the standard deviation

# Normalize the noise texture to be within the range [0, 1], as expected by PsychoPy
noise_texture = (noise_texture - noise_texture.min()) / (noise_texture.max() - noise_texture.min())
# Create an image stimulus from the RGB noise
noise_stim = visual.ImageStim(win, image=noise_texture, size = win.size, units='pix')

record_num = 0
emotibit_latest_osc_data = None
emotibit_last_collect_time = time.time()
current_annotation = ''
emotibit_collect_data = False
emotibit_test_output = False
emotibit_data_log = []
pupil_time_align_val = None
curr_image = ''
subject_response = ''

start_recording = False
stop_recording = False
send_annotation_to_pupil = False
exit_sensors = False
bookend_annotation = False
send_subject_response = False

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
    global curr_image
    global subject_response

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
            pupil_time_align_val = None
        group_data.append((['IMAGE'], curr_image))
        group_data.append((['SUBJECT_RESPONSE'], subject_response))
        if subject_response != '':
            subject_response = ''
        emotibit_data_log.append(group_data)
        emotibit_last_collect_time = current_time
        emotibit_latest_osc_data = f"data: {group_data}"
    elif emotibit_test_output:
        print(f'data test: {args}')
        emotibit_test_output = False

def emotibit_save_data(data_log):
    global subject_save_location
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
    full_path = os.path.join(subject_save_location, f'experiment_{experiment_num}')
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Create the full path to the file
    file_path = os.path.join(full_path,f'record_{record_num}.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
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
    #SUB SOCKET
    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()
    sub_socket = zmq.Socket(ctx, zmq.SUB)
    sub_socket.connect("tcp://127.0.0.1:{}".format(sub_port))

    return pupil_remote, pub_socket, sub_socket



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
    global start_recording
    global stop_recording
    global send_annotation_to_pupil
    global exit_sensors
    global curr_image
    global bookend_annotation
    global send_subject_response
    
    emotibit_server, emotibit_dispatch = emotibit_server_thread(emotibit_ip, emotibit_port)

    pupil_ip, pupil_port = "127.0.0.1", 50020
    emotibit_thread = Thread(target=emotibit_server.serve_forever)
    emotibit_thread.start()

        # 1. Setup network connection
    check_capture_exists(pupil_ip, pupil_port)
    pupil_remote, pub_socket, sub_socket = setup_pupil_remote_connection(pupil_ip, pupil_port)

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

    while True:
        if start_recording:
            print('starting recording session')

            emotibit_collect_data = True
            full_path = os.path.join(subject_save_location, f'experiment_{experiment_num}')
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            message = f'R {full_path}'
            pupil_remote.send_string(message)
            pupil_remote.recv_string()
            pupil_time_align_val = request_pupil_time(pupil_remote)
            print(pupil_time_align_val, 'time align')
            start_recording = False
        
        if stop_recording:
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
            print(pupil_time_align_val, 'time align')
            emotibit_collect_data = False
            emotibit_save_data(emotibit_data_log)
            emotibit_data_log = []
            pupil_remote.send_string("r")
            pupil_remote.recv_string()
            stop_recording = False
        
        if send_subject_response:
            local_time = local_clock()
            pupil_time_align_val = request_pupil_time(pupil_remote)
            print(pupil_time_align_val, 'time align')
            minimal_trigger = new_trigger(subject_response, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            send_subject_response = False
            
        if send_annotation_to_pupil:
            local_time = local_clock()
            print('sending this annotation label')
            pupil_time_align_val = request_pupil_time(pupil_remote)
            print(pupil_time_align_val, 'time align')
            minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)

            minimal_trigger = new_trigger(curr_image, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            
            pupil_time_align_val = request_pupil_time(pupil_remote)
            print(pupil_time_align_val, 'time align')
            send_annotation_to_pupil = False
            if bookend_annotation:
                current_annotation = ''
                curr_image = ''
                bookend_annotation = False

            
        if exit_sensors:
            print("exiting sensor streams")
            current_annotation = ''
            emotibit_server.shutdown()
            pupil_time_align_val = request_pupil_time(pupil_remote)
            print(pupil_time_align_val, 'time align')
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
            sub_socket.close()
            pupil_remote.close()
            pub_socket.close()
            exit_sensors = False


def learning_phase(images, practice = False):
    global win
    global noise_stim
    global learning_time
    global break_time
    global current_annotation
    global curr_image
    global send_annotation_to_pupil
    global bookend_annotation

    for img_path in images:
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Get the size of the window
        win_width, win_height = win.size

        # Calculate the scale factor for both dimensions
        scale_width = min((win_width * 0.2) / img_width, 1)
        scale_height = min((win_height * 0.2) / img_height, 1)

        # Use the smaller scale factor to ensure the image does not exceed 80% of the screen
        scale_factor = min(scale_width, scale_height)

        image = visual.ImageStim(win, image=img_path, size=(img_width * scale_factor, img_height * scale_factor), pos =(0,win_height*0.05), units = 'pix')
        text = "Do you recognize this face? \n \n (1: Yes, 2: No)"

        img_name = os.path.basename(img_path)  # Get the filename of the image
        text = f"Name: {images_to_info.get(img_name, '').get('Name')} \n \n Fact: {images_to_info.get(img_name, '').get('Fact')}"

        current_annotation = 'learning' if not practice else 'practice learning'
        curr_image = img_name
        send_annotation_to_pupil = True

        image.draw()
        text_stim = visual.TextStim(win, text=text, pos=(0, -0.5), color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        
        # Set different timing for different phases
        core.wait(learning_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()

        bookend_annotation = True
        send_annotation_to_pupil = True
        # Break between images (only during learning phase)
        noise_stim.draw()
        win.flip()
        core.wait(break_time)


def recognition_phase(shown_images, extra_images, repeats = False, ratio_shown = 1, practice = False):
    global win
    global noise_stim
    global recognition_time
    global break_time
    global current_annotation
    global curr_image
    global send_annotation_to_pupil
    global bookend_annotation
    global subject_response
    global send_subject_response


    images_to_show = random.sample(shown_images, int(len(shown_images)*ratio_shown))
    if repeats:
        images_to_show = images_to_show*2
    images = images_to_show + extra_images
    random.shuffle(images)

    for img_path in images:
        img_name = os.path.basename(img_path)

        img = Image.open(img_path)
        img_width, img_height = img.size

        # Get the size of the window
        win_width, win_height = win.size

        # Calculate the scale factor for both dimensions
        scale_width = min((win_width * 0.6) / img_width, 1)
        scale_height = min((win_height * 0.6) / img_height, 1)

        # Use the smaller scale factor to ensure the image does not exceed 80% of the screen
        scale_factor = min(scale_width, scale_height)

        image = visual.ImageStim(win, image=img_path, size=(img_width * scale_factor, img_height * scale_factor), units = 'pix')
        text = "Do you recognize this face? \n \n (1: Yes, 2: No)"

        image.draw()
        win.flip()

        current_annotation = 'recognition' if not practice else 'practice recognition'
        curr_image = img_name
        send_annotation_to_pupil = True
        
        # Wait for response or timeout
        timer = core.Clock()
        core.wait(recognition_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()

        
        text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        keys = event.waitKeys(keyList=['1', '2', 'num_1', 'num_2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            print(keys)
            if key == '1' or key == 'num_1':
                print('Yes')
                subject_response = 'Y'
                send_subject_response = True
            elif key == '2'or key == 'num_2':
                print('No')
                subject_response = 'N'
                send_subject_response = True
            if key == 'escape':
                core.quit()
        
        bookend_annotation = True
        send_annotation_to_pupil = True

        # Break between images (only during learning phase)
        noise_stim.draw()
        win.flip()
        core.wait(break_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()


def recall_phase(images_to_show, extra_images, recall_type, practice = False):
    global win
    global noise_stim
    global recall_time
    global break_time
    global current_annotation
    global curr_image
    global send_annotation_to_pupil
    global bookend_annotation
    global subject_response
    global send_subject_response
    global experiment_num

    images = images_to_show + extra_images
    random.shuffle(images)

    for img_path in images:
        img_name = os.path.basename(img_path)

        img = Image.open(img_path)
        img_width, img_height = img.size

        # Get the size of the window
        win_width, win_height = win.size

        # Calculate the scale factor for both dimensions
        scale_width = min((win_width * 0.6) / img_width, 1)
        scale_height = min((win_height * 0.6) / img_height, 1)

        # Use the smaller scale factor to ensure the image does not exceed 80% of the screen
        scale_factor = min(scale_width, scale_height)

        image = visual.ImageStim(win, image=img_path, size=(img_width * scale_factor, img_height * scale_factor), units = 'pix')

        text = ''
        
        if experiment_num == 2 and practice:
            if recall_type == 'name':
                text = "Use the following 10 seconds to try to recall the person's name in your mind. \n \n (Do not say out loud)"
            elif recall_type == 'fact':
                text = "Use the following 10 seconds to try to recall a fact about this celebrity in your mind. \n \n (Do not say out loud). \n \n Ex: They are an actor."
            elif recall_type == 'memory':
                text = "Use the following 10 seconds to try to recall a personal memory involving this celebrity in your mind. \n \n (Do not say out loud). \n \n Ex: I saw their movie with my friends in 2008."
        else:
            if recall_type == 'name':
                text = "Use the following 10 seconds to try to recall the person's name in your mind. \n \n (Do not say out loud)"
            elif recall_type == 'fact':
                text = "Use the following 10 seconds to try to recall a fact with this person in your mind. \n \n (Do not say out loud)"
            elif recall_type == 'memory':
                text = "Use the following 10 seconds to try to recall a personal memory with this person your mind. \n \n (Do not say out loud)"
        text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        if practice:
            core.wait(7)
        else:
            core.wait(5)
        
        image.draw()
        win.flip()

        current_annotation = f'recall {recall_type}' if not practice else f'practice recall {recall_type}'
        curr_image = img_name
        send_annotation_to_pupil = True
    
        # Wait for response or timeout
        timer = core.Clock()
        core.wait(recall_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()
        
        if recall_type == 'name':
            text = "Do you remember this person's name? \n \n (1: Yes, 2: No)"
        elif recall_type == 'fact':
            text = "Do you remember facts about this person? \n \n (1: Yes, 2: No)"
        elif recall_type == 'memory':
            text = "Do you have a memory involving this person? \n \n (1: Yes, 2: No)"
        text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        keys = event.waitKeys(keyList=['1', '2', 'num_1', 'num_2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            if key == '1' or key == 'num_1':
                subject_response = 'Y'
                send_subject_response = True
            elif key == '2'or key == 'num_2':
                subject_response = 'N'
                send_subject_response = True
            if key == 'escape':
                core.quit()
        
        current_annotation = f'recall {recall_type} verbal' if not practice else f'practice recall {recall_type} verbal'
        curr_image = img_name
        send_annotation_to_pupil = True

        if recall_type == 'name':
            text = "Now try your best to say the person's name out loud. \n \n It's ok if it's wrong or if you cannot recall it."
        elif recall_type == 'fact':
            text = "Now try your best to say the person's facts out loud. \n \n It's ok if it's wrong or if you cannot recall it."
        elif recall_type == 'memory':
            text = "Now try your best to say the memory out loud. \n \n It's ok if it's wrong or if you cannot recall it."

        text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        core.wait(recall_time)

        bookend_annotation = True
        send_annotation_to_pupil = True
        # Break between images (only during learning phase)
        noise_stim.draw()
        win.flip()
        core.wait(break_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()

def instructions(text):
    global win
    text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    event.waitKeys(keyList=['1', 'num_1'])

def game_break():
    global win
    global current_annotation
    global send_annotation_to_pupil
    global bookend_annotation
    text_stim = visual.TextStim(win, text="2 Minute Game Break", pos=(0,0), color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    current_annotation = f'game break'
    send_annotation_to_pupil = True
    core.wait(120)
    bookend_annotation = True
    send_annotation_to_pupil = True

def experiment_gui(exp_num):
    global exp_1_shown_images_dir
    global exp_1_extra_images_dir
    global exp_2_shown_images_dir
    global exp_2_extra_images_dir
    global win
    global start_recording
    global stop_recording
    global exit_sensors
    global noise_stim

    # Load images
    if exp_num == 1:
        shown_images = [os.path.join(exp_1_shown_images_dir, img) for img in os.listdir(exp_1_shown_images_dir) if img.endswith('.jpg')]
        extra_images = [os.path.join(exp_1_extra_images_dir, img) for img in os.listdir(exp_1_extra_images_dir) if img.endswith('.jpg')]
        practice_images = [os.path.join(exp_1_practice_images_dir, img) for img in os.listdir(exp_1_practice_images_dir) if img.endswith('.jpg')]
    if exp_num == 2:
        shown_images = [os.path.join(exp_2_shown_images_dir, img) for img in os.listdir(exp_2_shown_images_dir) if img.endswith('.jpg')]
        extra_images = [os.path.join(exp_2_extra_images_dir, img) for img in os.listdir(exp_2_extra_images_dir) if img.endswith('.jpg')]
        practice_images = [os.path.join(exp_2_practice_images_dir, img) for img in os.listdir(exp_2_practice_images_dir) if img.endswith('.jpg')]
    
    random.shuffle(shown_images)
    random.shuffle(extra_images)
    #TODO: change this
    # shown_images = shown_images[:2]
    # extra_images = []

    # Run experiment
    start_recording = True

    text = "Before we begin, please look at the screen and relax for two minutes. \n \n Press [1] to continue."
    instructions(text)
    noise_stim.draw()
    win.flip()
    core.wait(120)

    subtext = "" if exp_num == 1 else "famous people's "

    # practice phases are all here
    text = "We will now begin the practice section. \n \n Press [1] to continue."
    instructions(text)
    text = f"This study will consist of several sections that involve looking at images of {subtext}faces. \n \n You will be asked to try to remember as many details as you can and answer questions later. \n \n Press [1] to continue. "
    instructions(text)

    if exp_num == 1:
        text = f"We will now begin a practice learning phase of experiment {exp_num} out of 2. \n \n Press [1] to continue"
        instructions(text)
        text = "Instructions: \n \n You will be shown a sequence of images with the person's name and related facts. \n \n Please keep your attention on the screen and remember as many details as possible for each person. \n \n You will be tested on how much you remember after this. \n \n It will automatically move forward to the next part. \n \n Press [1] to continue."
        instructions(text)
        learning_phase(practice_images, practice = True)
        text = "End of practice learning phase. \n \n Take a quick break, ask any questions. \n \n Press [1] to continue."
    
    #practice recognition phase
    instructions(f"We will now begin a practice recognition phase of experiment {exp_num} out of 2. \n \n Press [1] to continue")
    text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    recognition_phase(practice_images, [], repeats = False, ratio_shown = 1, practice=True)
    instructions('End of practice recognition phase. \n \n Take a quick break, ask any questions. \n \n Press [1] to continue.')

    #practice names phase
    instructions(f"We will now begin a practice names phase of experiment {exp_num} out of 2. \n \n Press [1] to continue")
    text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    recall_phase(practice_images, [], 'name', practice=True)
    instructions('End of practice names phase. \n \n  Take a quick break, ask any questions. \n \n Press [1] to continue.')

    #practice facts phase
    instructions(f"We will now begin a practice facts phase of experiment {exp_num} out of 2. \n \n Press [1] to continue")
    text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    recall_phase(practice_images, [], 'fact', practice = True)
    instructions('End of practice facts phase. \n \n Take a quick break, ask any questions. \n \n Press [1] to continue.')

    if exp_num == 2:
        instructions(f"We will now begin a practice memory phase of experiment {exp_num} out of 2. \n \n Press [1] to continue")
        text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n You will be asked to recount a personal memory involving this person. \n \n Press [1] to continue."
        instructions(text)
        recall_phase(practice_images, [], 'memory', practice = True)
        instructions('End of practice memory phase. \n \n Take a quick break, ask any questions. \n \n Press [1] to continue.')

    text = "End of practice sections. \n \n If you have questions, please ask the researcher. \n \n Press [1] to continue"
    instructions(text)
    
    text = "We will now begin the main experiment. \n \n Press [1] to continue."
    instructions(text)

    if exp_num == 1:
        # Phase 1: Learning
        text = f"We will now begin the learning phase of experiment {exp_num} out of 2. \n \n Press [1] to continue"
        instructions(text)
        text = "Instructions: \n \n You will be shown a sequence of images with the person's name and related facts. \n \n Please keep your attention on the screen and remember as many details as possible for each person. \n \n You will be tested on how much you remember after this. \n \n It will automatically move forward to the next part. \n \n Press [1] to continue."
        instructions(text)
        learning_phase(shown_images)
        instructions('End of learning phase. \n \n Press [1] to continue to the game break.')
        game_break()
        instructions("End of game break. \n \n Press [1] to continue to the recognition phase")
   
    # Phase 2: Recognition  
    text = f"We will now begin the recognition phase of experiment {exp_num} out of 2. \n \n Press [1] to continue"
    instructions(text)
    text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    if exp_num == 1:
        recognition_phase(shown_images, extra_images, repeats = False, ratio_shown = 1)
    else:
        recognition_phase(shown_images, extra_images, repeats = False, ratio_shown = 1)
    instructions('End of recognition phase. \n \n Press [1] to continue to the game break.')

    game_break()

    # Phase 3: Names
    instructions("End of game break. \n \n Press [1] to continue to the names phase")
    text = f"We will now begin the names phase of experiment {exp_num} out of 2. \n \n Press [1] to continue"
    instructions(text)
    text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    recall_phase(shown_images, extra_images, 'name')
    instructions('End of names phase. \n \n Press [1] to continue to the game break.')
    game_break()

    # Phase 4: Facts
    instructions("End of game break. \n \n Press [1] to continue to the facts phase")
    text = f"We will now begin the facts phase of experiment {exp_num} out of 2. \n \n Press [1] to continue"
    instructions(text)
    text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    recall_phase(shown_images, extra_images, 'fact')

    # Phase 5: Memory
    if exp_num == 2:
        instructions('End of facts phase. \n \n Press [1] to continue to the game break.')
        game_break()
        instructions("End of game break. \n \n Press [1] to continue to the memory phase")
        text = f"We will now begin the memory phase of experiment {exp_num} out of 2. \n \n Press [1] to continue"
        instructions(text)
        text = f"Instructions: \n \n You will be shown a sequence of {subtext}images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n You will be asked to recount a personal memory involving this person. \n \n Press [1] to continue."
        instructions(text)
        recall_phase(shown_images, extra_images, 'memory')

    instructions(f"We have now completed experiment {exp_num} out of 2. \n \n Press [1] to exit")
    exit_sensors = True
    win.close()
    core.quit()

if __name__=='__main__':
    image_info_path = './experiment_1_names_facts.json'
    with open(image_info_path, 'r') as file:
        images_to_info = json.load(file)

    # EMOTIBIT_PORT_NUMBER = 12345
    # EMOTIBIT_IP_DEFAULT = "127.0.0.1"
    # sensor_thread = Thread(target=collect_sensor_data, args = (EMOTIBIT_IP_DEFAULT, EMOTIBIT_PORT_NUMBER))
    # sensor_thread.start()
    
    experiment_gui(experiment_num)

