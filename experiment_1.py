# from pythonosc import dispatcher, osc_server
from threading import Thread
import datetime
import time
# import pandas as pd
import os
import zmq
import msgpack as serializer
import socket
import sys
import os
import random
import json
import numpy as np
from PIL import Image
import faulthandler
from pylsl import StreamInfo, StreamOutlet
from psychopy import visual, core, event, monitors #, sound
faulthandler.enable()

#TODO: batches down to 24 images, 8 each

# VARIABLES THAT CAN CHANGE - ADJUST THESE TO CHANGE THE EXPERIMENT
on_lab_comp = True
EMOTIBIT_BUFFER_INTERVAL = 0.02  # 50hz, fastest datastream is 25Hz, can probably do 0.04
data_save_location = 'data'
subject_id = 'test_mona' #TODO: CHANGE THIS TO PID IN TABLE
experiment_num = 1
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
subject_save_location = os.path.join('./', data_save_location, subject_id, date_time)
emotibit_save_location = os.path.join('/Users/memeye/Desktop/memeye_data/raw', data_save_location, subject_id, date_time) if on_lab_comp else subject_save_location

info = StreamInfo(name='EmotibitDataSyncMarker', type='Tags', channel_count=1,
                      channel_format='string', source_id='12345')
outlet = StreamOutlet(info)  # Broadcast the stream.

mon = monitors.Monitor('testMonitor')  # Replace 'testMonitor' with the name of your monitor
screen_width, screen_height = mon.getSizePix()
print(mon.getSizePix())
window_height = screen_height - 50  # Adjust this value to leave space for the taskbar/dock
if not on_lab_comp:
    win = visual.Window(
        size=(1350, 740), 
        pos=(0, 25),  # This centers the window vertically. Adjust as needed.
        fullscr=False,  # Fullscreen is set to False
        screen=0,
        color=[0, 0, 0]
    )
else:
    win = visual.Window(
        fullscr=True,  # Fullscreen is set to False
        screen=1,
        color=[0, 0, 0]
    )
print(win.size)

noise_texture = np.random.normal(loc=0.5, scale=0.3, size=(win.size[1], win.size[0])) # loc is the mean, scale is the standard deviation

# Normalize the noise texture to be within the range [0, 1], as expected by PsychoPy
noise_texture = (noise_texture - noise_texture.min()) / (noise_texture.max() - noise_texture.min())
# Create an image stimulus from the RGB noise
noise_stim = visual.ImageStim(win, image=noise_texture, size = win.size, units='pix')


learning_time = 10  # Time each image is shown during learning phase (in seconds)
recognition_time = 7
break_time = 3  # Time between images during learning phase (in seconds)
recall_time = 7  # Max time for each image during remembering phase (in seconds)
recall_verbal_time = 7 #time to say name out loud

exp_1_shown_images_dir = './experiment_1_images/people/shown/'
exp_1_extra_images_dir = './experiment_1_images/people/extra/'
exp_1_practice_images_dir = './experiment_1_images/people/practice/'

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
subject_confidence = ''

start_recording = False
stop_recording = False
# batch_recording = False
send_annotation_to_pupil = False
exit_sensors = False
bookend_annotation = False
send_subject_response = False
subject_response_in_emotibit = False


def check_capture_exists(ip_address, port):
    """check pupil capture instance exists"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if not sock.connect_ex((ip_address, port)):
            print("Found Pupil Capture")
        else:
            print("Cannot find Pupil Capture")
            # raise Exception
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

def collect_sensor_data():
    global current_annotation
    global pupil_time_align_val
    global start_recording
    global stop_recording
    global send_annotation_to_pupil
    global exit_sensors
    global curr_image
    global bookend_annotation
    global send_subject_response
    global subject_response
    global subject_confidence
    global date_time
    global subject_save_location
    global outlet

    pupil_ip, pupil_port = "127.0.0.1", 50020

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

    while True:
        if start_recording:
            print('starting recording session')
            date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            subject_save_location = os.path.join('./', data_save_location, subject_id, date_time)
            full_path = os.path.join(subject_save_location, f'experiment_{experiment_num}')
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            message = f'R {full_path}'
            pupil_remote.send_string(message)
            pupil_remote.recv_string()
            outlet.push_sample(['starting pupil recording'])
            pupil_time_align_val = request_pupil_time(pupil_remote)
            outlet.push_sample([str(pupil_time_align_val)])
            print(pupil_time_align_val, 'time align')
            start_recording = False
            pupil_time_align_val = None

        if stop_recording:
            print('stopping recording session and saving')
            if current_annotation != '':
                local_time = local_clock()
                minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
                send_trigger(pub_socket, minimal_trigger)
                outlet.push_sample([current_annotation])
                # Add custom keys to your annotation
                current_annotation = ''
            
            pupil_time_align_val = request_pupil_time(pupil_remote)
            print(pupil_time_align_val, 'time align')
            outlet.push_sample([str(pupil_time_align_val)])
            pupil_remote.send_string("r")
            pupil_remote.recv_string()
            stop_recording = False
            pupil_time_align_val = None
        
        if send_subject_response:
            local_time = local_clock()
            pupil_time_align_val = request_pupil_time(pupil_remote)
            outlet.push_sample([str(pupil_time_align_val)])
            print(pupil_time_align_val, 'time align')
            pupil_time_align_val = None
            print(subject_response, subject_confidence)
            
            if subject_response != '':
                minimal_trigger = new_trigger(subject_response, duration, local_time + stable_offset_mean)
                send_trigger(pub_socket, minimal_trigger)
                outlet.push_sample([subject_response])
                
            if subject_confidence != '':
                minimal_trigger = new_trigger(subject_confidence, duration, local_time + stable_offset_mean)
                send_trigger(pub_socket, minimal_trigger)
                outlet.push_sample([subject_confidence])

            send_subject_response = False
            subject_response = ''
            subject_confidence = ''
            
        if send_annotation_to_pupil:
            local_time = local_clock()
            print('sending this annotation label')
            pupil_time_align_val = request_pupil_time(pupil_remote)
            outlet.push_sample([str(pupil_time_align_val)])
            print(pupil_time_align_val, 'time align')
            
            minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            outlet.push_sample([current_annotation])
            
            minimal_trigger = new_trigger(curr_image, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            outlet.push_sample([curr_image])
            
            # pupil_time_align_val = request_pupil_time(pupil_remote)
            # outlet.push_sample([str(pupil_time_align_val)])
            # print(pupil_time_align_val, 'time align')

            pupil_time_align_val = None
            
            send_annotation_to_pupil = False
            if bookend_annotation:
                current_annotation = ''
                curr_image = ''
                bookend_annotation = False

            
        if exit_sensors:
            print("exiting sensor streams")
            pupil_time_align_val = request_pupil_time(pupil_remote)
            outlet.push_sample([str(pupil_time_align_val)])
            print(pupil_time_align_val, 'time align')
            local_time = local_clock()
            outlet.push_sample([current_annotation])
            minimal_trigger = new_trigger(current_annotation, duration, local_time + stable_offset_mean)
            send_trigger(pub_socket, minimal_trigger)
            # Add custom keys to your annotation
            pupil_time_align_val = None
            current_annotation = ''

            # stop recording
            
            pupil_remote.send_string("r")
            pupil_remote.recv_string()
            pupil_remote.close()
            pub_socket.close()
            outlet.__del__()
            exit_sensors = False
            break


def learning_phase(images, practice = False):
    global win
    global noise_stim
    global learning_time
    global break_time
    global current_annotation
    global curr_image
    global send_annotation_to_pupil
    global bookend_annotation
    global on_lab_comp

    for img_path in images:
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Get the size of the window
        win_width, win_height = win.size

        # Calculate the scale factor for both dimensions
        scale = 0.3 if on_lab_comp else 0.2
        scale_width = min((win_width * scale) / img_width, 1)
        scale_height = min((win_height * scale) / img_height, 1)

        # Use the smaller scale factor to ensure the image does not exceed 80% of the screen
        scale_factor = min(scale_width, scale_height)
        pos_scale = 0.05 if on_lab_comp else 0.05
        image = visual.ImageStim(win, image=img_path, size=(img_width * scale_factor, img_height * scale_factor), pos =(0,win_height*pos_scale), units = 'pix') 

        img_name = os.path.basename(img_path)  # Get the filename of the image
        text = f"Name: {images_to_info.get(img_name, '').get('Name')} \n \n Fact: {images_to_info.get(img_name, '').get('Fact')}"

        current_annotation = 'learning' if not practice else 'practice learning'
        curr_image = img_name
        send_annotation_to_pupil = True

        image.draw()
        text_stim = visual.TextStim(win, text=text, pos=(0, -0.7), color=(1, 1, 1))
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
    global subject_confidence
    global send_subject_response

    if ratio_shown != 1:
        images_to_show = random.sample(shown_images, int(len(shown_images)*ratio_shown))
    else:
        images_to_show = shown_images
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
        scale_width = min((win_width * 0.4) / img_width, 1)
        scale_height = min((win_height * 0.4) / img_height, 1)

        # Use the smaller scale factor to ensure the image does not exceed 80% of the screen
        scale_factor = min(scale_width, scale_height)

        image = visual.ImageStim(win, image=img_path, size=(img_width * scale_factor, img_height * scale_factor), units = 'pix')

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

        send_annotation_to_pupil = True
        bookend_annotation = True
        

        text_stim = visual.TextStim(win, text="Do you recognize this face? \n \n (1: Yes, 2: No)", 
                                    pos=(0,0), color=(1, 1, 1))
        text_stim.draw()

        current_annotation = 'recognition questions' if not practice else 'practice recognition questions'
        curr_image = img_name
        send_annotation_to_pupil = True
        win.flip()
        core.wait(0.5)
        keys = event.waitKeys(keyList=['1', '2', 'num_1', 'num_2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            print(keys)
            if key == '1' or key == 'num_1':
                print('Yes')
                subject_response = 'Yes'
                send_subject_response = True
            elif key == '2'or key == 'num_2':
                print('No')
                subject_response = 'N0'
                send_subject_response = True
            if key == 'escape':
                core.quit()

        text_stim = visual.TextStim(win, text="How confident are you in your previous response? \n \n (1: High Confidence, 2: Low Confidence)", 
                                    pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        core.wait(0.5)
        keys = event.waitKeys(keyList=['1', '2', 'num_1', 'num_2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            if key == '1' or key == 'num_1':
                subject_confidence = 'High'
                send_subject_response = True
            elif key == '2'or key == 'num_2':
                subject_confidence = 'Low'
                send_subject_response = True
            if key == 'escape':
                core.quit()
        
        bookend_annotation = True
        send_annotation_to_pupil = True

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
    global recall_verbal_time
    global break_time
    global current_annotation
    global curr_image
    global send_annotation_to_pupil
    global bookend_annotation
    global subject_response
    global subject_confidence
    global send_subject_response
    global experiment_num

    images = images_to_show + extra_images
    random.shuffle(images)

    if recall_type == 'name':
        text = "Use the following 7 seconds to try to recall the person's first name (or anything else about them) in your mind. \n \n (Do not say out loud)"
        
    text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    if practice:
        core.wait(7)
    else:
        core.wait(5)

    for img_path in images:
        img_name = os.path.basename(img_path)

        img = Image.open(img_path)
        img_width, img_height = img.size

        # Get the size of the window
        win_width, win_height = win.size

        # Calculate the scale factor for both dimensions
        scale_width = min((win_width * 0.4) / img_width, 1)
        scale_height = min((win_height * 0.4) / img_height, 1)

        # Use the smaller scale factor to ensure the image does not exceed 80% of the screen
        scale_factor = min(scale_width, scale_height)
        image = visual.ImageStim(win, image=img_path, size=(img_width * scale_factor, img_height * scale_factor), units = 'pix')
        
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
        
        bookend_annotation = True
        send_annotation_to_pupil = True
        
        if recall_type == 'name':
            text = "Do you remember this person's first name (or anything else about them)? \n \n (1: Yes, 2: No)"

        text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        current_annotation = f'recall {recall_type} questions' if not practice else f'practice recall {recall_type} questions'
        curr_image = img_name
        send_annotation_to_pupil = True
        win.flip()
        core.wait(0.5)
        keys = event.waitKeys(keyList=['1', '2', 'num_1', 'num_2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            if key == '1' or key == 'num_1':
                subject_response = 'Yes'
                send_subject_response = True
            elif key == '2'or key == 'num_2':
                subject_response = 'No'
                send_subject_response = True
            if key == 'escape':
                core.quit()
        
        text_stim = visual.TextStim(win, text="How confident are you in your previous response? \n \n (1: High Confidence, 2: Low Confidence)", 
                                    pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        core.wait(0.5)
        keys = event.waitKeys(keyList=['1', '2', 'num_1', 'num_2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            if key == '1' or key == 'num_1':
                subject_confidence = 'High'
                send_subject_response = True
            elif key == '2'or key == 'num_2':
                subject_confidence = 'Low'
                send_subject_response = True
            if key == 'escape':
                core.quit()


        bookend_annotation = True
        send_annotation_to_pupil = True
        

        if recall_type == 'name':
            text = "Now try your best to say what you remember out loud."

        text_stim = visual.TextStim(win, text=text, pos=(0,0), color=(1, 1, 1))
        text_stim.draw()
        current_annotation = f'recall {recall_type} verbal' if not practice else f'practice recall {recall_type} verbal'
        curr_image = img_name
        send_annotation_to_pupil = True
        win.flip()
        core.wait(recall_verbal_time)

        send_annotation_to_pupil = True
        bookend_annotation = True

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
    core.wait(0.5)
    event.waitKeys(keyList=['1', 'num_1'])

def game_break():
    global win
    global current_annotation
    global send_annotation_to_pupil
    global bookend_annotation
    # global beep
    text_stim = visual.TextStim(win, text="1 Minute Game Break", pos=(0,0), color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    current_annotation = f'game break'
    send_annotation_to_pupil = True
    core.wait(60)
    bookend_annotation = True
    send_annotation_to_pupil = True
    # beep.play()
    text_stim = visual.TextStim(win, text="When you are ready, press [1] to continue", pos=(0,0), color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    core.wait(0.5)
    event.waitKeys(keyList=['1', 'num_1'])

def relax_break():
    global win
    global current_annotation
    global send_annotation_to_pupil
    global bookend_annotation
    # global beep
    text_stim = visual.TextStim(win, text="1 Minute Relax Break: \n \n You can use this time to rest and you can also close your eyes for a while.", pos=(0,0), color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    current_annotation = 'relax break'
    send_annotation_to_pupil = True
    core.wait(60)
    bookend_annotation = True
    send_annotation_to_pupil = True
    # beep.play()
    text_stim = visual.TextStim(win, text="When you are ready, press [1] to continue", pos=(0,0), color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    core.wait(0.5)
    event.waitKeys(keyList=['1', 'num_1'])


def experiment_gui():
    global exp_1_shown_images_dir
    global exp_1_extra_images_dir
    global win
    global start_recording
    global stop_recording
    global exit_sensors
    global noise_stim
    global start_recording
    global stop_recording
    # global batch_recording
    global on_lab_comp
    global outlet

    for _ in range(5):
        outlet.push_sample(['test'])
        core.wait(0.5)

    # Load images
    shown_images = [os.path.join(exp_1_shown_images_dir, img) for img in os.listdir(exp_1_shown_images_dir) if img.endswith('.jpg')]
    extra_images = [os.path.join(exp_1_extra_images_dir, img) for img in os.listdir(exp_1_extra_images_dir) if img.endswith('.jpg')]
    practice_images = [os.path.join(exp_1_practice_images_dir, img) for img in os.listdir(exp_1_practice_images_dir) if img.endswith('.jpg')]
        
    random.shuffle(shown_images)
    random.shuffle(extra_images)

    # TESTING VARS
    shown_images_batches = [shown_images[:1], shown_images[1:2], shown_images[2:3]]
    extra_images_batches = [extra_images[:1], extra_images[1:2], extra_images[2:3]]

    # shown_images = random.sample(shown_images, 33)
    # extra_images = random.sample(extra_images, 12)

    # random.shuffle(shown_images)
    # random.shuffle(extra_images)

    # shown_images_batches = [shown_images[:11], shown_images[11:22], shown_images[22:]]
    # extra_images_batches = [extra_images[:4], extra_images[4:8], extra_images[8:]]

    # Run experiment
    start_recording = True

    # baseline
    # text = "Before we begin, please relax and try to keep still while looking at the screen. \n\n  This part will be three minutes. \n \n Press [1] to continue." 
    # instructions(text)
    # noise_stim.draw()
    # win.flip()
    # core.wait(180)
    # practice phases are all here
    text = "We will now begin the practice section. \n \n Press [1] to continue."
    instructions(text)
    text = f"This study will consist of several sections that involve looking at images of faces. \n \n You will be asked to try to remember as many details as you can and answer questions later. \n \n Press [1] to continue. "
    instructions(text)

    text = f"We will now begin a practice learning phase of the experiment. \n \n Press [1] to continue"
    instructions(text)
    text = "Instructions: \n \n You will be shown a sequence of images with the person's name and related facts. \n \n Please keep your attention on the screen and remember as many details as possible for each person. \n \n You will be tested on how much you remember after this. \n \n It will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    learning_phase(practice_images, practice = True)
    text = "End of practice learning phase. \n \n Take a quick break, ask any questions. \n \n Press [1] to continue."

    #practice recognition phase
    instructions(f"We will now begin a practice recognition phase of the experiment. \n \n Press [1] to continue")
    text = f"Instructions: \n \n You will be shown a sequence of images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    recognition_phase(practice_images, [], repeats = False, ratio_shown = 1, practice=True)
    instructions('End of practice recognition phase. \n \n Take a quick break, ask any questions. \n \n Press [1] to continue.')

    #practice names phase
    instructions(f"We will now begin a practice names phase of the experiment. \n \n Press [1] to continue")
    text = f"Instructions: \n \n You will be shown a sequence of images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    instructions(text)
    recall_phase(practice_images, [], 'name', practice=True)
    instructions('End of practice names phase. \n \n  Take a quick break, ask any questions. \n \n Press [1] to continue.')

    text = "End of practice sections. \n \n If you have questions, please ask the researcher. \n \n Press [1] to continue to the game break."
    instructions(text)
    
    # # Practice batch recording
    # batch_recording = True

    game_break()

    text = "We will now begin the main experiment. \n \n Press [1] to continue."
    instructions(text)

    # for i in range(3):

    #     shown_images = shown_images_batches[i]
    #     extra_images = extra_images_batches[i]

    #     # Phase 1: Learning
    #     text = f"We will now begin the learning phase of the experiment (Batch {i+1} out of 3). \n \n Press [1] to continue"
    #     instructions(text)
    #     text = "Instructions: \n \n You will be shown a sequence of images with the person's name and related facts. \n \n Please keep your attention on the screen and remember as many details as possible for each person. \n \n You will be tested on how much you remember after this. \n \n It will automatically move forward to the next part. \n \n Press [1] to continue."
    #     instructions(text)
    #     learning_phase(shown_images)
    #     instructions('End of learning phase. \n \n Press [1] to continue to the game break.')
    #     game_break()

    #     # Phase 2: Recognition  
    #     text = f"We will now begin the recognition phase of the experiment (Batch {i+1} out of 3). \n \n Press [1] to continue"
    #     instructions(text)
    #     text = f"Instructions: \n \n You will be shown a sequence of images. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    #     instructions(text)
    #     recognition_phase(shown_images, extra_images, repeats = False, ratio_shown = 1)
    #     instructions('End of recognition phase. \n \n Press [1] to continue to the game break.')
    #     game_break()

    #     # Phase 3: Names
    #     text = f"We will now begin the names phase of the experiment (Batch {i+1} out of 3). \n \n Press [1] to continue"
    #     instructions(text)
    #     text = f"Instructions: \n \n You will be shown a sequence of images. \n \n Please keep your attention on the screen at all times. \n \n When you see the image, your job is just to look at it - it will automatically move forward to the next part. \n \n Press [1] to continue."
    #     instructions(text)
    #     recall_phase(shown_images, [], 'name')
    #     instructions('End of names phase. \n \n Press [1] to continue.')

    #     # batch recording
    #     if i < 2:
    #         instructions(f'End of batch {i+1} out of 3. \n \n Press [1] to continue to game/relax break.')
    #         # batch_recording = True
    #         game_break()
    #         relax_break()
    #     else:
    #         instructions(f'End of batch {i+1} out of 3. \n \n Press [1] to continue.')

    exit_sensors = True
    instructions(f"We have now completed the experiment. \n \n Press [1] to exit")
    win.close()
    # core.quit()

if __name__=='__main__':
    image_info_path = './experiment_1_names_facts.json'
    with open(image_info_path, 'r') as file:
        images_to_info = json.load(file)


    sensor_thread = Thread(target=collect_sensor_data)
    sensor_thread.start()
    
    experiment_gui()
    sensor_thread.join()

# ,
#       "lsl": {
#         "marker": {
#           "name": "DataSyncMarker",
#           "sourceId": "12345"
#         }
#       }

        # // "unicast": {
        # //   "enabled": true,
        # //   "ipMax": 254,
        # //   "ipMin": 2,
        # //   "nUnicastIpsPerLoop": 1,
        # //   "unicastMinLoopDelay_msec": 3
        # // }

