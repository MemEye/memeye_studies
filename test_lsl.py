"""Minimal example of how to send event triggers in PsychoPy with
LabStreamingLayer.
In this example, the words "hello" and "world" alternate on the screen, and
an event marker is sent with the appearance of each word.
TO RUN: open in PyschoPy Coder and press 'Run'. Or if you have the psychopy
Python package in your environment, run `python hello_world.py` in command line.
ID     EVENT
------------
1  --> hello
2  --> world
99 -->  test
------------
"""
from psychopy import core, visual, event
from pylsl import StreamInfo, StreamOutlet
from threading import Thread
import socket
import sys
import os
import zmq
import msgpack as serializer

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

def main():
    """Alternate printing 'Hello' and 'World' and send a trigger each time."""
    # Set up LabStreamingLayer stream.
    info = StreamInfo(name='EmotibitDataSyncMarker', type='Tags', channel_count=1,
                      channel_format='string', source_id='12345')
    outlet = StreamOutlet(info)  # Broadcast the stream.
    outlet.push_sample(['help'])
    outlet.push_sample(['help'])
    outlet.push_sample(['help'])

    pupil_ip, pupil_port = "127.0.0.1", 50020

    # core.wait(10)

        # 1. Setup network connection
    # check_capture_exists(pupil_ip, pupil_port)
    # pupil_remote, pub_socket = setup_pupil_remote_connection(pupil_ip, pupil_port)


    # This is not necessary but can be useful to keep track of markers and the
    # events they correspond to.
    markers = {
        'hello': ["Hello"],
        'world': ["World"],
        'test': ["Test5"],
    }
    
    # Send triggers to test communication.
    for _ in range(5):
        outlet.push_sample(markers['test'])
        core.wait(0.5)
        
    # Instantiate the PsychoPy window and stimuli.
    win = visual.Window([800, 600], allowGUI=False, monitor='testMonitor',
                        units='deg')
    hello = visual.TextStim(win, text="Hello")
    world = visual.TextStim(win, text="World")

    for i in range(200):
        if not i % 2:  # If i is even:
            hello.draw()
            # # Experiment with win.callOnFlip method. See Psychopy window docs.
            # win.callOnFlip(outlet.push_sample, markers['hello'])
            win.flip()
            outlet.push_sample(markers['hello'])
        else:
            world.draw()
            # win.callOnFlip(outlet.push_sample, markers['world'])
            win.flip()
            outlet.push_sample(markers['world'])
        if 'escape' in event.getKeys():  # Exit if user presses escape.
            break
        core.wait(1.0)  # Display text for 1.0 second.
        win.flip()
        core.wait(0.5)  # ISI of 0.5 seconds.

    win.close()
    core.quit()

if __name__ == "__main__":
    sensor_thread = Thread(target=main())
    sensor_thread.start()
    # main()
    # "broadcast": {
    #       "enabled": true
    #     },
    # ,
    #     "unicast": {
    #         "enabled": true,
    #         "ipMax": 254,
    #         "ipMin": 2,
    #         "nUnicastIpsPerLoop": 1,
    #         "unicastMinLoopDelay_msec": 3
    #       }