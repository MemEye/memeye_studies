from psychopy import visual, core, event
from combined_annotations import collect_sensor_data
from threading import Thread
import os
import random
import json
import numpy as np
from PIL import Image

learning_time = 7  # Time each image is shown during learning phase (in seconds)
recognition_time = 1 #5
recall_time = 10
break_time = 1 #3  # Time between images during learning phase (in seconds)
recall_time = 10  # Max time for each image during remembering phase (in seconds)

exp_1_shown_images_dir = '/Users/monaabd/Desktop/meng/memeye_studies/experiment_1_images/people/shown/'
exp_1_extra_images_dir = '/Users/monaabd/Desktop/meng/memeye_studies/experiment_1_images/people/extra/'

exp_2_shown_images_dir = '/Users/monaabd/Desktop/meng/memeye_studies/experiment_2_images/people/shown/'
exp_2_extra_images_dir = '/Users/monaabd/Desktop/meng/memeye_studies/experiment_2_images/people/extra/'

win = visual.Window(fullscr=True, color=[0, 0, 0])
noise_texture = np.random.normal(loc=0.5, scale=0.3, size=(win.size[1], win.size[0])) # loc is the mean, scale is the standard deviation

# Normalize the noise texture to be within the range [0, 1], as expected by PsychoPy
noise_texture = (noise_texture - noise_texture.min()) / (noise_texture.max() - noise_texture.min())
# Create an image stimulus from the RGB noise
noise_stim = visual.ImageStim(win, image=noise_texture, size = win.size, units='pix')

image_pos = (0, 0.2)  # Image is centered, slightly above the center of the screen
text_pos = (0, -0.5)  # Text is centered, below the image
center_pos = (0,0)


def learning_phase(images):
    global win
    global noise_stim
    global image_pos
    global text_pos
    global learning_time
    global break_time

    for img_path in images:
        image = visual.ImageStim(win, image=img_path, pos=image_pos, size=0.8)
        img_name = os.path.basename(img_path)  # Get the filename of the image
        text = f"Name: {images_to_info.get(img_name, '').get('Name')} \n Fact: {images_to_info.get(img_name, '').get('Fact')}"
            
        #TODO: send annotations to devices for the phase, 
        # also include image name to distinguish between sessions
        
        image.draw()
        text_stim = visual.TextStim(win, text=text, pos=text_pos, color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        
        # Set different timing for different phases
        core.wait(learning_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()

        # Break between images (only during learning phase)
        noise_stim.draw()
        win.flip()
        core.wait(break_time)

        #TODO: bookend the annotation for the phase, move onto the next image

def recognition_phase(images):
    global win
    global noise_stim
    global image_pos
    global text_pos
    global recognition_time
    global break_time

    for img_path in images:

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
        text = "Do you remember this face? \n (1: Yes, 2: No)"
            
        #TODO: send annotations to devices for the phase, 
        # also include image name to distinguish between sessions
        
        image.draw()
        win.flip()
        
        # Wait for response or timeout
        timer = core.Clock()
        core.wait(recognition_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()
        
        text_stim = visual.TextStim(win, text=text, pos=center_pos, color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        keys = event.waitKeys(keyList=['1', '2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            # Do something with the response and reaction time
            # TODO: send the response to the annotations
            if key == 'escape':
                core.quit()

        # Break between images (only during learning phase)
        noise_stim.draw()
        win.flip()
        core.wait(break_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()

        #TODO: bookend the annotation for the phase, move onto the next image


def recall_phase(images, recall_type):
    global win
    global noise_stim
    global image_pos
    global text_pos
    global recall_time
    global break_time

    for img_path in images:
        image = visual.ImageStim(win, image=img_path, pos=image_pos, size=0.8)
        text = ''
        if recall_type == 'name':
            text = "Use the following 10 seconds to try to recall the person's name in your mind. (Do not say out loud)"
        elif recall_type == 'fact':
            text = "Use the following 10 seconds to try to recall a fact with this person in your mind. (Do not say out loud)"
        elif recall_type == 'memory':
            text = "Use the following 10 seconds to try to recall a memory with this person your mind. (Do not say out loud)"

        text_stim = visual.TextStim(win, text=text, pos=center_pos, color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        core.wait(3)

        #TODO: send annotations to devices for the phase, 
        # also include image name to distinguish between sessions
        
        image.draw()
        win.flip()
    
        # Wait for response or timeout
        timer = core.Clock()
        core.wait(recall_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()
        

        if recall_type == 'name':
            text = "Do you remember this face? \n (1: Yes, 2: No)"
        elif recall_type == 'fact':
            text = "Do you remember facts about this person? \n (1: Yes, 2: No)"
        elif recall_type == 'memory':
            text = "Do you have a memory involing this person? \n (1: Yes, 2: No)"
        text_stim = visual.TextStim(win, text=text, pos=center_pos, color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        keys = event.waitKeys(keyList=['1', '2', 'escape'], timeStamped=timer)
        if keys:
            key, reaction_time = keys[0]
            # Do something with the response and reaction time
            # TODO: send the response to the annotations
            if key == 'escape':
                core.quit()
        
        if recall_type == 'name':
            text = "Now try your best to say the person's name out loud. It's ok it's wrong or if you cannot recall it."
        elif recall_type == 'fact':
            text = "Now try your best to say the person's facts out loud. It's ok it's wrong or if you cannot recall it."
        elif recall_type == 'memory':
            text = "Now try your best to say the memory out loud. It's ok it's wrong or if you cannot recall it."

        text_stim = visual.TextStim(win, text=text, pos=center_pos, color=(1, 1, 1))
        text_stim.draw()
        win.flip()
        core.wait(3)

        # Break between images (only during learning phase)
        noise_stim.draw()
        win.flip()
        core.wait(break_time)
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            core.quit()

        #TODO: bookend the annotation for the phase, move onto the next image

def instructions(text):
    text_stim = visual.TextStim(win, text=text, pos=center_pos, color=(1, 1, 1))
    text_stim.draw()
    win.flip()
    event.waitKeys(keyList=['1'])


def experiment_gui(exp_num):
    global exp_1_shown_images_dir
    global exp_1_extra_images_dir
    global exp_2_shown_images_dir
    global exp_2_extra_images_dir
    global win
    # Parameters
    
    def wait_for_continue(message):
        message_stim = visual.TextStim(win=win, text=message)
        message_stim.draw()
        win.flip()
        event.waitKeys(keyList=['1'])

    # Load images
    if exp_num == 1:
        shown_images = [os.path.join(exp_1_shown_images_dir, img) for img in os.listdir(exp_1_shown_images_dir) if img.endswith('.jpg')]
        extra_images = [os.path.join(exp_1_extra_images_dir, img) for img in os.listdir(exp_1_extra_images_dir) if img.endswith('.jpg')]
    
    if exp_num == 2:
        shown_images = [os.path.join(exp_2_shown_images_dir, img) for img in os.listdir(exp_2_shown_images_dir) if img.endswith('.jpg')]
        extra_images = [os.path.join(exp_2_extra_images_dir, img) for img in os.listdir(exp_2_extra_images_dir) if img.endswith('.jpg')]

    random.shuffle(shown_images)
    random.shuffle(extra_images)
    # shown_images = shown_images[:2]
    # extra_images = extra_images [:2]

    # Run experiment
    # TODO: add in practice rounds for each phase type

    if exp_num == 1:
        # Phase 1: Learning
        text = "Instructions: \n You will be shown a sequence of images with the person's name and related facts. Please keep your attention on the screen and remember as mush as details as possible for each person. You will be tested on how much you remember after this. \n Press [1] to continue."
        instructions(text)
        learning_phase(shown_images)
        wait_for_continue("Press [1] to continue to the recognition phase")
        text = "Instructions: \n You will be shown a sequence of images. Please keep your attention on the screen at all times. When you see the image, your job is just to look at it, it will automatically move forward to the next part. \n Press [1] to continue."
    elif exp_num == 2:
        text = "Instructions: \n You will be shown a sequence of famous people's images. Please keep your attention on the screen and remember as mush as details as possible for each person. You will be tested on how much you remember after this.  \n Press [1] to continue."
   
    # Phase 2: Recognition  
    instructions(text)
    recognition_phase(shown_images+extra_images)

    # Phase 3: Names
    wait_for_continue("Press [1] to continue to names phase")
    text = "Instructions: \n You will be shown a sequence of images. Please keep your attention on the screen at all times. When you see the image, your job is just to look at it, it will automatically move forward to the next part. \n Press [1] to continue."
    instructions(text)
    recall_phase(shown_images+extra_images, 'name')

    # Phase 4: Facts
    wait_for_continue("Press [1] to continue to facts phase")
    text = "Instructions: \n You will be shown a sequence of images. Please keep your attention on the screen at all times. When you see the image, your job is just to look at it, it will automatically move forward to the next part. \n Press [1] to continue."
    instructions(text)
    recall_phase(shown_images+extra_images, 'fact')

    # Phase 5: Memory
    if exp_num == 2:
        wait_for_continue("Press [1] to continue to memory phase")
        text = "Instructions: \n You will be shown a sequence of images. Please keep your attention on the screen at all times. When you see the image, your job is just to look at it, it will automatically move forward to the next part. \n Press [1] to continue."
        instructions(text)
        recall_phase(shown_images+extra_images, 'memory')


    win.close()
    core.quit()

if __name__=='__main__':
    image_info_path = 'experiment_1_names_facts.json'
    with open(image_info_path, 'r') as file:
        images_to_info = json.load(file)
    # EMOTIBIT_BUFFER_INTERVAL = 0.01  # 100hz
    # emotibit_save_location = './data_test'
    # record_num = 0
    # # emotibit_save_name = f'record_{record_num}.csv'

    # EMOTIBIT_PORT_NUMBER = 12345
    # EMOTIBIT_IP_DEFAULT = "127.0.0.1"
    # sensor_thread = Thread(target=collect_sensor_data, args = (EMOTIBIT_IP_DEFAULT, EMOTIBIT_PORT_NUMBER))
    # sensor_thread.start()
    experiment_gui(2)




