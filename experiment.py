from psychopy import visual, core, event
from combined_annotations import collect_sensor_data
from threading import Thread
import os
import random

images_to_names = {'CFD-AF-200-228-N.jpg': 'Emma', 'CFD-AF-202-122-N.jpg': 'Olivia', 
                   'CFD-AF-203-077-N.jpg': 'Ava', 'CFD-AF-204-067-N.jpg': 'Isabella', 
                   'CFD-AF-205-155-N.jpg': 'Sophia', 'CFD-AF-206-079-N.jpg': 'Charlotte', 
                   'CFD-BF-001-025-N.jpg': 'Mia', 'CFD-BF-002-001-N.jpg': 'Amelia', 
                   'CFD-BF-003-003-N.jpg': 'Harper', 'CFD-BF-004-014-N.jpg': 'Evelyn', 
                   'CFD-BF-007-001-N.jpg': 'Abigail','CFD-BF-008-001-N.jpg': 'Emily',
                   'CFD-WF-001-003-N.jpg': 'Elizabeth','CFD-WF-002-004-N.jpg': 'Sandra',
                   'CFD-WF-003-003-N.jpg': 'Avery','CFD-WF-005-010-N.jpg': 'Ella',
                   'CFD-WF-006-002-N.jpg': 'Scarlett','CFD-WF-007-001-N.jpg': 'Grace',
                   'CFD-LF-200-058-N.jpg': 'Chloe','CFD-LF-201-035-N.jpg': 'Victoria',
                   'CFD-LF-202-065-N.jpg': 'Riley','CFD-LF-203-066-N.jpg': 'Aria',
                   'CFD-LF-204-133-N.jpg': 'Lily','CFD-LF-205-100-N.jpg': 'Aubrey',
                   'CFD-AM-201-076-N.jpg': 'Liam','CFD-AM-202-079-N.jpg': 'Noah',
                   'CFD-AM-203-086-N.jpg': 'William','CFD-AM-204-122-N.jpg': 'James',
                   'CFD-AM-205-153-N.jpg': 'Oliver','CFD-AM-206-086-N.jpg': 'Benjamin',
                   'CFD-BM-001-014-N.jpg': 'Elijah','CFD-BM-002-013-N.jpg': 'Lucas',
                   'CFD-BM-003-003-N.jpg': 'Mason','CFD-BM-004-002-N.jpg': 'Logan',
                   'CFD-BM-005-003-N.jpg': 'Alexander','CFD-BM-011-016-N.jpg': 'Ethan', 
                   'CFD-LM-200-045-N.jpg': 'Jacob', 'CFD-LM-201-057-N.jpg': 'Michael',
                   'CFD-LM-202-072-N.jpg': 'Daniel','CFD-LM-203-026-N.jpg': 'Henry',
                   'CFD-LM-204-001-N.jpg': 'Jackson','CFD-LM-206-204-N.jpg': 'Sebastian',
                   'CFD-WM-001-014-N.jpg': 'Aiden','CFD-WM-002-009-N.jpg': 'Matthew',
                   'CFD-WM-003-002-N.jpg': 'Samuel','CFD-WM-004-010-N.jpg': 'David',
                   'CFD-WM-006-002-N.jpg': 'Joseph','CFD-WM-009-002-N.jpg': 'Carter'}

def experiment_1_gui():
    # Parameters
    shown_images_dir = '/Users/monaabd/Desktop/meng/memeye_studies/experiment_1_images/people/shown/'
    extra_images_dir = '/Users/monaabd/Desktop/meng/memeye_studies/experiment_1_images/people/extra/'
    learning_time = 5  # Time each image is shown during learning phase (in seconds)
    break_time = 3  # Time between images during learning phase (in seconds)
    remembering_time = 10  # Max time for each image during remembering phase (in seconds)
    batch_size = 12  # Number of images in a batch
    extra_images_size = 4  # Number of extra images shown later
    num_batches = 4  # Number of batches

    # Initialize window
    win = visual.Window(fullscr=True)
    image_pos = (0, 0.2)  # Image is centered, slightly above the center of the screen
    text_pos = (0, -0.3)  # Text is centered, below the image
    def wait_for_space(message):
        message_stim = visual.TextStim(win=win, text=message)
        message_stim.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
    # Function to show images
    def show_images(images, phase):
        for img_path in images:
            image = visual.ImageStim(win, image=img_path, pos=image_pos, size=0.8)
            img_name = os.path.basename(img_path)  # Get the filename of the image
            if phase == 'Learning':
                text = images_to_names.get(img_name, '') 
            elif phase == 'Remembering':
                text = "Do you remember this face? \n (1: Yes, 2: No)"
            elif phase == 'Names':
                text = "Do you remember their name? \n (1: Yes, 2: No)"
            #TODO: send annotations to devices for the phase, 
            # also include image name to distinguish between sessions
            
            image.draw()
            text_stim = visual.TextStim(win, text=text, pos=text_pos, color=(1, 1, 1))
            text_stim.draw()
            win.flip()
            
            # Set different timing for different phases
            if phase == 'Learning':
                core.wait(learning_time)
                keys = event.getKeys(keyList=['escape'])
                if 'escape' in keys:
                    core.quit()
            elif phase in ['Remembering', 'Names']:
                # Wait for response or timeout
                timer = core.Clock()
                keys = event.waitKeys(maxWait=remembering_time, keyList=['1', '2', 'escape'], timeStamped=timer)
                if keys:
                    key, reaction_time = keys[0]
                    # Do something with the response and reaction time
                    # TODO: send the response to the annotations
                    if key == 'escape':
                        core.quit()

            # Break between images (only during learning phase)
            #TODO: how long is the break for the other phases?
            if phase == 'Learning':
                win.flip()
                core.wait(break_time)

            #TODO: bookend the annotation for the phase, move onto the next image

    # Load images
    shown_images = [os.path.join(shown_images_dir, img) for img in os.listdir(shown_images_dir) if img.endswith('.jpg')]
    extra_images = [os.path.join(extra_images_dir, img) for img in os.listdir(extra_images_dir) if img.endswith('.jpg')]
    random.shuffle(shown_images)
    random.shuffle(extra_images)
    # Split into batches
    batches = [shown_images[i:i + batch_size] for i in range(0, len(shown_images), batch_size)]
    extra_image_groups = [extra_images[i:i + extra_images_size] for i in range(0, len(extra_images), extra_images_size)]

    # Run experiment
    for i, (batch, extra_group) in enumerate(zip(batches, extra_image_groups)):
        # Phase 1: Learning
        show_images(batch, 'Learning')
        
        # Wait for participant to press key to continue
        wait_for_space("Press space to continue to the remembering phase")
        
        # Phase 2: Remembering
        show_images(batch + extra_group, 'Remembering')
        
        # Wait for participant to press key to continue
        wait_for_space("Press space to continue to names phase")
        
        # Phase 3: Names
        show_images(batch + extra_group, 'Names')
        
        if i < num_batches - 1:
            # Wait for participant to press key to continue to next batch
            event.waitKeys(keyList=['space'])

    win.close()
    core.quit()

# experiment 1 loop:

# press s to start
# when looping through images, automatically label as a learnign phase
#then when intesting phase, label images as either never seen, or recognition phase
# then ask question if they can recognize
# if yes, then add the label of yes to the table
# then ask the details for recall, auto label
# if recall, then send yes, if no then send no
# recogniton response time
#double the amount of images 
#show in phases
# split up recall and recognition

#use the chicago database

# only keyboard inputs are yes or no, start, and stop maybe n to move onto the next segment

# see how to save pupilabs images 


if __name__=='__main__':
    # EMOTIBIT_BUFFER_INTERVAL = 0.01  # 100hz
    # emotibit_save_location = './data_test'
    # record_num = 0
    # # emotibit_save_name = f'record_{record_num}.csv'

    # EMOTIBIT_PORT_NUMBER = 12345
    # EMOTIBIT_IP_DEFAULT = "127.0.0.1"
    # sensor_thread = Thread(target=collect_sensor_data, args = (EMOTIBIT_IP_DEFAULT, EMOTIBIT_PORT_NUMBER))
    # sensor_thread.start()
    experiment_1_gui()




