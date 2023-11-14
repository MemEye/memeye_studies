### How To: Running Experiment Script

Prereqs:
- Pupil Labs Software Downloaded
- Emotibit Software Downloaded
- Emotibit Config set to run on same network as computer
- Restart lab computer

Steps:
0. Make sure to get consent and biographical/democgraphic info
1. Open Pupil Capture and adjust the headset camera views for each camera
2. Make sure in Pupil Capture that the save location is the desired location
3. Run calibration on the participant using the Pupil Capture app
4. Make sure the emotibit is connected to the computer and is streaming data, and it broadcasting data via OSC
5. Within the experiment_1.py script, make sure the following variables are set properly:
    - on_lab_comp is True if on the lab computer
    - EMOTIBIT_BUFFER_INTERVAL is set to the frequency we want to collect emotibit data at
    - data_save_location is set to the location we want to store all experiment data
    - subject_id is set to the current subjects ID/name
    - experiment_num is set to the experiment that this subject will perform
6. Run through experiment intro, inform participant to keep emotibit hand at rest and still, and to always look straight at the screen. Explain input system.
7. Start audio recording on phone
8. Run python script
    - During breaks check that the emotibit is still connected to the osciliscope.
9. When finished with experiments for the day, upload to media lab account for backup
