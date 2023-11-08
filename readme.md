### How To: Running Experiment Script

Prereqs:
- Pupil Labs Software Downloaded
- Emotibit Software Downloaded
- Emotibit Config set to run on same network as computer

Steps:
0. Make sure to get consent and biographical/democgraphic info
1. Open Pupil Capture and adjust the headset camera views for each camera
2. Run calibration on the participant using the Pupil Capture app
3. Make sure the emotibit is connected to the computer and is streaming data, and it broadcasting data via OSC
4. Within the experiment_full.py script, make sure the following variables are set properly:
    - EMOTIBIT_BUFFER_INTERVAL is set to the frequency we want to collect emotibit data at
    - data_save_location is set to the location we want to store all experiment data
    - subject_id is set to the current subjects ID/name
    - experiment_num is set to the experiment that this subject will perform
5. Run through experiment intro
6. Run python script
