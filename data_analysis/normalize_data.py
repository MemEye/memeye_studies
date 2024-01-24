import numpy as np
import csv
import pandas as pd
import os
from tqdm import tqdm
import glob

def normalize_user(user):
    
    save_loc = '/Users/kevinzhu/Desktop/MemEye/normalized_data'

    all_files = list(glob.iglob(f'/Users/kevinzhu/Desktop/MemEye/pupil_segmented_sac_updated/{user}' + f'/**/**/*.csv', recursive=True))
    files = [f for f in all_files if not os.path.basename(f).startswith(('game', 'negative'))]
    files = [f for f in all_files if not os.path.basename(f).startswith(('relax', 'saccade'))] 

    normalized_data = dict()

    super_df = pd.DataFrame()

    # print(files)
    for file in all_files:
        df = pd.read_csv(file, low_memory=False)
        super_df = pd.concat([super_df, df])

        """for col in df:
            if df[col].apply(lambda x: isinstance(x, (int, float))).all(): # if column data values are int or float
                mean = df[col].mean()
                if col in normalized_data.keys():
                    normalized_data[col].append(mean)
                else:
                    normalized_data[col] = [mean]
        
        normalized_data_df = pd.DataFrame(normalized_data)
        normalized = (normalized_data_df - normalized_data_df.min()) / (normalized_data_df.max() - normalized_data_df.min())"""

        
        normalized_data_path = os.path.join(save_loc, "global_DF.csv")
        super_df.to_csv(normalized_data_path)




def run(num_subjects):
    subjects = list(range(101, 100+num_subjects+1))
    files_loc = '/Users/kevinzhu/Desktop/MemEye/pupil_segmented_sac_updated'
    save_loc = '/Users/kevinzhu/Desktop/MemEye/normalized_data'
    os.makedirs(save_loc, exist_ok=True)


    segments = ['learning', 'negative', 'recall', 'recognition_familiar', 'recognition_new']

    for subject in subjects:
        for segment in segments[0:2]:
            files = list(glob.iglob(f'/Users/kevinzhu/Desktop/MemEye/pupil_segmented_sac_updated/{subject}' + f'/**/{segment}/*.csv', recursive=True)) # list of all files for specific subject, segment
            
            # creates the segment folders
            save_path = os.path.join(save_loc, f'{segment}')
            os.makedirs(save_path, exist_ok=True)

            normalized_data = dict()

            
            for file in files[0:2]:
                print(file)
                df = pd.read_csv(file, low_memory=False)

                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])

                for col in df:
                    if df[col].apply(lambda x: isinstance(x, (int, float))).all(): # if column data values are int or float
                        mean = df[col].mean()
                        if col in normalized_data.keys():
                            normalized_data[col].append(mean)
                        else:
                            normalized_data[col] = [mean]
    
    normalized_data_df = pd.DataFrame(normalized_data)
    normalized_data_path = os.path.join(save_path, "normalized_data.csv")
    normalized_data_df.to_csv(normalized_data_path)


if __name__=='__main__':
    num_subjects = 1
    normalize_user('101')
    #run(num_subjects)