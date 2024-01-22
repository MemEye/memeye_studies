import numpy as np
import csv
import pandas as pd
import os
from tqdm import tqdm
import glob
import pingouin as pg
from statsmodels.stats.anova import AnovaRM 
from scipy.stats import f_oneway


def run(num_subjects, global_loc):
    subjects = list(range(101, 100+num_subjects+1))
    needed_columns = ['sac_count']

    # ERROR with negative and recognition_familiar
    segments = ['learning', 'recall', 'recognition_new']

    # segments = ['learning', 'recognition_familiar']

    summary_data = {'subject': [1, 2, 3]}
    



    for segment in segments:
        
        segment_super_df = pd.DataFrame()

        if 'phase' in summary_data.keys():
            summary_data['phase'].append(segment)
        else:
            summary_data['phase'] = [segment]


        # adds all the sac data for each person into segment_super_df
        for subject in subjects:
            subject_file_list = list(glob.iglob(f'/Users/kevinzhu/Desktop/MemEye/pupil_segmented_sac_updated/{subject}' + f'/**/{segment}/saccade_summary.csv'))
            for file in subject_file_list:
                df = pd.read_csv(file, low_memory=False)
                segment_super_df = pd.concat([segment_super_df, df])

        for needed_column in needed_columns:
            mean_val = segment_super_df[needed_column].mean()

            # generate summary_data dataframe
            if 'sac_count' in summary_data.keys():
                summary_data['sac_count'].append(mean_val)
            else:
                summary_data['sac_count'] = [mean_val]
        
        # print(segment_super_df)

    summary_data_df = pd.DataFrame(summary_data)
    print(summary_data_df)


    # ANOVA_summary_data_df = (pg.rm_anova(data=summary_data_df, dv='sac_count', subject='subject', within=['phase'], detailed=True))
    
    # table = ANOVA_summary_data_df.fit()

    """summary_data_path = os.path.join('/Users/kevinzhu/Desktop/MemEye/', 'summary_data.csv')
    summary_data_df.to_csv(summary_data_path)"""



if __name__=='__main__':
    num_subjects = 32
    global_loc = '/Users/kevinzhu/Desktop/MemEye'
    run(num_subjects, global_loc)