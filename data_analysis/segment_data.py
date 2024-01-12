import pandas as pd
import numpy as np
import os
import json


def segment_dataframe(df, column):
    change_indicator = df[column].notna() & df[column].shift().isna()

    change_indicator.iloc[0] = df[column].notna().iloc[0]

    df['group_id'] = change_indicator.cumsum()

    grouped_dfs = [group for _, group in df.groupby('group_id')]

    return grouped_dfs

def save_segments(grouped_dfs, segmented_path):
    break_count = 0
    for i, group_df in enumerate(grouped_dfs):
        group_df.drop(columns=['group_id'], inplace=True)
        label = list(group_df.label.unique())[0]
        image = list(group_df.image.unique())[0]
        if not isinstance(label, str):
            label = ''
        if not isinstance(image, str):
            image = ''
        if 'practice' in label:
            group_save_path = os.path.join(segmented_path, 'practice', label.replace(' ', '_'))
            os.makedirs(group_save_path, exist_ok=True)
            group_df.to_csv(os.path.join(group_save_path,  f'{image}.csv'))
        elif 'recall' in label:
            group_save_path = os.path.join(segmented_path, 'recall')
            os.makedirs(group_save_path, exist_ok=True)
            group_df.to_csv(os.path.join(group_save_path, f'{image}.csv'))
        elif 'recognition' in label:
            if image in images_to_info.keys():
                group_save_path = os.path.join(segmented_path, 'recognition_familar')
                os.makedirs(group_save_path, exist_ok=True)
                group_df.to_csv(os.path.join(group_save_path, f'{image}.csv'))
            else:
                group_save_path = os.path.join(segmented_path, 'recognition_new')
                os.makedirs(group_save_path, exist_ok=True)
                group_df.to_csv(os.path.join(group_save_path, f'{image}.csv'))
        elif 'learning' in label:
            group_save_path = os.path.join(segmented_path, 'learning')
            os.makedirs(group_save_path, exist_ok=True)
            group_df.to_csv(os.path.join(group_save_path, f'{image}.csv'))
        else:
            if 'baseline' in label:
                group_save_path = os.path.join(segmented_path, 'negative')
                os.makedirs(group_save_path, exist_ok=True)
                group_df.to_csv(os.path.join(group_save_path, 'baseline.csv'))
            if 'break' in label:
                group_save_path = os.path.join(segmented_path, 'negative')
                os.makedirs(group_save_path, exist_ok=True)
                group_df.to_csv(os.path.join(group_save_path, f'{label}_{break_count}.csv'))
                break_count+=1
            else:
                group_save_path = os.path.join(segmented_path, 'negative')
                os.makedirs(group_save_path, exist_ok=True)
                group_df.to_csv(os.path.join(group_save_path, f'negative_{break_count}.csv'))
                break_count+=1
            

def run_emotibit(data_loc, subject_nums):
    for subject_id in subject_nums:
        data_path = os.path.join(data_loc, f'{subject_id}/processed/')

        emotibit_path = os.path.join(data_path, f'processed_emotibit_{subject_id}.csv')
        emotibit_segmented_path = os.path.join(data_path, 'segmented/emotibit/')
        emotibit_large = pd.read_csv(emotibit_path)
        grouped_dfs = segment_dataframe(emotibit_large, 'label')
        save_segments(grouped_dfs, emotibit_segmented_path)


def run_pupil(data_loc, subject_nums):
    for subject_id in subject_nums:
        data_path = os.path.join(data_loc, f'{subject_id}/processed_pupil/')

        pupil_path_left = os.path.join(data_path, f'processed_pupil_{subject_id}_eye_left.csv')
        pupil_path_right = os.path.join(data_path, f'processed_pupil_{subject_id}_eye_right.csv')
        
        pupil_segmented_path_left = os.path.join(data_path, 'segmented_left/')
        pupil_segmented_path_right = os.path.join(data_path, 'segmented_right/')
        pupil_left = pd.read_csv(pupil_path_left)
        pupil_right = pd.read_csv(pupil_path_right)

        grouped_dfs_left = segment_dataframe(pupil_left, 'label')
        grouped_dfs_right = segment_dataframe(pupil_right, 'label')

        save_segments(grouped_dfs_left, pupil_segmented_path_left)
        save_segments(grouped_dfs_right, pupil_segmented_path_right)


if __name__=='__main__':
    image_info_path = '/Users/monaabd/Desktop/meng/meng_thesis/memeye_studies/data_collection/experiment_scripts/experiment_1_names_facts.json'
    with open(image_info_path, 'r') as file:
        images_to_info = json.load(file)
    num_subjects = 32
    subject_nums = list(range(101, 100+num_subjects+1))
    data_loc = '/Users/monaabd/Desktop/processed_data/'
    # run_emotibit(data_loc, subject_nums)
    pupil_data_loc = '/Users/monaabd/Desktop/pupil_processed_new/'
    run_pupil(pupil_data_loc, subject_nums)


