import numpy as np
import csv
import pandas as pd
import os
from tqdm import tqdm
import glob

def load_subject_data(subject, data_loc):
    segment_to_df = {
            'baseline': None,
            'learning': None,
            'recognition_familar': None,
            'recognition_new': None,
            'recall': None,
        }
    cols_to_drop = ['EmotiBitTimestamp', 'estimated_pupil_time', 'label', 
                    'is_question', 'is_verbal', 'remembered', 'resp_confidence', 'image']
    baseline_loc = os.path.join(data_loc, 'negative', 'baseline.csv')
    segment_to_df['baseline'] = pd.read_csv(baseline_loc).drop(columns=cols_to_drop)

    for segment in segment_to_df:
        if segment != 'baseline':
            csv_files = glob.glob(os.path.join(data_loc, segment, '*.csv'))
            merged_df = None
            for file in csv_files:
                if merged_df is None:
                    merged_df = pd.read_csv(file)
                else:
                    file_df = pd.read_csv(file)
                    pd.concat([merged_df, file_df], ignore_index=True)
            merged_df.drop(columns=cols_to_drop, inplace=True)
            segment_to_df[segment] = merged_df
    return segment_to_df

def get_max_min_channels(df):
    cols_to_max_min = dict()
    for col in df.columns:
        cols_to_max_min[col] = (max(df[col]), min(df[col]))
    return cols_to_max_min

def get_max_min_df_dict(df_dict):
    cols_to_max_min = dict()
    for segment in df_dict.keys():
        segment_vals = get_max_min_channels(df_dict[segment])
        for col in segment_vals:
            if col in cols_to_max_min.keys():
                max_win = max(segment_vals[col][0], cols_to_max_min[col][0])
                min_win = min(segment_vals[col][1], cols_to_max_min[col][1])
                segment_vals[col] = (max_win, min_win)
            else:
                cols_to_max_min[col] = segment_vals[col]
    return cols_to_max_min

def normalize_df(df, cols_to_max_min):
    ret_df = df.copy()
    for col in df.columns:
        col_max, col_min = cols_to_max_min[col]
        if (col_min == 0 or col_min == 0.0) and (col_max == 0 or col_max == 0.0):
            continue
        if col_min == col_max:
            col_min = 0
        ret_df[col] = (ret_df[col] - col_min) / (col_max - col_min)
    return ret_df

def normalize_df_dict(df_dict, cols_to_max_min):
    normalized_df_dict = dict()
    for segment in df_dict:
        normalized_df_dict[segment] = normalize_df(df_dict[segment], cols_to_max_min)
    return normalized_df_dict

def calc_stats(norm_df, subject):
    stats_dict = {'subject_id': [subject]}
    for column in norm_df.columns:
        stats_dict[f'mean_{column}'] = [norm_df[column].mean()]
        stats_dict[f'var_{column}'] = [norm_df[column].var(ddof=0)]
    return stats_dict

def calc_stats_df_dict(df_dict, subject):
    stats_df_dict = dict()
    for segment in df_dict:
        stats_df_dict[segment] = calc_stats(df_dict[segment], subject)
    return stats_df_dict

def merge_stats_dict(old_stats_dict, new_stats_dict):
    updated_stats_dict = old_stats_dict.copy()
    for segment in old_stats_dict:
        for col in old_stats_dict[segment]:
            updated_stats_dict[segment][col].append(new_stats_dict[segment][col][0])
    return updated_stats_dict

def merge_df_dicts(global_df_dict, subject_df_dict):
    for segment in subject_df_dict:
        if segment in global_df_dict.keys():
            global_df_dict[segment] = pd.concat([global_df_dict[segment], subject_df_dict[segment]], ignore_index=True)
        else:
            global_df_dict[segment] = subject_df_dict[segment]
    return global_df_dict

def save_stats_dict(stats_dict, save_loc):
    for segment in stats_dict:
        segment_df = pd.DataFrame(stats_dict[segment])
        cols_to_drop = []
        for col in segment_df.columns:
            if 'Unnamed' in col:
                cols_to_drop.append(col)
        segment_df.drop(columns=cols_to_drop, inplace=True)
        segment_save_loc = os.path.join(save_loc, f'{segment}_analysis.csv')
        segment_df.to_csv(segment_save_loc)

def run(subjects, data_loc, save_loc):
    global_df_dict = dict()
    stats_dict = None
    
    for subject in subjects:
        subject_data_loc = os.path.join(data_loc, str(subject))
        subject_df_dict = load_subject_data(subject, subject_data_loc)
        global_df_dict = merge_df_dicts(global_df_dict, subject_df_dict)
        subject_cols_to_max_min = get_max_min_df_dict(subject_df_dict)
        subject_normalized_df_dict = normalize_df_dict(subject_df_dict, 
                                                 subject_cols_to_max_min)
        if stats_dict is None:
            stats_dict = calc_stats_df_dict(subject_normalized_df_dict, subject)
        else:
            stats_dict = merge_stats_dict(stats_dict, 
                                          calc_stats_df_dict(subject_normalized_df_dict, subject))
        
    global_cols_to_max_min = get_max_min_df_dict(global_df_dict)
    global_normalized_df_dict = normalize_df_dict(global_df_dict, 
                                                 global_cols_to_max_min)
    stats_dict = merge_stats_dict(stats_dict, 
                                  calc_stats_df_dict(global_normalized_df_dict, 'global'))

    save_loc = os.path.join(save_loc)
    os.makedirs(save_loc, exist_ok=True)
    save_stats_dict(stats_dict, save_loc)

if __name__=='__main__':
    num_subjects = 32
    subjects = list(range(101, 100+num_subjects+1))
    subjects.remove(105)
    subjects.remove(122)
    data_loc = '/Users/monaabd/Desktop/emotibit_segmented/'
    save_loc = '/Users/monaabd/Desktop/emotibit_mean_var_data/'

    run(subjects, data_loc, save_loc)