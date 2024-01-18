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
    cols_to_drop = ['sac_array_dir', 'blinks_confidence', 'is_blink', 
                'label', 'is_question', 'is_verbal', 'remembered', 
                'resp_confidence', 'pupil_timestamp']
    #TODO: fill this in
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
    for col in df.cols:
        col_max, col_min = cols_to_max_min[col]
        ret_df[col] = ret_df[col] - col_min / (col_max - col_min)
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
        stats_dict[f'var_{column}'] = [norm_df[column].var(ddof=0)]  #TODO: ask sam population vs sample variance ddorf val
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
        if segment in global_df_dict:
            global_df_dict[segment] = pd.concat([global_df_dict[segment], subject_df_dict[segment]], ignore_index=True)
        else:
            global_df_dict[segment] = subject_df_dict[segment]

def save_stats_dict(stats_dict, save_loc):
    pass
    #TODO

def run(subjects, data_loc, save_loc, eye):
    global_df_dict = dict()
    stats_dict = None
    
    for subject in subjects:
        subject_data_loc = None #TODO
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

    save_loc = save_loc # TODO: update this
    save_stats_dict(stats_dict, save_loc)

if __name__=='__main__':
    num_subjects = 2
    subjects = list(range(101, 100+num_subjects+1))
    data_loc = ''
    save_loc = ''

    run(subjects, data_loc, save_loc, 'left') #TODO: do this for left and right eye
    run(subjects, data_loc, save_loc, 'right')