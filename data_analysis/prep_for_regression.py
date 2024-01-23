import numpy as np
import csv
import pandas as pd
import os
from tqdm import tqdm
import glob

def calc_blinks(df):
    df['is_blink'] = df['is_blink'] == 'y'
    blinks_start = df['is_blink'] & ~(df['is_blink'].shift(1).fillna(False))
    blinks_end = df['is_blink'] & ~(df['is_blink'].shift(-1).fillna(False))
    start_times = df['pupil_timestamp'][blinks_start]
    end_times = df['pupil_timestamp'][blinks_end]
    blink_durations = end_times.values - start_times.values
    blink_count = len(blink_durations)
    if list(blink_durations) == []:
        blink_durations = [0]
    max_duration = max(blink_durations)
    blink_durations= np.array(blink_durations)
    mean_dur = np.mean(blink_durations)
    var_dur = np.var(blink_durations)
    return {'blink_dur_mean': mean_dur, 'blink_dur_var': var_dur, 
            'blink_max_duration': max_duration, 'blink_count':blink_count}

def calc_fixations(df):
    # get all the dispersions, and get the mean and var for them
    df['is_fixation'] = ~df['fixations_dispersion'].isna()
    fixations_start = df['is_fixation'] & ~df['is_fixation'].shift(1).fillna(False)
    start_times = df['pupil_timestamp'][fixations_start]
    end_times = start_times.shift(-1, fill_value=df['pupil_timestamp'].iloc[-1])
    fixation_durs = end_times.values - start_times.values
    dispersions = df['fixations_dispersion'][fixations_start].tolist()

    dispersions = np.array(dispersions)
    fix_durs = np.array(fixation_durs)
    mean_disp = np.mean(dispersions)
    mean_durs = np.mean(fix_durs)
    var_disp = np.var(dispersions)
    var_durs = np.var(fix_durs)

    return {'fixation_count': len(fixation_durs), 'fixation_dur_mean':mean_durs,
            'fixation_dur_var':var_durs, 'fixation_disp_mean': mean_disp, 
            'fixation_disp_var': var_disp}
    

def mean_var_col(df):
    cols_to_keep = ['diameter', 'ellipse_angle', 'diameter_3d']
    df_filtered = df[cols_to_keep]
    means = df_filtered.mean()
    variances = df_filtered.var()
    stats_dict = {f'{col}_mean': means[col] for col in df_filtered.columns}
    stats_dict.update({f'{col}_variance': variances[col] for col in df_filtered.columns})
    return stats_dict

def load_saccade_info(file, segment):
    cols_to_keep = ['filename', 'sac_count', 'sac_dur_mean', 'sac_dur_var', 
                    'sac_len_mean', 'sac_len_var', 'sac_vel_mean', 'sac_vel_var', 
                    'sac_dir_N', 'sac_dir_NE', 'sac_dir_E', 'sac_dir_SE', 
                    'sac_dir_S', 'sac_dir_SW', 'sac_dir_W', 'sac_dir_NW']
    df = pd.read_csv(file)
    df_filtered = df[cols_to_keep]
    df_filtered['filename'] = df_filtered['filename'].str[:-4]
    if segment == 'negative':
        df_filtered = df_filtered[df_filtered['filename'] == 'baseline']
    return df_filtered

def prep_pupil_files(files, segment, eye, subject_id):
    saccade_info = None
    merged_df = {'filename': [],
                 'label': [], 
                 'eye': [],
                 'subject_id': []}
    for file in files:
        if 'saccade_summary' in file:
            saccade_info = load_saccade_info(file, segment)
        else:
            if (segment == 'negative' and 'baseline' in file) or (segment != 'negative'):
                image_name = file.split('/')[-1][:-4]
                df = pd.read_csv(file)
                blink_info = calc_blinks(df)
                fix_info = calc_fixations(df)
                mean_var = mean_var_col(df)
                merged_dict = {**blink_info, **fix_info, **mean_var}
                merged_df['filename'].append(image_name)
                merged_df['label'].append(segment)
                merged_df['eye'].append(eye)
                merged_df['subject_id'].append(subject_id)
                for key, value in merged_dict.items():
                    if key in merged_df:
                        merged_df[key].append(value)
                    else:
                        merged_df[key] = [value]
    merged_df = pd.DataFrame(merged_df)
    merged = pd.merge(merged_df, saccade_info, on='filename')
    return merged

def emotibit_mean_var_col(df):
    cols_to_keep = ['T1', 'TH', 'EA', 'EL', 'PR', 'PI', 'PG', 'SA', 'SR', 'SF', 'HR']
    df_filtered = df[cols_to_keep]
    means = df_filtered.mean()
    variances = df_filtered.var()
    stats_dict = {f'{col}_mean': means[col] for col in df_filtered.columns}
    stats_dict.update({f'{col}_variance': variances[col] for col in df_filtered.columns})
    return stats_dict


def prep_emotibit_files(files, segment, subject_id):
    saccade_info = None
    merged_df = {'filename': [],
                 'label': [],
                 'subject_id': []}
    for file in files:
        if (segment == 'negative' and 'baseline' in file) or (segment != 'negative'):
            image_name = file.split('/')[-1][:-4]
            df = pd.read_csv(file)
            mean_var = emotibit_mean_var_col(df)
            merged_df['filename'].append(image_name)
            merged_df['label'].append(segment)
            merged_df['subject_id'].append(subject_id)
            for key, value in mean_var.items():
                if key in merged_df:
                    merged_df[key].append(value)
                else:
                    merged_df[key] = [value]
    merged_df = pd.DataFrame(merged_df)
    return merged_df


def run_pupil(subjects, pupil_data_loc, eye):
    segments = ['negative', 'learning', 'recall', 'recognition_familar', 'recognition_new']
    merged_df = None
    for subject in subjects:
        for segment in segments:
            pupil_subject_data_loc = os.path.join(pupil_data_loc, str(subject), f'segmented_{eye}', segment)
            pupil_files = glob.glob(os.path.join(pupil_subject_data_loc, '*.csv'))
            pupil_merged_df = prep_pupil_files(pupil_files, segment, eye, subject)
            if type(merged_df) == (pd.DataFrame):
                merged_df = pd.concat([merged_df, pupil_merged_df])
            else:
                merged_df = pupil_merged_df.copy()
    return merged_df

def run_emotibit(subjects, emotibit_data_loc):
    segments = ['negative', 'learning', 'recall', 'recognition_familar', 'recognition_new']
    merged_df = None
    for subject in subjects:
        for segment in segments:
            emotibit_subject_data_loc = os.path.join(emotibit_data_loc, str(subject), segment)
            emotibit_files = glob.glob(os.path.join(emotibit_subject_data_loc, '*.csv'))
            emotibit_merged_df = prep_emotibit_files(emotibit_files, segment, subject)
            if type(merged_df) == (pd.DataFrame):
                merged_df = pd.concat([merged_df, emotibit_merged_df])
            else:
                merged_df = emotibit_merged_df.copy()
    return merged_df
# emotibit: drope verything not PPG EDA Temp, SCR

def merge_eye_emotibit(eye_df, emotibit_df):
    merged = pd.merge(eye_df, emotibit_df, on=['filename', 'subject_id', 'label'])
    return merged

def merge_left_right(left, right, save_path):
    merged = pd.concat([left, right])
    merged.reset_index(inplace=True)
    # merged.drop(columns=['label_y', 'index'], inplace=True)
    # merged.rename(columns={'label_x': 'label'}, inplace=True)
    os.makedirs(save_path, exist_ok=True)
    merged.to_csv(os.path.join(save_path, 'prepared.csv'))
    

if __name__=='__main__':
    num_subjects = 32
    subjects = list(range(101, 100+num_subjects+1))
    pupil_data_loc = '/Users/monaabd/Desktop/pupil_segmented_sac_updated/'
    emotibit_data_loc = '/Users/monaabd/Desktop/emotibit_segmented_new/'
    save_loc = '/Users/monaabd/Desktop/pupil_regression_prep/'

    left_processed = run_pupil(subjects, pupil_data_loc, 'left')
    right_processed = run_pupil(subjects, pupil_data_loc, 'right')
    emotibit_processed = run_emotibit(subjects, emotibit_data_loc)
    left_merged = merge_eye_emotibit(left_processed, emotibit_processed)
    right_merged = merge_eye_emotibit(right_processed, emotibit_processed)
    merge_left_right(left_merged, right_merged, save_loc)
    