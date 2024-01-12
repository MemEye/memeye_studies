#TODO: higher level feature processing (vergence, saccade duration, etc)

# take in the processed eye_0 and eye_1 files for each user, 
# and calculate the saccades and vergence for each eye
# saccades code similar to sams, vergence will need to be calculated
# output is the same files as before, with saccades and vergence added in

import numpy as np
import csv
import pandas as pd
import os
from math import atan2,degrees
from scipy.stats import skew, kurtosis


hampelwindow = 10 # hampel filter (median filter) window size 
L_pupil_base = 3.5
R_pupil_base = 3.5


def hampel_filter_forloop_numba(input_series, window_size, n_sigmas=3):
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    indices = []
    
    for i in range((window_size),(n - window_size)):
        if not np.all(np.isnan(input_series[(i - window_size):(i + window_size)])):
            x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
            S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices

def preprocess_xy(dataframe):
    
    input_df = dataframe.copy()
    
    ## Filtering invalid xy points

    # Replace low confidence points, conf. score < 0.6, with NaN
    input_df.loc[(input_df.confidence < 0.60),'norm_pos_x'] = np.nan
    input_df.loc[(input_df.confidence < 0.60),'norm_pos_y'] = np.nan

    # Replace non-normal, out-of-range values with NaN
    input_df.loc[(input_df.norm_pos_x >= 1.00),'norm_pos_x'] = np.nan
    input_df.loc[(input_df.norm_pos_x <= 0.00),'norm_pos_x'] = np.nan
    input_df.loc[(input_df.norm_pos_y >= 1.00),'norm_pos_y'] = np.nan
    input_df.loc[(input_df.norm_pos_y <= 0.00),'norm_pos_y'] = np.nan

    ## Filtering out outliers 
    # Lan et al. 2020 - median filter with sliding window of 10s

    # Applying Hampel Filter, Median filter
    x, outlier_x = hampel_filter_forloop_numba(input_df.norm_pos_x.to_numpy(), hampelwindow)
    y, outlier_y = hampel_filter_forloop_numba(input_df.norm_pos_y.to_numpy(), hampelwindow)

    # Adding new columns to df for filtered values
    input_df['x_ham'] = x.tolist()
    input_df['y_ham'] = y.tolist()

    ## Spline Interpolation
    input_df['x_int'] = input_df['x_ham'].interpolate(method='spline', order=3)
    input_df['y_int'] = input_df['y_ham'].interpolate(method='spline', order=3)
    
    return input_df

def saccade_params(dataframe, minlen=5, maxvel=40, maxacc=340):

    """Detects saccades, defined as consecutive samples with an inter-sample
    velocity of over a velocity threshold or an acceleration threshold
    arguments

    x - numpy array of x positions
    y - numpy array of y positions
    time - numpy array of tracker timestamps in milliseconds

    keyword arguments

    missing - value to be used for missing data (default = 0.0)
    minlen - minimal length of saccades in milliseconds; all detected
                saccades with len(sac) < minlen will be ignored
                (default = 5)
    maxvel - velocity threshold in pixels/second (default = 40)
    maxacc - acceleration threshold in pixels / second**2
                (default = 340)

    returns
    Ssac, Esac
            Ssac - list of lists, each containing [starttime]
            Esac - list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
    """
    
    input_df = dataframe.copy()
    
    # Remove rows with missing x or y
    xy_df = input_df.dropna(subset=['norm_pos_x', 'norm_pos_y'])
    xy_df = xy_df.reset_index()
    width = 1920
    height = 1080
    x = xy_df['norm_pos_x']*width # convert to pixels
    y = xy_df['norm_pos_y']*height 
    time = xy_df['pupil_timestamp']*1000 # convert to ms

    Ssac = []
    Esac = []
    
    # INTER-SAMPLE MEASURES
    # the distance between samples is the square root of the sum
    # of the squared horizontal and vertical interdistances
    intdist = (np.diff(x)**2 + np.diff(y)**2)**0.5
    # get inter-sample times
    inttime = np.diff(time)
    # recalculate inter-sample times to seconds
    inttime = inttime / 1000.0
    
    # VELOCITY AND ACCELERATION
    # the velocity between samples is the inter-sample distance
    # divided by the inter-sample time
    vel = intdist / inttime
    # the acceleration is the sample-to-sample difference in
    # eye movement velocity
    acc = np.diff(vel)

    # SACCADE START AND END
    t0i = 0
    stop = False
    while not stop:
        # saccade start (t1) is when the velocity or acceleration
        # surpass threshold, saccade end (t2) is when both return
        # under threshold
    
        # detect saccade starts
        sacstarts = np.where((vel[1+t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
        if len(sacstarts) > 0:
            # timestamp for starting position
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time)-1:
                t1i = len(time)-2
            t1 = time[t1i]

            # add to saccade starts
            Ssac.append([t1])

            # detect saccade endings
            sacends = np.where((vel[1+t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
            if len(sacends) > 0:
                # timestamp for ending position
                t2i = sacends[0] + 1 + t1i + 2
                if t2i >= len(time):
                    t2i = len(time)-1
                t2 = time[t2i]
                dur = t2 - t1

                # ignore saccades that did not last long enough
                if dur >= minlen:
                    # add to saccade ends
                    Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                else:
                    # remove last saccade start on too low duration
                    Ssac.pop(-1)

                # update t0i
                t0i = 0 + t2i
            else:
                stop = True
        else:
            stop = True
    
    #print(Ssac)
    #print(Esac)
        
    dur_list = []
    len_list = []
    vel_list = []
    ang_list = []
    dir_list = []
    start_list = []
    end_list = []
    zones = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    
    if len(Esac) > 0:
        for i in Esac:
            dur = i[2]
            dur_list.append(dur)
            start = i[0]
            start_list.append(start)
            end = i[1]
            end_list.append(end)
            # TODO: maybe add check that end-start == duration (or close enough)
            dist = np.sqrt((i[5] - i[3])**2 + (i[6] - i[4])**2)
            len_list.append(dist)
            vel = dist/dur
            vel_list.append(vel)
            ang = degrees(atan2((i[6] - i[4]), (i[5] - i[3])))
            if ang < 0:
                angle = 360 + ang
            else:
                angle = ang
            ang_list.append(angle)
            zone_idx = round(angle / 45)
            dir_list.append(zones[zone_idx])
    else:
        dur_list = []
        len_list = []
        vel_list = []
        ang_list = []
        
    array_dur = np.array(dur_list)
    array_len = np.array(len_list)
    array_vel = np.array(vel_list)
    array_ang = np.array(ang_list)
    array_dir = np.array(dir_list)
    
    ## word book

    if len(array_dur) > 0:
        sac_count = array_dur.size
        sac_dur_mean = array_dur.mean()
        sac_dur_median = np.median(array_dur)
        sac_dur_max = array_dur.max()
        sac_dur_var = array_dur.var()
        sac_dur_sd = array_dur.std()
        sac_dur_skew = skew(array_dur)
        sac_dur_kurt = kurtosis(array_dur)
        
        sac_len_mean = array_len.mean()
        sac_len_median = np.median(array_len)
        sac_len_max = array_len.max()
        sac_len_var = array_len.var()
        sac_len_sd = array_len.std()
        sac_len_skew = skew(array_len)
        sac_len_kurt = kurtosis(array_len)
        
        sac_vel_mean = array_vel.mean()
        sac_vel_median = np.median(array_vel)
        sac_vel_max = array_vel.max()
        sac_vel_var = array_vel.var()
        sac_vel_sd = array_vel.std()
        sac_vel_skew = skew(array_vel)
        sac_vel_kurt = kurtosis(array_vel)
        
        sac_dir_N = (array_dir == 'N').sum()
        sac_dir_NE = (array_dir == 'NE').sum()
        sac_dir_E = (array_dir == 'E').sum()
        sac_dir_SE = (array_dir == 'SE').sum()
        sac_dir_S = (array_dir == 'S').sum()
        sac_dir_SW = (array_dir == 'SW').sum()
        sac_dir_W = (array_dir == 'W').sum()
        sac_dir_NW = (array_dir == 'NW').sum()
    
    else: # no saccades
        sac_count = 0
        sac_dur_mean = 0
        sac_dur_median = 0
        sac_dur_max = 0
        sac_dur_var = 0
        sac_dur_sd = 0
        sac_dur_skew = 0
        sac_dur_kurt = 0
        
        sac_len_mean = 0
        sac_len_median = 0
        sac_len_max = 0
        sac_len_var = 0
        sac_len_sd = 0
        sac_len_skew = 0
        sac_len_kurt = 0
        
        sac_vel_mean = 0
        sac_vel_median = 0
        sac_vel_max = 0
        sac_vel_var = 0
        sac_vel_sd = 0
        sac_vel_skew = 0
        sac_vel_kurt = 0
        
        sac_dir_N = 0
        sac_dir_NE = 0
        sac_dir_E = 0
        sac_dir_SE = 0
        sac_dir_S = 0
        sac_dir_SW = 0
        sac_dir_W = 0
        sac_dir_NW = 0
        
    print(array_len.mean())

    sac_result = {
        'sac_count': sac_count,
        'sac_array_dir': array_dir,
        'sac_array_len': array_len,
        'sac_dur_mean': sac_dur_mean,
        'sac_dur_median': sac_dur_median,
        'sac_dur_max': sac_dur_max,
        'sac_dur_var': sac_dur_var,
        'sac_dur_sd': sac_dur_sd,
        'sac_dur_skew': sac_dur_skew,
        'sac_dur_kurt': sac_dur_kurt,
        'sac_len_mean': sac_len_mean,
        'sac_len_median': sac_len_median,
        'sac_len_max': sac_len_max,
        'sac_len_var': sac_len_var,
        'sac_len_sd': sac_len_sd,
        'sac_len_skew': sac_len_skew,
        'sac_len_kurt': sac_len_kurt,
        'sac_vel_mean': sac_vel_mean,
        'sac_vel_median': sac_vel_median,
        'sac_vel_max': sac_vel_max,
        'sac_vel_var': sac_vel_var,
        'sac_vel_sd': sac_vel_sd,
        'sac_vel_skew': sac_vel_skew,
        'sac_vel_kurt': sac_vel_kurt,
        'sac_dir_N': sac_dir_N,
        'sac_dir_NE': sac_dir_NE,
        'sac_dir_E': sac_dir_E,
        'sac_dir_SE': sac_dir_SE,
        'sac_dir_S': sac_dir_S,
        'sac_dir_SW': sac_dir_SW,
        'sac_dir_W': sac_dir_W,
        'sac_dir_NW': sac_dir_NW
    }

    return sac_result

# TODO: psuedocode, change as needed, nothing is set in stone :)
def convert_sac_to_df(sac_result):
    return pd.DataFrame(sac_result)

def append_sac_result(df, sac_result):
    return pd.concat([df, sac_result], ignore_index=True)

def run_on_segment(files_loc, subject_id):
    segments = ['segmented_left/learning', 'segmented_left/negative', 'segmented_left/practice', 'segmented_left/recall', 'segmented_left/recognition_new', 'segmented_left/recognition_familiar, segmented_right/learning', 'segmented_right/negative', 'segmented_right/practice', 'segmented_right/recall', 'segmented_right/recognition_new', 'segmented_right/recognition_familiar']

    files_loc = f'/Users/kevinzhu/Desktop/MemEye/pupil_segmented_new/{subject_id}' # fill this in
    for segment in segments:
        files = []
        segment_path = os.path.join(files_loc, segment)
        files += [os.path.join(segment_path, f) for f in os.listdir(segment_path) if f.endswith('.csv')]
        
        for file in files:
            df = pd.read_csv(file)
            new_segment = os.path.dirname(file)
            save_path = os.path.join(files_loc, new_segment)
            os.makedirs(save_path, exist_ok=True)

            df1 = preprocess_xy(df)
            sac_result = saccade_params(df)
            sac_result = convert_sac_to_df(sac_result)
            combined = append_sac_result(df, sac_result)

            output_sav_path = os.path.join(save_path, 'SOMETHING.csv')
            combined.to_csv(output_sav_path)


def run(subjects, data_loc):
    for subject_id in subjects:
        #TODO: pull out the file paths so its not hard codded in a fxn
        run_on_segment('/Users/kevinzhu/Desktop/MemEye/pupil_segmented_new', subject_id)

if __name__=='__main__':
    num_subjects = 1
    subjects = list(range(101, 100+num_subjects+1))
    run(subjects)
