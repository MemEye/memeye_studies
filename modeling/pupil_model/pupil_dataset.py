import torch
import pandas as pd 
import glob

cols_to_keep = ['pupil_timestamp', 'label', 'norm_pos_x', 'norm_pos_y', 
                'diameter', 'ellipse_center_x', 'ellipse_center_y', 
                'ellipse_axis_a', 'ellipse_axis_b', 'ellipse_angle', 
                'fixations_norm_pos_x', 'fixations_norm_pos_y', 
                'fixations_dispersion', 'is_blink', 'sac_array_dir']

# 101-120 for training
# 121-123 for validation
# 124-132 for testing

def create_fixation_col():
    # is_fixation
    pass

def create_image_col():
    # imagename
    pass

def remap_blink_col():
    # is_blink,
    pass

def remap_sac_dir():
    # sac_array_dir
    pass

def merge_eyes():
    pass



def run(subjects, pupil_data_loc, eye):
    pass


if __name__=='__main__':
    pass



