#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 06:52:12 2023

@author: Qing Wang (Vincent)
"""
import os
from pathlib import Path
import os.path
import pickle

#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
#from mne_bids import BIDSPath, read_raw_bids, write_raw_bids, print_dir_tree, make_report, print_dir_tree, inspect_dataset
#from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
import numpy as np
import pandas as pd
from datetime import datetime


# project folders
proj_dir    = "/scratch/tms_eeg/"
project_path = Path(proj_dir)

data_path = project_path / "data"
raw_data_path = data_path / "raw_eeglab_set"

## 
visit_list = [1, 2, 3]
raw_event_mapping = {'0, Impedance': 1, '1001, pre-open': 2, '1002, pre-close': 3}
target_event_dict = {'open':'rs-eyesOpen', 'close': 'rs-eyesClose'}
event_mapping     = {'rs-eyesOpen': 1, 'rs-eyesClose': 2}

def find_event(event_dict, tar_string_list):
    # search for event with tar_string and return event coding.
    res_dict = {}
    for k, v in event_dict.items():
        for tar_string in tar_string_list:
            if tar_string.lower() in k.lower():
                res_dict[tar_string] = v
    return res_dict

def find_event_timing(events_data, event_dict):
    # function to correct the event marker from 2 short event to a duration.
    event_codes = list(event_dict.values())
    pop_index = []
    for i in range(len(events_data)):
        if events_data[i][2] in event_codes:
            if i == 0:
                pre_ = np.nan
            else:
                pre_ = events_data[i-1][2]
            if events_data[i][2] == pre_:
                events_data[i-1][1] = events_data[i][0] - events_data[i-1][0]
                pop_index.append(i)
        else:
            pop_index.append(i)
    events_data = np.delete(events_data, pop_index, 0)
    return events_data

# proc parameters
resample_freq = 100
pass_band  = [0.4, 46]
Delta_band = [0.5, 4]
Theta_band = [4, 8]
Alpha_band = [8, 12]
Beta_band  = [12, 30]
Gama_band  = [30, 45] #

vis_dict = dict(mag=1e-12, grad=4e-11, eeg=10e-5, eog=150e-6, ecg=5e-4,
                emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                resp=1, chpi=1e-4, whitened=1e2)

selected_bands = {'delta': Delta_band, 'theta': Theta_band, 'alpha': Alpha_band, 'beta': Beta_band, 'gamma':Gama_band}

#
subj_list = os.listdir(raw_data_path)
res_dict = {}
err_dict = {}
err_dict['missing_event'] = {}
err_dict['missing_eeg'] = {}
# loop over subjects for preprocessing
for subj_id in subj_list:
    print(subj_id)
    res_dict[subj_id]={}
    for _visit in visit_list:
        res_dict[subj_id][_visit]={}
        subj_eeg_visit_file_path = raw_data_path / subj_id / (subj_id+"_ses-"+str(_visit)+"_eeg.set")
        if not os.path.isfile(subj_eeg_visit_file_path):
            err_dict['missing_eeg'][subj_id]=_visit
            continue
        else:
            subj_eeg = mne.io.read_raw_eeglab(subj_eeg_visit_file_path, uint16_codec="utf-8")
            # adding montage
            # montage = mne.channels.make_standard_montage(montage_name)
            # subj_eeg.set_montage("biosemi64")
            # creating new annotations: event of interest
            # EEG info
            # 1. downsample EEG
            subj_eeg = subj_eeg.resample(resample_freq)
            # 2. reference EEG
            subj_eeg = subj_eeg.set_eeg_reference(ref_channels=['M1', 'M2'])
            # 3. init filtering
            subj_eeg=subj_eeg.filter(h_freq=1, l_freq=None, l_trans_bandwidth='auto', filter_length='auto', phase='zero')
            # 3. get events and crop data
            events_from_annot, event_dict = mne.events_from_annotations(subj_eeg)
            event_dict = find_event(event_dict, list(target_event_dict.keys()))
            events_data = find_event_timing(events_from_annot, event_dict)
            fs_ = subj_eeg.info['sfreq']
            if len(event_dict)==0:
                err_dict['missing_event'][subj_id]={}
                err_dict['missing_event'][subj_id]["eo"]=_visit
                err_dict['missing_event'][subj_id]["ec"]=_visit
            elif len(event_dict)==1 and ('open' in event_dict.keys()):
                err_dict['missing_event'][subj_id]={}
                err_dict['missing_event'][subj_id]["ec"]=_visit
                eyes_open_annot = mne.Annotations(onset=events_data[0,0]/fs_, duration=events_data[0,1]/fs_, description=list(event_dict.keys())[0])
                eyes_open_eeg  = subj_eeg.crop_by_annotations(annotations=eyes_open_annot)
                eo_psd_df=eyes_open_eeg[0].compute_psd(method='multitaper',  fmin=pass_band[0], fmax=pass_band[1], n_jobs=-1, low_bias=True).to_data_frame()
                res_dict[subj_id][_visit]['eo_psd_df'] = eo_psd_df
            elif len(event_dict)==1 and ('close' in event_dict.keys()):
                err_dict['missing_event'][subj_id]={}
                err_dict['missing_event'][subj_id]["eo"]=_visit
                eyes_close_annot = mne.Annotations(onset=events_data[0,0]/fs_, duration=events_data[0,1]/fs_, description=list(event_dict.keys())[0])
                eyes_close_eeg  = subj_eeg.crop_by_annotations(annotations=eyes_close_annot)
                ec_psd_df=eyes_close_eeg[0].compute_psd(method='multitaper', fmin=pass_band[0], fmax=pass_band[1], n_jobs=-1, low_bias=True).to_data_frame()
                res_dict[subj_id][_visit]['ec_psd_df'] = ec_psd_df
            else:
                # in case of missing: eyes open
                if events_data[0][1] == 0:
                    err_dict['missing_event'][subj_id]={}
                    err_dict['missing_event'][subj_id]["eo"]=_visit
                    continue
                else:
                    eyes_open_annot = mne.Annotations(onset=events_data[0,0]/fs_, duration=events_data[0,1]/fs_, description=list(event_dict.keys())[0])
                    eyes_open_eeg  = subj_eeg.crop_by_annotations(annotations=eyes_open_annot)
                    eo_psd_df=eyes_open_eeg[0].compute_psd(method='multitaper',  fmin=pass_band[0], fmax=pass_band[1], n_jobs=-1, low_bias=True).to_data_frame()
                    res_dict[subj_id][_visit]['eo_psd_df'] = eo_psd_df
                # eyes close
                if events_data[1][1] == 0:
                    err_dict['missing_event'][subj_id] = {}
                    err_dict['missing_event'][subj_id]["ec"] = _visit
                    continue
                else:
                    eyes_close_annot = mne.Annotations(onset=events_data[1,0]/fs_, duration=events_data[1,1]/fs_, description=list(event_dict.keys())[1])
                    eyes_close_eeg = subj_eeg.crop_by_annotations(annotations=eyes_close_annot)
                    ec_psd_df = eyes_close_eeg[0].compute_psd(method='multitaper', fmin=pass_band[0], fmax=pass_band[1], n_jobs=-1, low_bias=True).to_data_frame()
                    res_dict[subj_id][_visit]['ec_psd_df'] = ec_psd_df
                # remove EOG
                #eyes_open_eeg.set_channel_types({'EOG': 'eog'})
                #eog_evoked = mne.preprocessing.create_eog_epochs(eyes_open_eeg, baseline=(-0.5, -0.2))
                #eog_epochs.average().plot()
                #ica = ICA(n_components=15, max_iter="auto", random_state=97)
                #ica.fit(eyes_open_eeg)
                #ica.exclude = []
                # find which ICs match the EOG pattern
                #eog_indices, eog_scores = ica.find_bads_eog(eyes_open_eeg)
                #ica.exclude = eog_indices
                #ica.plot_scores(eog_scores)
                #ica.plot_properties(eyes_open_eeg, picks=eog_indices)
                #ica.plot_sources(raw, show_scrollbars=False)
                #ica.plot_sources(eog_evoked)
                # Compute PSD and take power

        # end of loop
        
f = open(str(data_path/"psd_tmp_20240429.pkl"), 'wb')
pickle.dump(res_dict,f)
f.close()
