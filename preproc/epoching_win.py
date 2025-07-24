# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:27:19 2022

@author: Vincent
"""
import os
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids, print_dir_tree, make_report, print_dir_tree, inspect_dataset
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
import numpy as np
import pandas as pd
from os import path

# project folders
bids_dir    = "D:\\000CurrentDir\\EEGManyPipelines_vincent\\eeg_BIDS"
codes_dir   = "D:\\000CurrentDir\\EEGManyPipelines_vincent\\EEGManyPipelines_QingWang"
derive_dir  = "D:\\000CurrentDir\\EEGManyPipelines_vincent\\derivative\\preproc"
epoch_dir   = "D:\\000CurrentDir\\EEGManyPipelines_vincent\\derivative\\epochs"

# subject list
sub_list = [f"{a:03}" for a in range(1,34,1)]

# sub eeg and events files
sub_eeg_list    = [bids_dir+"\\sub-"+x+"\\eeg\\sub-"+x+"_task-xxxx_eeg.vhdr"   for x in sub_list]

sub_eeg_preproc_list = [derive_dir+"\\sub-"+x+"\\sub-"+x+"_task-xxxx_desc-preproc_eeg.fif"   for x in sub_list]

sub_epoch_file_list  = [ epoch_dir+"\\sub-"+x+"\\sub-"+x+"_task-xxxx_desc-epoched_epo.fif"   for x in sub_list]

sub_report_file_list  = [ epoch_dir+"\\sub-"+x+"_report.html"   for x in sub_list]

sub_folder_report_file_list = [ epoch_dir+"\\sub-"+x+"\\sub-"+x+"_report.html"               for x in sub_list]

sub_raw_eeg_dict = dict(zip(sub_list, sub_eeg_list))
sub_eeg_dict = dict(zip(sub_list , sub_eeg_preproc_list))
sub_epoch_dict = dict(zip(sub_list , sub_epoch_file_list))
sub_report_dict = dict(zip(sub_list , sub_report_file_list))
sub_folder_report_dict = dict(zip(sub_list , sub_folder_report_file_list))

# overall triger data
trigger_data_dir = codes_dir+"\\tab_data\\TriggerTable.csv"
trigger_df = pd.read_csv(trigger_data_dir, sep=',');


# Epoching block
# Epoch rejection threshold: 200 µV, comparatively conservative
reject_criteria = dict(eeg=200e-6) 
# Using -0.5~0 as baseline for activation correction.
baseline_th = (-0.5, 0)
# preserving larger time span (-0.5, 1) to include baselien, N1 (150~200ms) and 300~500ms
T_MIN = -0.5
T_MAX = 1

# full trigger data
trigger_codes=trigger_df.trigger.unique();
full_trigger_dict=dict(zip(["Stimulus/"+str(x) for x in trigger_codes], trigger_codes))

# loop over for epoching
for sub_id in sub_list:
    ## read eeg data sub_list
    print("sub-"+sub_id+": reading preproc eeg...")
    eeg = mne.io.read_raw_fif(sub_eeg_dict[sub_id], preload=True)
    
    # updating events coding
    events_from_annot, event_dict = mne.events_from_annotations(eeg, event_id=full_trigger_dict)
    events_keys=list(set(event_dict.keys()) & set(full_trigger_dict.keys()))
    updated_events_dict={}
    for x in events_keys:
        updated_events_dict[x]=full_trigger_dict[x]
    # epochs rejection
    print("sub-"+sub_id+": epoching...")
    epochs = mne.Epochs(eeg, events_from_annot, event_id=updated_events_dict, baseline = baseline_th,
                        tmin=T_MIN, tmax=T_MAX, reject=reject_criteria,
                        preload=True, on_missing="warn");
    epochs.add_annotations_to_metadata(overwrite=True)
    epoch_sub_dir = epoch_dir+"\\sub-"+sub_id
    if (not path.exists(epoch_sub_dir)):
        os.mkdir(epoch_sub_dir) 
    epochs.save(sub_epoch_dict[sub_id], overwrite=True)
    # Generate report
    report = mne.Report(title=sub_id+' Report')
    report.add_raw(raw=eeg, title='Basic Info', psd=True)   # omit PSD plot
    report.add_events(events=events_from_annot, title='Events from annotations', sfreq=eeg.info["sfreq"])
    report.add_epochs(epochs=epochs, title="Epochs")
    report.save(sub_report_dict[sub_id], overwrite=True)
    report.save(sub_folder_report_dict[sub_id], overwrite=True)
