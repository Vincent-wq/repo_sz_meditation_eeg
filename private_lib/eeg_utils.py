# eeg_utils.py
import os
import pandas as pd
import numpy as np
import mne

def get_dataset_df(data_path):
    """
    Retrieve the .set EEG file paths for each subject and organize them into resting-state and task EEG columns.
    
    Parameters:
        data_path (str): The path containing subject directories.
    
    Returns:
        pd.DataFrame: A DataFrame with columns 'Subject_ID', 'Resting_State_EEG', and 'Task_EEG'.
    """
    data = []  # List to hold subject IDs and their EEG file paths
    
    try:
        # List all directories in the data path
        subject_ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        for subj_id in subject_ids:
            subj_path = os.path.join(data_path, subj_id)
            
            # Find .set files in the subject's folder
            eeg_files = [f for f in os.listdir(subj_path) if f.endswith('.set')]
            eeg_files.sort()  # Sort files to ensure a consistent order
            
            # Assign resting-state and task EEG files based on the rules
            resting_state_file = os.path.join(subj_path, eeg_files[0]) if len(eeg_files) > 1 else ""
            task_file = os.path.join(subj_path, eeg_files[1] if len(eeg_files) > 1 else eeg_files[0]) if eeg_files else ""
            
            # Add the subject and file paths to the data list
            data.append({
                "participant_id": subj_id,
                "rsEEG_file": resting_state_file if resting_state_file else None,
                "taskEEG_file": task_file if task_file else None,
            })
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        print(f"Error retrieving EEG file paths: {e}")
        return pd.DataFrame()

def preproc_eeg(eeg_file_path, pass_band = [0.5, 61], random_STAT = 24, elp_file = 'standard-10-5-cap385_dehead.elp'):
    import mne
    from mne.preprocessing import ICA
    NOTCH_Freq = 50
    ## EEG preproc
    # 0. Read EEG
    raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True)
    print(raw)  # Print information about the raw dataset
    print("Channel names:", len(raw.info['ch_names']), raw.info['ch_names'])
    # Check sampling frequency
    print("Sampling frequency:", raw.info['sfreq'], "Hz")
    # Duration of recording
    duration = raw.n_times / raw.info['sfreq']
    print(f"Recording duration: {duration:.2f} seconds")
    # Check annotations (markers/events)
    print("Annotations (events):", raw.annotations)
    # 1. downsample EEG
    #subj_eeg = raw.resample(resample_freq)
    subj_eeg = raw.copy()
    # 2. setup special channel and montage
    subj_eeg.set_channel_types({'EOG': 'eog'})
    # 3. reference EEG
    subj_eeg_rereference = subj_eeg.set_eeg_reference(ref_channels='average')
    #subj_eeg.plot_psd(fmax=60)
    # 4. init filtering (general filter for ERP)
    subj_eeg_rereference_hp = subj_eeg_rereference.filter(l_freq=pass_band[0], h_freq=None, fir_design='firwin')
    subj_eeg_rereference_hp_lp = subj_eeg_rereference_hp.filter(l_freq=None, h_freq=pass_band[1], fir_design='firwin')
    # notch filter
    subj_eeg_rereference_hp_lp_notched = subj_eeg_rereference_hp_lp.copy().notch_filter(freqs=NOTCH_Freq, picks='eeg', #filter_length='10s',
                                                                                    trans_bandwidth=0.5,  method='fir', phase='minimum', verbose=True )
    # 5. Remove artifact by ICA
    # Detect EOG artifacts using annotations or thresholding
    eog_events = mne.preprocessing.find_eog_events(subj_eeg_rereference_hp_lp_notched)
    # Visualize EOG events
    print(f"Detected {len(eog_events)} EOG artifacts")
    #mne.viz.plot_events(eog_events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)
    # Initialize ICA
    ica = ICA(n_components=20, method='fastica', random_state=random_STAT) # , n_jobs=N_JOBS
    # Fit ICA to the raw data
    ica.fit(subj_eeg_rereference_hp_lp_notched)
    # Automatically find EOG-related components using the EOG channel
    eog_indices, eog_scores = ica.find_bads_eog(subj_eeg_rereference_hp_lp_notched)
    # Mark EOG-related components
    print(f"EOG-related components: {eog_indices}")
    ica.exclude = eog_indices
    # Apply ICA to remove EOG artifacts
    subj_eeg_cleaned = ica.apply(subj_eeg_rereference_hp_lp_notched.copy())
    # setup montage (no EOG in elp file), comment this, in case channel name error
    custome_montage = mne.channels.read_custom_montage(elp_file)
    subj_eeg_cleaned.set_montage(custome_montage)
    return subj_eeg_cleaned

def epoching_by_events(eeg, events_dict_, tmin=-1, tmax = 2, baseline = (-0.1, 0)):
    events, event_id = mne.events_from_annotations(eeg)
    filtered_events_dict = {key: events_dict_[key] for key in event_id.keys() if key in events_dict_}
    print("Filtered events dictionary:", filtered_events_dict)
    event_id_str = {value: int(key) for key, value in filtered_events_dict.items()}
    print("Event ID mapping:", event_id_str)
    mapped_event_id = {filtered_events_dict[key]: value for key, value in event_id.items() if key in filtered_events_dict}
    print("Mapped event IDs:", mapped_event_id)  
    epochs = mne.Epochs(eeg, events, event_id=mapped_event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    return epochs, event_id_str