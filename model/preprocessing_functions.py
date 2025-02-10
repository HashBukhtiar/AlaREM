import os
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_power_bands(signal, sampling_frequency):
    freqs = np.fft.rfftfreq(len(signal), d=1/sampling_frequency)
    fft_vals = np.abs(np.fft.rfft(signal))**2

    total_power = np.sum(fft_vals)
    power_bands = {
        'subdelta': round(np.sum(fft_vals[freqs < 0.5]) / total_power, 5),
        'delta': round(np.sum(fft_vals[(freqs >= 0.5) & (freqs < 4)]) / total_power, 5),
        'theta': round(np.sum(fft_vals[(freqs >= 4) & (freqs < 8)]) / total_power, 5),
        'alpha': round(np.sum(fft_vals[(freqs >= 8) & (freqs < 12)]) / total_power, 5),
        'beta': round(np.sum(fft_vals[(freqs >= 12) & (freqs < 30)]) / total_power, 5),
        'gamma': round(np.sum(fft_vals[(freqs >= 30)]) / total_power, 5)
    }
    
    return power_bands

def load_edf_files(edf_files):
    for edf_file in edf_files:
        if 'Hypnogram' in edf_file:
            continue
        yield edf_file

def process_edf_file(edf_file):
    if edf_file[1] == 'T':
        raw = mne.io.read_raw_edf(os.path.join('data', 'physionet', 'sleep-telemetry', edf_file), preload=True, verbose=False)
        data_type = 'telemetry'
    elif edf_file[1] == 'C':
        raw = mne.io.read_raw_edf(os.path.join('data', 'physionet', 'sleep-cassette', edf_file), preload=True, verbose=False)
        data_type = 'cassette'
    else:
        raise ValueError('Invalid file name')
    
    sampling_frequency = raw.info['sfreq']
    subject_number = edf_file[3:5]
    night_number = edf_file[5]
    
    df = raw.to_data_frame()
    df = df[['time', 'EEG Fpz-Cz', 'EEG Pz-Oz']]
    df.columns = ['time', 'eegAnterior', 'eegPosterior']
    df['type'] = data_type
    df['subject'] = subject_number
    df['night'] = night_number
    df['epochNum'] = ((df['time'] - df['time'][0]) // 30).astype(int) # new epoch assigned for every 30 seconds
    df['epochId'] = data_type + '-' + subject_number + '-' + night_number + '-' + df['epochNum'].apply(lambda x: f"{x:04d}")

    return df, sampling_frequency

def compute_power_bands_for_epochs(df, sampling_frequency):
    epochs = df.groupby('epochId')
    power_bands_list = []

    for epoch_id, epoch_df in epochs:
        eeg_anterior = epoch_df['eegAnterior'].values
        eeg_posterior = epoch_df['eegPosterior'].values

        power_bands_anterior = compute_power_bands(eeg_anterior, sampling_frequency)
        power_bands_posterior = compute_power_bands(eeg_posterior, sampling_frequency)

        power_bands = {
            'epochId': epoch_id,
            **{f'anterior_{band}': power for band, power in power_bands_anterior.items()},
            **{f'posterior_{band}': power for band, power in power_bands_posterior.items()}
        }
        power_bands_list.append(power_bands)

    return pd.DataFrame(power_bands_list)

def preprocess_data(preprocess_data, download_files):

    if preprocess_data:
        edf_files = []
        edf_files.extend(os.listdir(os.path.join('data', 'physionet', 'sleep-cassette')))
        edf_files.extend(os.listdir(os.path.join('data', 'physionet', 'sleep-telemetry')))
        edf_files = [edf_file for edf_file in edf_files if 'Hypnogram' not in edf_file]
        all_epochs_power_bands_df = []

        for edf_file in tqdm(edf_files, desc='Processing Nights'):
            raw_data_df, sampling_freq = process_edf_file(edf_file)
            epochs_power_bands_df = compute_power_bands_for_epochs(raw_data_df, sampling_freq)
            all_epochs_power_bands_df.append(epochs_power_bands_df)

        all_epochs_power_bands_df = pd.concat(all_epochs_power_bands_df, ignore_index=True)

        if download_files:
            all_epochs_power_bands_df.to_csv(os.path.join('data', 'physionet', 'frequency_spectrum_data.csv'), index=False)
            print('Data saved to data/physionet/frequency_spectrum_data.csv')
            print(f"File Size: {all_epochs_power_bands_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

        else:
            all_epochs_power_bands_df = pd.read_csv(os.path.join('data', 'physionet', 'frequency_spectrum_data.csv'))

    return all_epochs_power_bands_df