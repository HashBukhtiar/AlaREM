import os
import mne

# Load the EDF file
edf_file = os.path.join('data', 'physionet', 'sleep-cassette', 'SC4001E0-PSG.edf')
# edf_file = os.path.join('data', 'physionet', 'sleep-telemetry', 'ST7011J0-PSG.edf')
raw = mne.io.read_raw_edf(edf_file, preload=True)

# print basic info
print(raw.info)

# convert whole thing to dataframe
df = raw.to_data_frame()
print(df.head())
print(df.tail())