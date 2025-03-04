from preprocessing_functions import *

def main():
    all_epochs_power_bands_df = preprocess_features(preprocess_features=False, download_files=False)
    labelled_epochs_power_bands_df = preprocess_labels(all_epochs_power_bands_df, preprocess_labels=True, download_files=True)

    print(labelled_epochs_power_bands_df)
    print(labelled_epochs_power_bands_df.describe().T)
    print(labelled_epochs_power_bands_df['sleep_stage'].value_counts())

    return

if __name__ == "__main__":
    main()