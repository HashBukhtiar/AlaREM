from preprocessing_functions import *
from model_training import *

def main():
    all_epochs_power_bands_df = preprocess_features(preprocess_features=False, download_files=False)
    labelled_epochs_power_bands_df = preprocess_labels(all_epochs_power_bands_df, preprocess_labels=False, download_files=False)

    # print(labelled_epochs_power_bands_df)
    # print(labelled_epochs_power_bands_df.describe().T)
    # print(labelled_epochs_power_bands_df['sleep_stage'].value_counts())

    # Train the model
    model = train_model(labelled_epochs_power_bands_df, train_type='cross_validation')

    return

if __name__ == "__main__":
    main()