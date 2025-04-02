from preprocessing_functions import *
from model_training import *

def main():
    all_epochs_power_bands_df = preprocess_features(preprocess_features=False, download_files=False)
    labelled_epochs_power_bands_df = preprocess_labels(all_epochs_power_bands_df, preprocess_labels=False, download_files=False)

    # Train with LightGBM and all regions
    model = train_model(
        labelled_epochs_power_bands_df, 
        train_type='rapid',
        learning_rate=0.01,
        n_estimators=100,
        max_depth=-1,
        num_leaves=63,
        lambda_l1=0.1,
        lambda_l2=0.1,
        use_all_regions=True
    )

    return

if __name__ == "__main__":
    main()