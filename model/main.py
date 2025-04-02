from preprocessing_functions import *
from model_training import *

def main():
    all_epochs_power_bands_df = preprocess_features(preprocess_features=False, download_files=False)
    labelled_epochs_power_bands_df = preprocess_labels(all_epochs_power_bands_df, preprocess_labels=False, download_files=False)

    model = train_model(
        labelled_epochs_power_bands_df, 
        train_type='rapid',
        model_id=32, 
        use_all_regions=True,
        use_ratios=True, 
        hidden_layer_sizes=(100, 50), 
        activation='relu',
        solver='adam',
        max_iter=200,
        lambda_l2=0.0001 
    )

    return

if __name__ == "__main__":
    main()