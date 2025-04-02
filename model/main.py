from preprocessing_functions import *
from model_training import *

def main():
    all_epochs_power_bands_df = preprocess_features(preprocess_features=False, download_files=False)
    labelled_epochs_power_bands_df = preprocess_labels(all_epochs_power_bands_df, preprocess_labels=False, download_files=False)

    model = train_model(
        labelled_epochs_power_bands_df, 
        train_type='cross-validation',
        model_id=34, 
        use_all_regions=True,
        use_ratios=True, 
        hidden_layer_sizes=(256, 128, 64, 32), 
        activation='relu',
        learning_rate='adaptive',
        solver='adam',
        max_iter=200,
    )

    return

if __name__ == "__main__":
    main()