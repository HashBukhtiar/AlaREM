import time
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, log_loss, confusion_matrix, precision_recall_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_model(labelled_epochs_power_bands_df, train_type):
    start_time = time.time()
    train_df = labelled_epochs_power_bands_df.copy(deep=True)
    train_df['person'] = train_df['epochId'].apply(lambda x: x.split('-')[0][0] + x.split('-')[1])
    train_df = train_df[~train_df['sleep_stage'].isin(['N', '?', 'M'])]

    features = ['anterior_subdelta', 'anterior_delta', 'anterior_theta', 'anterior_alpha', 'anterior_beta', 'anterior_gamma']
    label = 'sleep_stage'

    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)

    if train_type == 'rapid':

        X = train_df[features]
        y = train_df[label].apply(lambda x: 1 if x in ('1', '2') else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_prob)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_log_loss = log_loss(y_train, y_train_prob)
        train_conf_matrix = confusion_matrix(y_train, y_train_pred)
        train_precision_lst, train_recall_lst, _ = precision_recall_curve(y_train, y_train_prob)
        train_auc_pr = auc(train_recall_lst, train_precision_lst)
        train_mcc = matthews_corrcoef(y_train, y_train_pred)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_log_loss = log_loss(y_test, y_test_prob)
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)
        test_precision_lst, test_recall_lst, _ = precision_recall_curve(y_test, y_test_prob)
        test_auc_pr = auc(test_recall_lst, test_precision_lst)
        test_mcc = matthews_corrcoef(y_test, y_test_pred)

    elif train_type == 'cross_validation':
        train_df = train_df.sample(frac=0.35, random_state=42) ### DELETE THIS

        # initialize lists to store metrics for each fold
        train_metrics = {'accuracy': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': [], 'log_loss': [], 'auc_pr': [], 'mcc': []}
        test_metrics = {'accuracy': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': [], 'log_loss': [], 'auc_pr': [], 'mcc': []}
        train_conf_matrices = []
        test_conf_matrices = []

        # perform LOOCV variant
        folds = 0
        for person in tqdm(train_df['person'].unique()):
            folds += 1
            X_train = train_df[train_df['person'] != person][features]
            y_train = train_df[train_df['person'] != person][label].apply(lambda x: 1 if x in ('1', '2') else 0)
            X_test = train_df[train_df['person'] == person][features]
            y_test = train_df[train_df['person'] == person][label].apply(lambda x: 1 if x in ('1', '2') else 0)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]

            precision, recall, _ = precision_recall_curve(y_train, y_train_prob)
            recall, precision = zip(*sorted(zip(recall, precision)))
            train_metrics['auc_pr'].append(auc(recall, precision))

            train_metrics['accuracy'].append(accuracy_score(y_train, y_train_pred))
            train_metrics['roc_auc'].append(roc_auc_score(y_train, y_train_prob))
            train_metrics['precision'].append(precision_score(y_train, y_train_pred))
            train_metrics['recall'].append(recall_score(y_train, y_train_pred))
            train_metrics['f1'].append(f1_score(y_train, y_train_pred))
            train_metrics['log_loss'].append(log_loss(y_train, y_train_prob))
            train_metrics['mcc'].append(matthews_corrcoef(y_train, y_train_pred))

            precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
            recall, precision = zip(*sorted(zip(recall, precision)))
            test_metrics['auc_pr'].append(auc(recall, precision))

            test_metrics['accuracy'].append(accuracy_score(y_test, y_test_pred))
            test_metrics['roc_auc'].append(roc_auc_score(y_test, y_test_prob))
            test_metrics['precision'].append(precision_score(y_test, y_test_pred))
            test_metrics['recall'].append(recall_score(y_test, y_test_pred))
            test_metrics['f1'].append(f1_score(y_test, y_test_pred))
            test_metrics['log_loss'].append(log_loss(y_test, y_test_prob))
            test_metrics['mcc'].append(matthews_corrcoef(y_test, y_test_pred))

            train_conf_matrices.append(confusion_matrix(y_train, y_train_pred))
            test_conf_matrices.append(confusion_matrix(y_test, y_test_pred))

        # Calculate average metrics
        train_accuracy = sum(train_metrics['accuracy']) / folds
        train_roc_auc = sum(train_metrics['roc_auc']) / folds
        train_precision = sum(train_metrics['precision']) / folds
        train_recall = sum(train_metrics['recall']) / folds
        train_f1 = sum(train_metrics['f1']) / folds
        train_log_loss = sum(train_metrics['log_loss']) / folds
        train_auc_pr = sum(train_metrics['auc_pr']) / folds
        train_mcc = sum(train_metrics['mcc']) / folds

        test_accuracy = sum(test_metrics['accuracy']) / folds
        test_roc_auc = sum(test_metrics['roc_auc']) / folds
        test_precision = sum(test_metrics['precision']) / folds
        test_recall = sum(test_metrics['recall']) / folds
        test_f1 = sum(test_metrics['f1']) / folds
        test_log_loss = sum(test_metrics['log_loss']) / folds
        test_auc_pr = sum(test_metrics['auc_pr']) / folds
        test_mcc = sum(test_metrics['mcc']) / folds

        # Calculate average confusion matrices
        train_conf_matrix = sum(train_conf_matrices)
        test_conf_matrix = sum(test_conf_matrices)

    print('-- TRAINING METRICS --')
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Train ROC AUC: {train_roc_auc}")
    print(f"Train Precision: {train_precision}")
    print(f"Train Recall: {train_recall}")
    print(f"Train F1-score: {train_f1}")
    print(f"Train Log Loss: {train_log_loss}")
    print(f"Train AUC-PR: {train_auc_pr}")
    print(f"Train MCC: {train_mcc}")
    print(f"Train Confusion Matrix:\n{train_conf_matrix}")

    print('\n-- TESTING METRICS --')
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test ROC AUC: {test_roc_auc}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test F1-score: {test_f1}")
    print(f"Test Log Loss: {test_log_loss}")
    print(f"Test AUC-PR: {test_auc_pr}")
    print(f"Test MCC: {test_mcc}")
    print(f"Test Confusion Matrix:\n{test_conf_matrix}")

    # Model performance summary metadata
    print('\n -- MODEL PERFORMANCE SUMMARY --')
    test_tp = test_conf_matrix[1][1]
    test_tn = test_conf_matrix[0][0]
    test_fp = test_conf_matrix[0][1]
    test_fn = test_conf_matrix[1][0]
    train_tp = train_conf_matrix[1][1]
    train_tn = train_conf_matrix[0][0]
    train_fp = train_conf_matrix[0][1]
    train_fn = train_conf_matrix[1][0]
    end_time = time.time()
    training_time = end_time - start_time
    print(f"{test_mcc}, {test_auc_pr}, {test_roc_auc}, {test_f1}, {test_precision}, {test_recall}, {test_log_loss}, {test_accuracy}, {test_tp}, {test_tn}, {test_fp}, {test_fn}, {train_mcc}, {train_auc_pr}, {train_roc_auc}, {train_f1}, {train_precision}, {train_recall}, {train_log_loss}, {train_accuracy}, {train_tp}, {train_tn}, {train_fp}, {train_fn}, {training_time}\n")

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    auc_pr = auc(recall, precision)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ROC Curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % test_roc_auc)
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")

    # Precision-Recall Curve
    axes[0, 1].plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve (AUC = %0.3f)' % auc_pr)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")

    # Confusion Matrix
    sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Other', 'N1/N2 Sleep'], yticklabels=['Other', 'N1/N2 Sleep'], ax=axes[1, 0], cbar=False)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')

    # Metrics Table
    metrics_data = {
        'Metric': ['MCC', 'AUC-PR', 'F1-Score', 'ROC AUC', 'Log Loss', 'Precision', 'Recall', 'Accuracy'],
        'Testing': [round(test_mcc, 4), round(test_auc_pr, 4), round(test_f1, 4), round(test_roc_auc, 4), round(test_log_loss, 4), round(test_precision, 4), round(test_recall, 4), round(test_accuracy, 4)],
        'Training': [round(train_mcc, 4), round(train_auc_pr, 4), round(train_f1, 4), round(train_roc_auc, 4), round(train_log_loss, 4), round(train_precision, 4), round(train_recall, 4), round(train_accuracy, 4)],
        'GenRatio': [
            round(test_mcc / train_mcc, 4),
            round(test_auc_pr / train_auc_pr, 4),
            round(test_f1 / train_f1, 4),
            round(test_roc_auc / train_roc_auc, 4),
            round(test_log_loss / train_log_loss, 4),
            round(test_precision / train_precision, 4),
            round(test_recall / train_recall, 4),
            round(test_accuracy / train_accuracy, 4)
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        cell.set_height(0.1)
    axes[1, 1].set_title('Metrics Summary')

    plt.tight_layout(pad=3.0)
    plt.show()

    return model