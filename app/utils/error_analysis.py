
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


def get_error_analysis_data(model, scaler, df, features, current_threshold=0.5):
    """
    Performs a localized evaluation to identify misclassifications
    and error patterns, including ROC metrics.
    """
    X = df[features]
    y = df['diagnosis']

    # Consistent split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Use threshold for predictions
    y_pred = (y_prob >= current_threshold).astype(int)

    # Identify misclassifications
    test_results = X_test.copy()
    test_results['Actual'] = y_test
    test_results['Predicted'] = y_pred
    test_results['Probability'] = y_prob

    misclassified = test_results[test_results['Actual'] != test_results['Predicted']].copy()
    misclassified['Error_Type'] = np.where(misclassified['Predicted'] == 1, 'False Positive', 'False Negative')  # noqa: E501

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC Curve Metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        'misclassified': misclassified,
        'cm': cm,
        'total_test': len(y_test),
        'total_errors': len(misclassified),
        'accuracy': (len(y_test) - len(misclassified)) / len(y_test),
        'roc_data': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
    }


def analyze_error_patterns(misclassified):
    """Identifies which features are most extreme in misclassified samples."""
    if misclassified.empty:
        return "No errors identified in the current test split."

    # Analyze if errors happen at high/low ranges for certain features
    # (Simple logic: what's the mean of top feature in FP vs FN)
    return misclassified.groupby('Error_Type').mean().drop(columns=['Actual', 'Predicted', 'Probability'])
