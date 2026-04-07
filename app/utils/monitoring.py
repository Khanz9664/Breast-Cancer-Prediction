import pandas as pd
import os
from datetime import datetime

LOG_FILE = "logs/prediction_history.csv"


def init_logging():
    """Ensure the logs directory and history file exist."""
    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists(LOG_FILE):
        # We don't know the features yet, so we'll init when the first log happens
        return False
    return True


def log_prediction(input_df, result, confidence):
    """Log a single prediction to the CSV file."""
    init_logging()

    log_entry = input_df.copy()
    log_entry['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry['prediction'] = result
    log_entry['confidence'] = confidence

    # Append to CSV
    if not os.path.exists(LOG_FILE):
        log_entry.to_csv(LOG_FILE, index=False)
    else:
        log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)


def calculate_drift(current_input, training_df, features):
    """
    Calculate data drift using Z-score.
    Returns a dictionary of drift metrics per feature.
    """
    drift_report = {}
    total_drift_score = 0

    for feature in features:
        if feature in training_df.columns:
            train_mean = training_df[feature].mean()
            train_std = training_df[feature].std()

            # Use the first row of current_input (since it's a single sample)
            val = current_input[feature].iloc[0]

            # Calculate Z-score
            z_score = (val - train_mean) / train_std if train_std != 0 else 0

            # Status: Healthy if |Z| < 2, Warning if |Z| < 3, Critical if |Z| >= 3
            status = "Healthy"
            if abs(z_score) >= 3:
                status = "Critical"
                total_drift_score += 1
            elif abs(z_score) >= 2:
                status = "Warning"
                total_drift_score += 0.5

            drift_report[feature] = {
                'value': val,
                'mean': train_mean,
                'std': train_std,
                'z_score': z_score,
                'status': status
            }

    # System health percentage
    health_score = max(0, 100 - (total_drift_score / len(features) * 100))

    return {
        'metrics': drift_report,
        'health_score': health_score,
        'is_drifting': total_drift_score > (len(features) * 0.2)  # Threshold: 20% features drifting
    }


def load_prediction_history():
    """Load the history of predictions."""
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame()
