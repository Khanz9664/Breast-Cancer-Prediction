import numpy as np
import pandas as pd


def generate_synthetic_samples(df, features, n_samples=5, class_target=None):
    """
    Generates synthetic samples using Gaussian sampling based on the
    distribution of the target class (or the whole dataset).
    """
    if class_target is not None:
        source_df = df[df['diagnosis'] == class_target][features]
    else:
        source_df = df[features]

    means = source_df.mean()
    cov = source_df.cov()

    # Generate samples from multivariate normal distribution
    synthetic_data = np.random.multivariate_normal(means, cov, n_samples)

    # Clip values to ensure they stay within observed bounds (realistic constraint)
    for i, feature in enumerate(features):
        synthetic_data[:, i] = np.clip(
            synthetic_data[:, i],
            source_df[feature].min(),
            source_df[feature].max(),
        )

    return pd.DataFrame(synthetic_data, columns=features)


def evaluate_generalization(model, scaler, synthetic_df, features):
    """Evaluates how the model performs on a batch of synthetic samples."""
    scaled_samples = scaler.transform(synthetic_df[features])
    probs = model.predict_proba(scaled_samples)
    preds = model.predict(scaled_samples)

    results = synthetic_df.copy()
    results['Predicted_Class'] = ["Malignant" if p == 1 else "Benign" for p in preds]
    results['Confidence'] = [max(p) for p in probs]

    return results
