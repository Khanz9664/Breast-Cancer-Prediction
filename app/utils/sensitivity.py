import numpy as np
import pandas as pd


def calculate_sensitivity_curve(model, scaler, current_input, feature_to_vary, features, df, steps=50):
    """
    Varies a single feature across its observed range while keeping others constant,
    returning the probability curve.
    """
    # Get range from dataset
    f_min = df[feature_to_vary].min()
    f_max = df[feature_to_vary].max()
    feature_range = np.linspace(f_min, f_max, steps)

    # Create batch input
    batch_input = pd.concat([current_input] * steps, ignore_index=True)
    batch_input[feature_to_vary] = feature_range

    # Scale and predict
    scaled_batch = scaler.transform(batch_input[features])
    probs = model.predict_proba(scaled_batch)[:, 1]

    return {
        'x': feature_range.tolist(),
        'y': probs.tolist(),
        'current_val': current_input[feature_to_vary].iloc[0],
        'current_prob': model.predict_proba(scaler.transform(current_input[features]))[0][1]
    }
