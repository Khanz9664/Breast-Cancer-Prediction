import pandas as pd


def find_benign_counterfactual(model, scaler, current_input, features, training_df, max_iter=20):
    """
    Finds a 'benign' version of a malignant sample by incrementally moving
    the top features towards the population mean.
    """
    # Initialize targeted sample (scaled)
    current_scaled = scaler.transform(current_input[features])
    target_scaled = current_scaled.copy()

    # Calculate group means for all features
    training_means = training_df[features].mean()
    scaled_means = scaler.transform(pd.DataFrame([training_means], columns=features))

    # Check if already Benign
    if model.predict(target_scaled)[0] == 0:
        return None, "Sample is already classified as Benign."

    # Search loop
    # We move 5% closer to the mean each iteration for all features
    # (A more advanced version would use SHAP to target specific features)
    step_size = 0.05

    for i in range(max_iter):
        target_scaled = target_scaled * (1 - step_size) + scaled_means * step_size

        if model.predict(target_scaled)[0] == 0:
            # Converged to Benign territory
            target_unscaled = scaler.inverse_transform(target_scaled)
            target_df = pd.DataFrame(target_unscaled, columns=features)

            # Calculate changes
            changes = []
            for f in features:
                diff = target_df[f].iloc[0] - current_input[f].iloc[0]
                if abs(diff) > (current_input[f].iloc[0] * 0.01):  # Only track >1% change
                    changes.append({
                        'feature': f.replace('_', ' ').title(),
                        'from': current_input[f].iloc[0],
                        'to': target_df[f].iloc[0],
                        'change': diff
                    })

            return {
                'target_df': target_df,
                'changes': changes,
                'iterations': i + 1
            }, None

    return None, "Reached iteration limit without finding a benign boundary. The tumor characteristics may be extremely atypical."  # noqa: E501
