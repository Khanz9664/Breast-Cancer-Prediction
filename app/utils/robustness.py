import numpy as np

def calculate_model_robustness(model, scaled_input):
    """
    Calculates uncertainty by analyzing the disagreement between 
    individual trees in the forest.
    """
    if not hasattr(model, 'estimators_'):
        # For non-forest models, we use a simple confidence score
        return {
            'uncertainty_score': 0.0,
            'confidence_interval': (0.8, 0.9), # Placeholder
            'stability_level': 'High (Deterministic)'
        }
    
    # Get predictions from every tree in the forest
    tree_preds = []
    for tree in model.estimators_:
        # Each tree gives a probability distribution [p_benign, p_malignant]
        tree_preds.append(tree.predict_proba(scaled_input)[0][1]) # Probability of Malignant
        
    tree_preds = np.array(tree_preds)
    
    # Uncertainty metrics
    variance = np.var(tree_preds)
    std_dev = np.std(tree_preds)
    mean_prob = np.mean(tree_preds)
    
    # Confidence Interval (95%)
    ci_lower = max(0, mean_prob - 1.96 * std_dev)
    ci_upper = min(1, mean_prob + 1.96 * std_dev)
    
    # Stability Score (Inverse of variance, normalized 0-100)
    stability_score = max(0, 100 - (std_dev * 200)) # Simple heuristic
    
    if stability_score > 85:
        level = "Highly Stable"
        color = "#51cf66"
    elif stability_score > 60:
        level = "Stable"
        color = "#fcc419"
    else:
        level = "Borderline / High Variance"
        color = "#ff6b6b"
        
    return {
        'all_tree_probs': tree_preds.tolist(),
        'variance': float(variance),
        'std_dev': float(std_dev),
        'ci': (float(ci_lower), float(ci_upper)),
        'stability_score': float(stability_score),
        'stability_level': level,
        'stability_color': color
    }
