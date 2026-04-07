
def get_clinical_insights(user_input):
    """
    Provides rule-based clinical insights based on feature thresholds.
    """
    insights = []

    # Radius / Size rules
    if user_input['radius_mean'].iloc[0] > 18:
        insights.append({
            'feature': 'Radius Mean',
            'insight': (
                'Tumor radius exceeds the typical benign threshold (18mm),'
                ' suggesting significant horizontal growth.'
            ),
            'severity': 'high'
        })
    elif user_input['radius_mean'].iloc[0] > 15:
        insights.append({
            'feature': 'Radius Mean',
            'insight': 'Tumor radius is in the upper quartile of the dataset.',
            'severity': 'medium'
        })

    # Concavity / Shape rules
    if user_input['concave_points_mean'].iloc[0] > 0.1:
        insights.append({
            'feature': 'Concave Points',
            'insight': 'Extremely high concave points count, a strong indicator of malignant cell complexity.',  # noqa: E501
            'severity': 'high'
        })
    elif user_input['concave_points_mean'].iloc[0] > 0.05:
        insights.append({
            'feature': 'Concave Points',
            'insight': 'Concave points are elevated, suggesting irregular boundary patterns.',
            'severity': 'medium'
        })

    # Perimeter / Area rules
    if user_input['area_worst'].iloc[0] > 1000:
        insights.append({
            'feature': 'Area Worst',
            'insight': (
                'The largest recorded area of the tumor is significantly high (>1000),'
                ' which correlates with advanced stage development.'
            ),
            'severity': 'high'
        })

    # Compactness / Texture rules
    if user_input['compactness_mean'].iloc[0] > 0.2:
        insights.append({
            'feature': 'Compactness Mean',
            'insight': 'High cell compactness indicates potential density issues relevant to malignancy.',
            'severity': 'medium'
        })

    # Default insight if nothing else matches
    if not insights:
        insights.append({
            'feature': 'General Morphology',
            'insight': 'All recorded tumor metrics are within standard statistical ranges.',
            'severity': 'low'
        })

    return insights
