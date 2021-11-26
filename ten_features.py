from scipy.stats import pearsonr
from operator import itemgetter

def ten_features(input_data):
    # Present 10 features that are most reflective to fetal health conditions (there
    # are more than one way of selecting features and any of these are acceptable).
    # Present if the correlation is statistically significant (using 95% and 90%
    # critical values). (2)

    # Extract classification from csv
    # Create dictionary to hold correlation values of each feature
    classification = input_data.fetal_health
    feature_correlation = {
        "baseline value": 0,
        "accelerations": 0,
        "fetal_movement": 0,
        "uterine_contractions": 0,
        "light_decelerations": 0,
        "severe_decelerations": 0,
        "prolongued_decelerations": 0,
        "abnormal_short_term_variability": 0,
        "mean_value_of_short_term_variability": 0,
        "percentage_of_time_with_abnormal_long_term_variability": 0,
        "mean_value_of_long_term_variability": 0,
        "histogram_width": 0,
        "histogram_min": 0,
        "histogram_max": 0,
        "histogram_number_of_peaks": 0,
        "histogram_number_of_zeroes": 0,
        "histogram_mode": 0,
        "histogram_mean": 0,
        "histogram_median": 0,
        "histogram_variance": 0,
        "histogram_tendency": 0,
    }

    # Loop through dictionary
    # Compute and update dictionary with absolute value o proper correlation
    i = 0
    for item in feature_correlation:
        feature_correlation[item] = abs(pearsonr(input_data.iloc[:,i], classification)[0])
        i += 1

    # Sort the dictionary by values
    # Grab last 10 items in the dictionary
    sorted_correlation = sorted(feature_correlation.items(), key=itemgetter(1))[11:]
    print("10 features that are most reflective to fetal health conditions")
    for item in sorted_correlation:
        print(f"{item[0]}\n{item[1]}\n")