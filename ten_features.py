from scipy.stats import pearsonr
from operator import itemgetter
from scipy.stats import t
import statistics
import pandas as pd

fetal_data = pd.read_csv("fetal_health-1.csv")

def ten_features(input_data):
    # Present 10 features that are most reflective to fetal health conditions (there
    # are more than one way of selecting features and any of these are acceptable).
    # Present if the correlation is statistically significant (using 95% and 90%
    # critical values). (2)

    # Extract classification from csv
    # Create nested list to hold correlation values of each feature
    classification = input_data.fetal_health
    feature_correlation = [
        ["baseline value"],
        ["accelerations"],
        ["fetal_movement"],
        ["uterine_contractions"],
        ["light_decelerations"],
        ["severe_decelerations"],
        ["prolongued_decelerations"],
        ["abnormal_short_term_variability"],
        ["mean_value_of_short_term_variability"],
        ["percentage_of_time_with_abnormal_long_term_variability"],
        ["mean_value_of_long_term_variability"],
        ["histogram_width"],
        ["histogram_min"],
        ["histogram_max"],
        ["histogram_number_of_peaks"],
        ["histogram_number_of_zeroes"],
        ["histogram_mode"],
        ["histogram_mean"],
        ["histogram_median"],
        ["histogram_variance"],
        ["histogram_tendency"]
    ]

    # Loop through list
    # Compute and update list with abs(correlation), pvalue
    i = 0
    for item in feature_correlation:
        correlation = abs(pearsonr(input_data.iloc[:,i], classification)[0])
        pvalue = pearsonr(input_data.iloc[:,i], classification)[1]
        item.append( [correlation, pvalue] )
        i += 1

    # Sort the list by correlation values
    # Grab last 10 items in the list
    sorted_correlation = sorted(feature_correlation, key=itemgetter(1))[11:]
    
    print("10 features that are most reflective to fetal health conditions")
    for item in sorted_correlation:
        print(f"{item[0]}\n{item[1]}\n")