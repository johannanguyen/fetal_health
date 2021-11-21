import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr
from operator import itemgetter


def distribution():
    # Present a visual distribution of the 3 classes. Is the data balanced?
    # How do you plan to circumvent the data imbalance problem, if there is one?
    # (Hint: stratification needs to be included.) (1)
    print("Distribution placeholder") 


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


def linear_regression_model(x_train, x_test, y_train):
    # Develop two different models to classify CTG features into the three fetal health
    # states (I intentionally did not name which two models. Note that this is a multiclass
    # problem that can also be treated as regression, since the labels are numeric.) (2+2) 

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate Linear Regression Coefficients
    b1, b0 = np.polyfit(x_test.alcohol, y_pred, 1)
    print(f"B1: {b1}, B0: {b0}")

    # Graph scatter plot and linear regression line
    plt.plot(x_test.alcohol, b1 * x_test.alcohol + b0)
    plt.scatter(x_test.alcohol, y_pred, color="pink")
    plt.xlabel("Alcohol")
    plt.ylabel("Quality")
    plt.show()


def decision_tree_model(fdx_train, fdy_train):
    # Develop two different models to classify CTG features into the three fetal health
    # states (I intentionally did not name which two models. Note that this is a multiclass
    # problem that can also be treated as regression, since the labels are numeric.) (2+2) 

    model_tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(fdx_train, fdy_train)
    plt.figure(figsize=(8, 6))
    tree.plot_tree(model_tree, fontsize=6)
    plt.show()


def confusion_matrix():
    # Visually present the confusion matrix (1)
    print("Confusion matrix placeholder")


def scores():
    # With a testing set of size of 30% of all available data, calculate (1.5) 
    #     Area under the ROC Curve 
    #     F1 Score 
    #     Area under the Precision-Recall Curve 
    #     (For both models in 3) 
    print("Scores placeholder")


def k_means_clustering():
    # Without considering the class label attribute, use k-means clustering to cluster
    # the records in different clusters and visualize them (use k to be 5, 10, 15). (2.5) 
    print("K means clustering placeholder")

#From homework 3
def q2(input_data, x, y):
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(input_data):
        x_train, x_test, y_train, y_test = \
            x.iloc[train_index], x.iloc[test_index], \
            y.iloc[train_index], y.iloc[test_index]
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_train, y_train)

    print(f"Precision:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='precision_macro')}\n")
    print(f"Accuracy:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='accuracy')}\n")
    print(f"Recall:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='recall_macro')}\n")
    print(f"F1 Score:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='f1_macro')}\n")


def main():
    # Read and split up data set (70/30)
    input_data = pd.read_csv("winequality-red.csv")
    x = input_data.drop("quality", axis=1)
    y = input_data.quality
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    
    fetal_data = pd.read_csv("fetal_health-1.csv")
    fdx = fetal_data.drop("fetal_health", axis=1)
    fdy = fetal_data.fetal_health
    fdx_train, fdx_test, fdy_train, fdy_test = train_test_split(x,y,test_size=0.3,random_state=0)

    distribution()
    ten_features(fetal_data)
    linear_regression_model(x_train, x_test, y_train)
    decision_tree_model(fdx_train, fdy_train)
    confusion_matrix()
    scores()
    k_means_clustering()
    # From homework 3
    q2(input_data, x, y)


if __name__=="__main__":
    main()