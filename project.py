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
    classification = input_data.fetal_health
    feature_correl = {
        "baseline value": abs(pearsonr(input_data.iloc[:,0], classification)[0]),
        "accelerations": abs(pearsonr(input_data.iloc[:,1], classification)[0]),
        "fetal_movement": abs(pearsonr(input_data.iloc[:,2], classification)[0]),
        "uterine_contractions": abs(pearsonr(input_data.iloc[:,3], classification)[0]),
        "light_decelerations": abs(pearsonr(input_data.iloc[:,4], classification)[0]),
        "severe_decelerations": abs(pearsonr(input_data.iloc[:,5], classification)[0]),
        "prolongued_decelerations": abs(pearsonr(input_data.iloc[:,6], classification)[0]),
        "abnormal_short_term_variability": abs(pearsonr(input_data.iloc[:,7], classification)[0]),
        "mean_value_of_short_term_variability": abs(pearsonr(input_data.iloc[:,8], classification)[0]),
        "percentage_of_time_with_abnormal_long_term_variability": abs(pearsonr(input_data.iloc[:,9], classification)[0]),
        "mean_value_of_long_term_variability": abs(pearsonr(input_data.iloc[:,10], classification)[0]),
        "histogram_width": abs(pearsonr(input_data.iloc[:,11], classification)[0]),
        "histogram_min": abs(pearsonr(input_data.iloc[:,12], classification)[0]),
        "histogram_max": abs(pearsonr(input_data.iloc[:,13], classification)[0]),
        "histogram_number_of_peaks": abs(pearsonr(input_data.iloc[:,14], classification)[0]),
        "histogram_number_of_zeroes": abs(pearsonr(input_data.iloc[:,15], classification)[0]),
        "histogram_mode": abs(pearsonr(input_data.iloc[:,16], classification)[0]),
        "histogram_mean": abs(pearsonr(input_data.iloc[:,17], classification)[0]),
        "histogram_median": abs(pearsonr(input_data.iloc[:,18], classification)[0]),
        "histogram_variance": abs(pearsonr(input_data.iloc[:,19], classification)[0]),
        "histogram_tendency": abs(pearsonr(input_data.iloc[:,20], classification)[0])
    }
    feature_correl = sorted(feature_correl.items(), key=itemgetter(1))[11:]

    print("10 features that are most reflective to fetal health conditions")
    for item in feature_correl:
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


def decision_tree_model(x_train, y_train):
    # Develop two different models to classify CTG features into the three fetal health
    # states (I intentionally did not name which two models. Note that this is a multiclass
    # problem that can also be treated as regression, since the labels are numeric.) (2+2) 

    model_tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_train, y_train)
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


def main():
    # Read and split up data set (70/30)
    input_data = pd.read_csv("fetal_health-1.csv")
    # x = input_data.drop("quality", axis=1)
    # y = input_data.quality
    # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

    #distribution()
    ten_features(input_data)
    # linear_regression_model(x_train, x_test, y_train)
    # decision_tree_model(x_train, y_train)
    # confusion_matrix()
    # scores()
    # k_means_clustering()


if __name__=="__main__":
    main()