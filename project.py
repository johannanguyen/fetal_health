import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score


# Import each task from separate python files
from distribution import distribution
from ten_features import ten_features
from linear_regression import linear_regression_model
from decision_tree import decision_tree_model


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
# def q2(input_data, x, y):
#     kf = KFold(n_splits=10, shuffle=True)
#     for train_index, test_index in kf.split(input_data):
#         x_train, x_test, y_train, y_test = \
#             x.iloc[train_index], x.iloc[test_index], \
#             y.iloc[train_index], y.iloc[test_index]
#     model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_train, y_train)

#     print(f"Precision:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='precision_macro')}\n")
#     print(f"Accuracy:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='accuracy')}\n")
#     print(f"Recall:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='recall_macro')}\n")
#     print(f"F1 Score:\n{cross_validate(estimator=model, X=x,y=y,cv=kf,scoring='f1_macro')}\n")


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

    print( "a. Distributions\n",
        "b. Ten Features\n",
        "c1. Linear Regression Model\n",
        "c2. Decision Tree Model\n",
        "d. Confusion Matrix\n",
        "e. Scores\n",
        "f. K-Means Cluster\n\n")
        
    task = input("Select a task to view, q to quit: ")
    while(task != "q"):
        if task == "a":
             distribution(fetal_data)
        elif task == "b":
            ten_features(fetal_data)
        elif task == "c1":
            linear_regression_model(x_train, x_test, y_train)
        elif task == "c2":
            decision_tree_model(fdx_train, fdy_train)
        elif task == "d":
            confusion_matrix()
        elif task == "e":
            scores()
        elif task == "f":
            k_means_clustering()
        else:
            print("Not a valid task")
        task = input("Select a task to view, q to quit: ")


if __name__=="__main__":
    main()