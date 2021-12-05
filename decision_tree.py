from operator import mod, pos
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from conf_matrix import conf_matrix
from scores import scores

def decision_tree_model(fdx_train, fdx_test, fdy_train, fdy_test, menuOption):
    # Develop two different models to classify CTG features into the three fetal health
    # states (I intentionally did not name which two models. Note that this is a multiclass
    # problem that can also be treated as regression, since the labels are numeric.) (2+2) 
    model_tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(fdx_train, fdy_train)
    y_pred = model_tree.predict(fdx_test)
    pred_prob = model_tree.predict_proba(fdx_test)

    if menuOption == "c2":
        plt.figure(figsize=(8, 6))
        tree.plot_tree(model_tree, fontsize=6)
        plt.show()
    elif menuOption == "d":
        #create confusion matrix
        conf_matrix(fdy_test, y_pred, "DECISION TREE")
    elif menuOption == "e":
        #create scores
        scores(fdy_test, y_pred, pred_prob, "DECISION TREE")