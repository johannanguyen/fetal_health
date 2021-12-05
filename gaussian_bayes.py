from operator import mod, pos
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from conf_matrix import conf_matrix
from scores import scores

def g_n_b_model(fdx_train, fdx_test, fdy_train, fdy_test, menuOption):
    # Develop two different models to classify CTG features into the three fetal health
    # states (I intentionally did not name which two models. Note that this is a multiclass
    # problem that can also be treated as regression, since the labels are numeric.) (2+2)
    gnb = GaussianNB()
    y_pred = gnb.fit(fdx_train, fdy_train).predict(fdx_test)
    pred_prob = gnb.predict_proba(fdx_test)

    if menuOption == "c1":
        print("GNB Visualization goes here")
    elif menuOption == "d":
        #create confusion matrix
        conf_matrix(fdy_test, y_pred, "GNB")
    elif menuOption == "e":
        #create scores
        scores(fdy_test, y_pred, pred_prob, "GNB")