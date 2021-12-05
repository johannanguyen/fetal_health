import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, auc, precision_recall_curve

def scores(fdy_test, y_pred, pred_prob, model_name): 
    precision, recall, thresholds = precision_recall_curve(fdy_test, y_pred, pos_label=1)

    print("\n"+model_name+" SCORES")
    print("ROC AUC Score: \t\t", roc_auc_score(fdy_test, pred_prob, multi_class='ovo'))
    print("F1 Score: \t\t", f1_score(fdy_test,y_pred, average='macro'))
    print("Precision-Recall AUC: \t", auc(recall, precision))