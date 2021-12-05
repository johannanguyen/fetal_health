import pandas as pd
from sklearn.model_selection import train_test_split
from helper_functions import *

# Import each task from separate python files
from distribution import distribution
from ten_features import ten_features
from decision_tree import decision_tree_model
from gaussian_bayes import g_n_b_model
from kmeans import kmeans


def main():    
    fetal_data = pd.read_csv("fetal_health-1.csv")
    fdx = fetal_data.drop("fetal_health", axis=1)
    fdy = fetal_data.fetal_health
    fdx_train, fdx_test, fdy_train, fdy_test = train_test_split(fdx,fdy,test_size=0.3, stratify=fetal_data['fetal_health'], random_state=0)
        
    print_tasks()
    task = input("Select a task to view, q to quit: ")
    while(task != "q"):
        if task == "a":
             distribution(fetal_data)
        elif task == "b":
            ten_features(fetal_data)
        elif task == "c1":
            g_n_b_model(fdx_train, fdx_test, fdy_train, fdy_test, task)
        elif task == "c2":
            decision_tree_model(fdx_train, fdx_test, fdy_train, fdy_test, task)
        elif task == "d":
            decision_tree_model(fdx_train, fdx_test, fdy_train, fdy_test, task)
            g_n_b_model(fdx_train, fdx_test, fdy_train, fdy_test, task)
        elif task == "e":
            decision_tree_model(fdx_train, fdx_test, fdy_train, fdy_test, task)
            g_n_b_model(fdx_train, fdx_test, fdy_train, fdy_test, task)
        elif task == "f":
            kmeans(fetal_data, 5)
            kmeans(fetal_data, 10)
            kmeans(fetal_data, 15)
        else:
            print("Not a valid task")
        print_tasks()
        task = input("Select a task to view, q to quit: ")


if __name__=="__main__":
    main()