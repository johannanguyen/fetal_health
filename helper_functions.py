import matplotlib.pyplot as plt

def divide_classes(baseline_health, class_num):
    # Helper function for Question 1
    # Divides data into three different classes
    class_list = []
    for item in baseline_health:
        if item[1] == class_num:
            class_list.append(item[0])
    return class_list

def histogram(class_list, class_num):
    # Helper function for Question 1
    # Draws the histogram
    plt.hist(class_list, bins=10)
    plt.title("Class " + class_num)
    plt.show()

def print_tasks():
    print( "\na. Distributions\n",
        "b. Ten Features\n",
        "c1. Gaussian Naive Bayes Model\n",
        "c2. Decision Tree Model\n",
        "d. Confusion Matrix\n",
        "e. Scores\n",
        "f. K-Means Cluster\n\n")