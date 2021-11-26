import matplotlib.pyplot as plt

def divide_classes(baseline_health, class_num):
    # Helper function for Question 1
    # Divides data into three different classes
    class_list = []
    for item in baseline_health:
        if item[1] == class_num:
            class_list.append(item[0])
    return class_list

def histogram(class_list):
    # Helper function for Question 1
    # Draws the histogram
    plt.hist(class_list, bins=10)
    plt.show()