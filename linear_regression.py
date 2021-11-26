import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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