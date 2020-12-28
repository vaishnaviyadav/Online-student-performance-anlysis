# Importing all libraries
import pandas as pd
import numpy as np
#import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style



# Creating a dataset with pandas
# Data is separated with a comma
# Print all the data
data = pd.read_csv ("C:/Users/Vaishnavi/anaconda3/student-mat.csv")


data.head()

# Filtering dataset with only the attributes I want to keep
# Print filtered data
data = data[["G1", "G2", "G3", "failures", "absences", "studytime", "traveltime"]]
print(data.head())

# Creating a label (The final output)
predict = "G3"

# Return a new data frame with all our attributes that doesn't have G3 in it 
x = np.array(data.drop([predict], 1))

# All of our labels
y = np.array(data[predict])

# Taking all attributes and labels, splitting them into 4 different arrays
# x_train is a section of the array 'x'
# y_train is a section of the array 'y'
# The two test datas (x_test, y_test) is used to test the accuracy of the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) 
# Splitting up 10% of data into test samples so when we test, we can test off of new data that the machine hasn't seen before

# Making best fit line (linear regression)
linear = linear_model.LinearRegression()

# Fit this data to find a best fit line
linear.fit(x_train, y_train)

# Store accuracy of model
acc = linear.score(x_test, y_test)
print(acc)

# Save a pickle file of our model in our directory 
with open("model.pkl", "wb") as f:
    pickle.dump(linear, f)

# Load our model
model = open("model.pkl", "rb")
#linear = pickle.load(pickle_in)

# Print all coefficients of our attributes
print("Coefficients: " + "\n", linear.coef_)
# Print our y-intercept
print("Y-Intercept: " + "\n", linear.intercept_)

# Take an array and make predictions based on data not already given
predictions = linear.predict(x_test)

# Print each prediction next to the actual attributes and final output
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# G3 is our label and y coordinate
# G1 represents first semester grade, final grade is the final grade (obviously)
# You can replace 'p' with any attribute to see the correlation between that and the final grade
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


# Here's another example 
# Correlating number of failures to a students final grade
# As seen, students with less failures clearly have the higher final grades
p = "failures"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

