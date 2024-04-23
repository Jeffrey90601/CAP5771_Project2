import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn import metrics

# Read the CSV files
df_train = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_test.csv')

# Create target variable on the training set for x and set a variable with only the target variable
x_train = df_train.drop(columns=['Class'])
y_train = df_train['Class']

x_test = df_test.drop(columns=['Class'])
y_test = df_test['Class']

# Splitting the training data into two parts: Training (80%) and Validation (20%)
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=9)

# STATISTICS
# Print out number of instances and features in the provided files
print('dataset_train.csv:')
print("Number of instances: " + str(len(df_train)))
print("Number of features: " + str(len(df_train.columns)))
print('\n')
print('dataset_test.csv')
print("Number of instances: " + str(len(df_test)))
print("Number of features: " + str(len(df_test.columns)))
print('\n')

# Print out number of instances from each class in the training/validation/test sets:
print('Training Set - Number of instances: ' + str(len(x_train)))
print('Validation Set - Number of instances: ' + str(len(x_validation)))
print('Test Set - Number of instances: ' + str(len(df_test)))

# DecisionTreeClassifier with Validation Set
nleafnodes = [2, 4, 10, 20, 50, 100, 200, 400, 600, None]
validation_accuracy = []
validation_f1_score = []

for nleaves in nleafnodes:
    clf = DecisionTreeClassifier(max_leaf_nodes=nleaves)
    clf = clf.fit(x_train, y_train)
    predictions = clf.predict(x_validation)
    accuracy = metrics.accuracy_score(y_validation, predictions)
    f1_score = metrics.f1_score(y_validation, predictions)

    validation_accuracy.append(accuracy)
    validation_f1_score.append(f1_score)

print(validation_accuracy)
print(validation_f1_score)

# DecisionTreeClassifier with Test Set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=9)
test_accuracy = []
test_recall = []
test_f1_score = []

for nleaves in nleafnodes:
    clf = DecisionTreeClassifier(max_leaf_nodes=nleaves)
    clf = clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions)

    test_accuracy.append(accuracy)
    test_recall.append(recall)
    test_f1_score.append(f1_score)

print(test_accuracy)
print(test_recall)
print(test_f1_score)

# Plot #1
plt.plot(nleafnodes, validation_accuracy, label='Training')
plt.plot(nleafnodes, test_accuracy, label='Test')
plt.title('nleafnodes vs. Accuracy')
plt.xlabel('nleafnodes')
plt.ylabel('Accuracy')
plt.legend()
plt.show()