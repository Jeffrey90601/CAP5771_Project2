import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier

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

# BaggingClassifier with Validation Set
max_leaf_nodes = [20, 50, 100, 200, 400]
n_base_classifiers = [2, 5, 10, 15, 20]
random_state = 9

validation_accuracy = []
validation_precision = []
validation_f1_score = []

for nleafnode in max_leaf_nodes:
    for n_classifier in n_base_classifiers:
        dt = DecisionTreeClassifier(max_leaf_nodes=nleafnode, random_state=random_state)
        clf = BaggingClassifier(estimator=dt, n_estimators=n_classifier, random_state=random_state)
        clf = clf.fit(x_train, y_train)
        predictions = clf.predict(x_validation)
        accuracy = metrics.accuracy_score(y_validation, predictions)
        precision = metrics.precision_score(y_validation, predictions, average='binary')
        f1_score = metrics.f1_score(y_validation, predictions)

        validation_accuracy.append(accuracy)
        validation_precision.append(precision)
        validation_f1_score.append(f1_score)

print("Validation Accuracy: ", str(max(validation_accuracy)))
print("Validation Precision: ", str(max(validation_precision)))
print("Validation F1 Score: ", str(max(validation_f1_score)))

# BaggingClassifier with Test Set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=9)
test_accuracy = []
test_recall = []
test_f1_score = []
test_precision = []

max_20_accuracy = []
max_400_accuracy = []

for nleafnode in max_leaf_nodes:
    for n_classifier in n_base_classifiers:
        dt = DecisionTreeClassifier(max_leaf_nodes=nleafnode, random_state=random_state)
        clf = BaggingClassifier(estimator=dt, n_estimators=n_classifier, random_state=random_state)
        clf = clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        recall = metrics.recall_score(y_test, predictions)
        f1_score = metrics.f1_score(y_test, predictions)
        test_precision = metrics.precision_score(y_test, predictions, average='binary')

        test_accuracy.append(accuracy)
        test_recall.append(recall)
        test_f1_score.append(f1_score)

        if nleafnode == 20:
            max_20_accuracy.append(accuracy)
        elif nleafnode == 400:
            max_400_accuracy.append(accuracy)

print('\n')
print("Test Accuracy:", str(max(test_accuracy)))
print("Test Recall:", str(max(test_recall)))
print("Test Precision:", test_precision)
print("Test F1-Score:", str(max(test_f1_score)))

# Plot #2
plt.plot(n_base_classifiers, max_20_accuracy, label='Max_20_Accuracy')
plt.plot(n_base_classifiers, max_400_accuracy, label='Max_400_Accuracy')
plt.title('n_base_classifiers vs. Accuracy')
plt.xlabel('n_base_classifiers')
plt.ylabel('Accuracy')
plt.legend()
plt.show()