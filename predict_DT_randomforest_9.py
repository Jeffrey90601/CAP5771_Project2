import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, metrics
import numpy as np

# Read the CSV files
df_train = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_test.csv')

# Separate features and target variable
x_train = df_train.drop(columns=['Class'])
y_train = df_train['Class']
x_test = df_test.drop(columns=['Class'])
y_test = df_test['Class']

# Splitting the training data into Training (80%) and Validation (20%)
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(
    x_train, y_train, test_size=0.2, random_state=9) 

# Print out statistics
print(f'Train instances: {x_train.shape[0]}, Features: {x_train.shape[1]}')
print(f'Validation instances: {x_validation.shape[0]}, Features: {x_validation.shape[1]}')
print(f'Test instances: {x_test.shape[0]}, Features: {x_test.shape[1]}')
print('Class distribution in Train set:', y_train.value_counts())
print('Class distribution in Validation set:', y_validation.value_counts())
print('Class distribution in Test set:', y_test.value_counts())

# Initialize hyperparameters
max_features_options = [0.1, 0.15, 0.2]
max_leaf_nodes_options = [20, 50, 100, 200, 400]
n_estimators_options = [2, 5, 10, 15, 20]
n_base_classifiers = [2, 5, 10, 15, 20]  # Added for individual classifier evaluation

results = []

# Build models and evaluate
for max_features in max_features_options:
    for max_leaf_nodes in max_leaf_nodes_options:
        for n_estimators in n_estimators_options:
            # Initialize RandomForestClassifier with max_features
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=9
            )

            # Fit the model on the training data
            rf.fit(x_train, y_train)

            # Evaluate on validation set
            predictions = rf.predict(x_validation)
            accuracy = metrics.accuracy_score(y_validation, predictions)
            f1_score = metrics.f1_score(y_validation, predictions, average='binary')
            results.append((max_features, max_leaf_nodes, n_estimators, accuracy, f1_score))


# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results, columns=['max_features', 'max_leaf_nodes', 'n_estimators', 'accuracy', 'f1_score'])

# Model Selection
# Find the best model based on validation accuracy
best_model = results_df.sort_values(by='accuracy', ascending=False).iloc[0]

# Use the best model found based on validation accuracy to predict on the test set
rf_best = RandomForestClassifier(
    n_estimators=int(best_model['n_estimators']),
    max_features=best_model['max_features'],
    max_leaf_nodes=int(best_model['max_leaf_nodes']),
    random_state=9
)
rf_best.fit(x_train, y_train)
test_predictions = rf_best.predict(x_test)

# Evaluate on the test set
test_accuracy = metrics.accuracy_score(y_test, test_predictions)
test_f1_score = metrics.f1_score(y_test, test_predictions, average='binary')
test_recall = metrics.recall_score(y_test, test_predictions, average='binary')
test_precision = metrics.precision_score(y_test, test_predictions, average='binary')

print("Test Accuracy:", test_accuracy)
print("Test Recall:", test_recall)
print("Test Precision:", test_precision)
print("Test F1-Score:", test_f1_score)

# Calculate ensemble accuracy (accuracy of the best model on the validation set)
ensemble_accuracy = best_model['accuracy']

# Calculate average individual accuracy
avg_individual_accuracy = results_df.groupby('max_features')['accuracy'].mean()

# Plot#3
bar_width = 0.35
x_values = np.arange(len(max_features_options))

plt.bar(x_values - bar_width/2, [ensemble_accuracy] * len(max_features_options), width=bar_width, label='Ensemble Accuracy')
plt.bar(x_values + bar_width/2, avg_individual_accuracy, width=bar_width, label='Avg Individual Accuracy')

plt.xticks(x_values, max_features_options)
plt.xlabel('Max Features')
plt.ylabel('Accuracy')
plt.title('Ensemble vs Avg Individual Accuracy')
plt.legend()
plt.show()
