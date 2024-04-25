import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the CSV files
df_train = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_test.csv')

# Prepare the data
X_train = df_train.drop(columns=['Class'])
Y_train = df_train['Class']
X_test = df_test.drop(columns=['Class'])
Y_test = df_test['Class']

# Splitting the data into training and validation
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=9)

# Define hyperparameters
n_base_classifiers = [2, 5, 10, 15, 20]
max_features = [0.1, 0.15, 0.2]
max_leaf_nodes = [20, 50, 100, 200, 400]
random_state = 9  

# Store results
results = []

# Train Random Forests
for features in max_features:
    for leaf_nodes in max_leaf_nodes:
        clf = RandomForestClassifier(
            n_estimators=5,  # Using 5 as we need to compare when n_base_classifiers = 5
            max_features=features,
            max_leaf_nodes=leaf_nodes,
            random_state=random_state
        )
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_validation)
        accuracy = accuracy_score(Y_validation, predictions)
        results.append({
            'max_features': features,
            'max_leaf_nodes': leaf_nodes,
            'accuracy': accuracy
        })

# Convert results to DataFrame 
results_df = pd.DataFrame(results)
print(results_df)
