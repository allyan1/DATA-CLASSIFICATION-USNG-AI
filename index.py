"""
PROJECT: DATA CLASSIFICATION USING AI
"""
# pip installing sciket pandas
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


iris = load_iris()


X = iris.data
y = iris.target


df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

print(" 5 rows of the dataset:")
print(df.head())
print("\nHow many flowers of each species?")
print(df['species'].value_counts())
print(f"\nTotal samples: {len(df)}, Features: {X.shape[1]}, Classes: {len(iris.target_names)}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# apply the logic to new data


print("\nRESULTS ")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"F1 Score (weighted): {f1_score(y_test, predictions, average='weighted'):.2f}")

print("\nMatrix confusion:")
print("(rows = actual species, and  columns = predicted species)")
print(confusion_matrix(y_test, predictions))

print("\nreport:")
print(classification_report(y_test, predictions, target_names=iris.target_names))