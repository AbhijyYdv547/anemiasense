import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

print("Starting model training process...")


df = pd.read_csv('anemia.csv')

df.info()
df.isnull().sum()

majorclass = df[df['Result'] == 0]
minorclass = df[df['Result'] == 1]
print(f"Original counts: Result 0 = {len(majorclass)}, Result 1 = {len(minorclass)}")

major_downsample = resample(majorclass, replace=False, n_samples=len(minorclass), random_state=42)
df = pd.concat([major_downsample, minorclass])
df['Result'].value_counts()


if 'Gender' in df.columns and df['Gender'].dtype == 'object':
    print("Encoding 'Gender' column...")
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    print("'Gender' column encoded to numerical.")
elif 'Gender' in df.columns and df['Gender'].dtype in ['int64', 'float64']:
    print("'Gender' column is already numerical. Skipping encoding.")
else:
    print("Warning: 'Gender' column not found or not in expected string format for encoding.")


X = df.drop('Result', axis=1)
Y = df['Result']


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000) 
logistic_regression.fit(x_train,y_train)
y_pred_lr = logistic_regression.predict(x_test)
acc_lr = accuracy_score(y_test,y_pred_lr)
print('\nLogistic Regression Accuracy Score: ',acc_lr)
print(classification_report(y_test,y_pred_lr))

# Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(x_train,y_train)
y_pred_rf = random_forest.predict(x_test)
acc_rf = accuracy_score(y_test,y_pred_rf)
print('\nRandom Forest Accuracy Score: ',acc_rf)
print(classification_report(y_test,y_pred_rf))

# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(x_train,y_train)
y_pred_dt = decision_tree_model.predict(x_test)
acc_dt = accuracy_score(y_test,y_pred_dt)
print('\nDecision Tree Accuracy Score: ',acc_dt)
print(classification_report(y_test,y_pred_dt))

# Gaussian Naive Bayes
NB = GaussianNB()
NB.fit(x_train,y_train)
y_pred_nb = NB.predict(x_test)
acc_nb = accuracy_score(y_test,y_pred_nb)
print('\nGaussian Naive Bayes Accuracy Score: ',acc_nb)
print(classification_report(y_test,y_pred_nb))

# Support Vector Classifier
support_vector = SVC(random_state=42)
support_vector.fit(x_train,y_train)
y_pred_svc = support_vector.predict(x_test)
acc_svc = accuracy_score(y_test,y_pred_svc)
print('\nSupport Vector Classifier Accuracy Score: ',acc_svc)
print(classification_report(y_test,y_pred_svc))

# Gradient Boosting Classifier (This is the one you chose to save)
print("\nTraining GradientBoostingClassifier...")
GBC = GradientBoostingClassifier(random_state=42)
GBC.fit(x_train,y_train)
y_pred_gbc = GBC.predict(x_test)
acc_gbc = accuracy_score(y_test,y_pred_gbc)
print('GradientBoostingClassifier Accuracy Score: ',acc_gbc)
print(classification_report(y_test,y_pred_gbc))


model_path = "model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(GBC, file)

print(f"\nModel (GradientBoostingClassifier) saved successfully to {model_path}")

# Optional: Display model comparison
model_scores = pd.DataFrame({
    'Model':['Logistic Regression','Decision Tree Classifier','RandomForest Classifier','Gaussian Naive Bayes','Support Vector Classifier','Gradient Boost Classifier'],
    'Score':[acc_lr,acc_dt,acc_rf,acc_nb,acc_svc,acc_gbc]
})
print("\nModel Comparison:")
print(model_scores)