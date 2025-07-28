# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import sklearn.pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


# %%
df = pd.read_csv('E:/OneDrive/Documents/coding/MLInternship/week2/breast-cancer.csv')

# %%
print(df.head())

# %%
print (df.isnull().sum())

# %%
df.duplicated()

# %%
df = df.drop('id', axis=1)

# %%
print (df.info())
#df = df.drop(['radius_worst', 'perimeter_worst', 'fractal_dimension_worst'], axis=1)


# %%
print (df.describe())


# %%
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'].astype(str))

print (df.head(20))
# 1 = M
# 0 = B

# %%
correlations = df.corr()['diagnosis'].sort_values(ascending=False)
print ("correlations with each feature: ")
print (correlations)

# %%
#df = df.drop(['fractal_dimension_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean', 'smoothness_se'],axis =1)
print (df.head())

# %%
x = df.drop(['diagnosis'], axis =1)
y = df['diagnosis']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# %%

model = RandomForestClassifier(random_state =42)
model.fit(x_scaled,y)
importances = model.feature_importances_
feature_names = x.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display the most important features
rf_top_features = feature_importance_df['Feature'].head(10).tolist()
print("Top features selected by Random Forest:", rf_top_features)




# %%

x = df.drop(['diagnosis'], axis =1)
y = df['diagnosis']
lasso = LassoCV().fit(x_scaled, y)


lasso_coefs = lasso.coef_
lasso_features_df = pd.DataFrame({
    'Feature': x.columns,
    'Lasso_Coefficient': lasso_coefs
}).sort_values(by='Lasso_Coefficient', key=abs, ascending=False)

print("Lasso Feature Coefficients:")
print(lasso_features_df)

# Select features with non-zero coefficients
lasso_top_features = lasso_features_df[lasso_features_df['Lasso_Coefficient'] != 0]['Feature'].tolist()
print("Top features selected by Lasso:", lasso_top_features)


# %%
#Select the top 10 features by absolute coefficient value from Lasso
lasso_top10_features = lasso_features_df.head(10)['Feature'].tolist()
print("Top 10 features selected by Lasso (by absolute coefficient value):", lasso_top10_features)


# %%
selected_features = list(set(rf_top_features) & set(lasso_top_features))
print("Final selected features (intersection):", selected_features)

# Filter X with selected features
x_selected = x[selected_features]

# %%
# Remove outliers from selected features using IQR method
filtered_df = df.copy()
for col in selected_features:
    Q1 = filtered_df[col].quantile(0.25)
    Q3 = filtered_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]

# Update x_selected and y after outlier removal
x_selected = filtered_df[selected_features]
y = filtered_df['diagnosis']
print('Shape after outlier removal:', x_selected.shape)




# %%
# Correlation heatmap for selected features
plt.figure(figsize=(10, 8))
selected_corr = filtered_df[selected_features + ['diagnosis']].corr()
sns.heatmap(selected_corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap of Selected Features and Diagnosis')
plt.show()

# %%
'''
x = df.drop(['diagnosis'], axis =1)
y = df['diagnosis']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print (x)
'''

# %%
pipelines = {
    'DecisionTree' : Pipeline([
        ('scalar', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth= 4, random_state=42))

    ]),
    'KNN': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=10))
    ]),
    'Naive-Bayes': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', GaussianNB())

        ])
}



# %%
X_train, X_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=42)

# %%

k_range = range(1, 21)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    pipeline = Pipeline([
        ('scalar', StandardScaler()),
        ('clf', knn)
    ])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    scores.append(score)
    print(f"K={k}: Test Accuracy = {score:.4f}")

best_k = k_range[scores.index(max(scores))]
print(f"Best K: {best_k} with accuracy {max(scores):.4f}")

# %%

param_grid = {'clf__max_depth': range(1, 21)}

dt_pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('clf', DecisionTreeClassifier(random_state=42))
])

grid_search = GridSearchCV(dt_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best max_depth: {grid_search.best_params_['clf__max_depth']}")
print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_dt = grid_search.best_estimator_
test_acc = best_dt.score(X_test, y_test)
print(f"Test set accuracy with best max_depth: {test_acc:.4f}")

# %%
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"{name} training accuracy: {train_score:.4f}")
    print(f"{name} testing accuracy: {test_score:.4f}")
    print("-" * 30)

# %%
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} prediction accuracy on test set: {acc:.4f}")

# %%

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"], output_dict=True)
    metrics = ['precision', 'recall', 'f1-score', 'support']
    classes = ["Benign (0)", "Malignant (1)"]
    for metric in metrics:
        values = [report[cls][metric] for cls in classes]
        plt.figure(figsize=(5,3))
        plt.bar(classes, values, color=['skyblue', 'salmon'])
        plt.title(f'{name} - {metric.capitalize()} by Class')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, max(values)*1.2 if metric != 'support' else max(values)*1.1)
        for i, v in enumerate(values):
            plt.text(i, v + (0.02 if metric != 'support' else 2), f'{v:.2f}' if metric != 'support' else f'{int(v)}', ha='center')
        plt.show()


# %%

for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)
    print(f"Confusion Matrix for {name}:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign (0)", "Malignant (1)"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]))
    print("-" * 50)



