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
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

#imported ALL libraries in this cell even the ones used in the last cells
# to avoid import errors in the pytest tests


# %%
df = pd.read_csv('E:/OneDrive/Documents/coding/MLInternship/week2/winequalityN.csv')

# %% [markdown]
# deciding to add the median value in the null columns

# %%
df.head()





# %%
df.isnull().sum()

# %%
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# %%
print(df.isnull().sum())


# %%
print(df['type'].unique())  #checking for values in type column


# %%
df['type'] = df['type'].str.lower().str.strip().map({'white': 0, 'red': 1})


# %%
df.head()

# %%
correlations = df.corr()['type'].sort_values(ascending=False)
print ('Correlations agianst type for feature selection')
print (correlations)

plt.figure(figsize=(10,6))
sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
plt.title('Feature Correlations with Wine Type')
plt.xlabel('Correlation with type (red/white)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %%
df.head()


# %%
#running model first without dropping any features

x = df.drop('type', axis =1)
y = df['type']

X_train,X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# %%
pipelines = {
    'DecisionTree' : Pipeline([
        ('scalar', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth= 4, random_state=42))

    ]),
    'KNN': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ]),
    'Naive-Bayes': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', GaussianNB())

        ])
}



# %%
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"{name} training accuracy: {train_score:.4f}")
    print(f"{name} testing accuracy: {test_score:.4f}")
    print("-" * 30)

# %%
#running model after dropping alcohol (-0.032) upon correaltion

x = df.drop(['type','alcohol'], axis =1)
y = df['type']

X_train,X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# %%
pipelines = {
    'DecisionTree' : Pipeline([
        ('scalar', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth= 4, random_state=42))

    ]),
    'KNN': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ]),
    'Naive-Bayes': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', GaussianNB())

        ])
}


# %%
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"{name} training accuracy: {train_score:.4f}")
    print(f"{name} testing accuracy: {test_score:.4f}")
    print("-" * 30)

# %% [markdown]
# minor increase in KNN testing without alcohol
# 

# %%
#running lasso and random forest classifier model for feature selection

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%

model = RandomForestClassifier(random_state =42)
model.fit(X_train_scaled,y_train)
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

x = df.drop(['type'], axis =1)
y= df['type']
lasso = LassoCV().fit(x,y)



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

# %% [markdown]
# 

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
x_selected = x[selected_features]
y = df['type']


# %%
X_train, X_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=42)

# %%
pipelines = {
    'DecisionTree' : Pipeline([
        ('scalar', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth= 4, random_state=42))

    ]),
    'KNN': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ]),
    'Naive-Bayes': Pipeline([
        ('scalar', StandardScaler()),
        ('clf', GaussianNB())

        ])
}


# %%
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"{name} training accuracy: {train_score:.4f}")
    print(f"{name} testing accuracy: {test_score:.4f}")
    print("-" * 30)

# %%
model_reports = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n=== {name} Model ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['White', 'Red'], digits=4))
    model_reports[name] = classification_report(y_test, y_pred, target_names=['White', 'Red'], output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['White', 'Red'])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# Individual metric bar graphs for all models
classes = ['White', 'Red']
metrics = ['precision', 'recall', 'f1-score']
colors = ['#4C72B0', '#C44E52', '#55A868']

for metric in metrics:
    plt.figure(figsize=(8, 5))
    bar_width = 0.2
    x = np.arange(len(classes))
    for idx, (model_name, report) in enumerate(model_reports.items()):
        values = [report[cls][metric] for cls in classes]
        plt.bar(x + bar_width * idx, values, width=bar_width, label=model_name, color=colors[idx])
    plt.xticks(x + bar_width, classes)
    plt.ylim(0, 1)
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} by Model and Wine Type')
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for idx, (model_name, report) in enumerate(model_reports.items()):
        values = [report[cls][metric] for cls in classes]
        for i, v in enumerate(values):
            plt.text(i + bar_width * idx, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

# Bar graph for support (number of samples per class) for each model
plt.figure(figsize=(8, 5))
bar_width = 0.2
x = np.arange(len(classes))
for idx, (model_name, report) in enumerate(model_reports.items()):
    values = [report[cls]['support'] for cls in classes]
    plt.bar(x + bar_width * idx, values, width=bar_width, label=model_name, color=colors[idx])
plt.xticks(x + bar_width, classes)
plt.ylabel('Support (Number of Samples)')
plt.title('Support by Model and Wine Type')
plt.legend(title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.5)
for idx, (model_name, report) in enumerate(model_reports.items()):
    values = [report[cls]['support'] for cls in classes]
    for i, v in enumerate(values):
        plt.text(i + bar_width * idx, v + 2, f"{int(v)}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()


# %%


# %%
# Save the trained KNN model and scaler for wine detection
import joblib

# Fit the KNN pipeline on the selected features and training data if not already done
knn_pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('clf', KNeighborsClassifier())
])
knn_pipeline.fit(X_train, y_train)
joblib.dump(knn_pipeline, 'winemodel.joblib')
print("Saved KNN wine model to winemodel.joblib")

# Save the scaler separately if needed
winescaler = StandardScaler().fit(X_train)
joblib.dump(winescaler, 'winescaler.joblib')
print("Saved wine scaler to winescaler.joblib")


