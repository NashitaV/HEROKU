import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models from Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

df_raw = pd.read_csv("C:/Users/nvnas/OneDrive/Desktop/heroku/main/ff_syn.csv")
df_raw

df_raw.info()
df_raw.describe()

df_raw["Shirtsize"].value_counts()
df_raw["Pantsize"].value_counts()
sns.countplot(x=df_raw["Shirtsize"])
sns.countplot(x=df_raw["Pantsize"])
sns.displot(df_raw["Chest"])
sns.displot(df_raw["Weight"])
sns.displot(df_raw["Height"])

dfs = []
sizes = []
for size_type in df_raw['Pantsize'].unique():
    sizes.append(size_type)
    ndf = df_raw[['Height','Weight']][df_raw['Pantsize'] == size_type]
    zscore = ((ndf - ndf.mean())/ndf.std())
    dfs.append(zscore)
    
for i in range(len(dfs)):
    # dfs[i]['age'] = dfs[i]['age'][(dfs[i]['age']>-3) & (dfs[i]['age']<3)]
    dfs[i]['Height'] = dfs[i]['Height'][(dfs[i]['Height']>-3) & (dfs[i]['Height']<3)]
    dfs[i]['Weight'] = dfs[i]['Weight'][(dfs[i]['Weight']>-3) & (dfs[i]['Weight']<3)]

for i in range(len(sizes)):
    dfs[i]['Pantsize'] = sizes[i]
df_raw = pd.concat(dfs)
df_raw.head()

df_raw.isna().sum()
# df_raw["age"] = df_raw["age"].fillna(df_raw['age'].median())
df_raw["Height"] = df_raw["Height"].fillna(df_raw['Height'].median())
df_raw["Weight"] = df_raw["Weight"].fillna(df_raw['Weight'].median())
df_raw.isna().sum()
df_raw
df_raw["bmi"] = df_raw["Height"]/df_raw["Weight"]
df_raw["Weight-squared"] = df_raw["Weight"] * df_raw["Weight"]

df_raw

corr = sns.heatmap(df_raw.corr(), annot=True)

# Features
X = df_raw.drop("Pantsize", axis=1)

# Target
y = df_raw["Pantsize"]

X.head()
y.head()
X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size=0.10)
len(X_train), len(X_test)

models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier(),
         "Decision Tree": DecisionTreeClassifier(),
         "Linear Regression": LinearRegression()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
   
    """
   Fits and evaluates given machine learning models.
   models: a dict of different Scikit_Learn machine learning models
   X_train: training data (no labels)
   X_test: testing data (no labels)
   y_train: training labels
   y_test: test labels
   """ 
    # Set random seed
    np.random.seed(18)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit model to data
        model.fit(X_train, y_train)
        # Evaluate model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)

    return model_scores

model_scores = fit_and_score(models,X_train,X_test,y_train,y_test)

model_scores

model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar()

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

import pickle
  
# Save the trained model as a pickle string.
saved_model = pickle.dumps(model)
  
# Load the pickled model
from_pickle = pickle.loads(saved_model)
  
# Use the loaded pickled model to make predictions
from_pickle.predict(X_test)

pickle.dump(saved_model, open('saved_model.pkl', 'wb'))
pickled_model = pickle.load(open('saved_model.pkl', 'rb'))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))