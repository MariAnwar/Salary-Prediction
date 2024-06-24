#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib


# In[2]:


df = pd.read_csv("D:\Courses\Mentorness ML Internship\Project 1\Salary Prediction of Data Professions.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# # Data Preprocessing

# ### Checking for missing values and the Dublicates

# In[5]:


df.isnull().sum()


# In[6]:


df = df.dropna()
df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df = df.drop_duplicates()
df.duplicated().sum()


# In[9]:


df.shape


# # EDA

# In[10]:


numeric_data = df.select_dtypes(include=['number']).columns
df[numeric_data]


# In[12]:


# Statistics about data
df[numeric_data].describe().transpose() # correlation between numeric attributes
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_data].corr(),annot=True,cmap = 'Blues')


# In[14]:


# correlation between numeric attributes
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_data].corr(),annot=True,cmap = 'Blues');


# In[17]:


df[numeric_data].hist(figsize=(12,9),layout=(3,3),color='mediumpurple',edgecolor='indigo',grid=False)
plt.suptitle('Numeric Data Distributions (Frequency)');


# In[18]:


# Checking the relationship between the target and some of the numeric features
numerical_features = ['RATINGS','PAST EXP', 'AGE']
for feature in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.set_palette("plasma")
    sns.barplot(x=df[feature], y=df['SALARY'])
    plt.title(f'Relationship between Average Salary and {feature}')
    plt.xlabel(feature)
    plt.ylabel('Average Salary')
    plt.xticks(rotation=45)
    plt.show()


# In[42]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='PAST EXP', y='SALARY', data=df)
sns.regplot(x='PAST EXP', y='SALARY', data=df, scatter=False, color='red')
plt.title('Past Experience vs. Salary')
plt.xlabel('Past Experience (Years)')
plt.ylabel('Salary')
plt.show()


# #### I found out that the correlation between (Past experience, Ratings, and Age) features and Salary (Target) is strong. So that they will be considered as important features while training the model.

# In[11]:


categorical_data = df.select_dtypes(include=['object']).columns
df[categorical_data]


# In[19]:


# Unique values
df[categorical_data].describe().transpose()


# In[156]:


fig = plt.figure(figsize=(10,12),layout='constrained')
fig.suptitle(' Categorical Data Distribution (Frequency)')

gs = fig.add_gridspec(3,3)

sns.set_palette("rocket")

ax1 = fig.add_subplot(gs[0, 0])
sns.countplot(ax=ax1, data=df, x=categorical_data[2])
ax1.tick_params(axis='x', rotation=0)

ax2 = fig.add_subplot(gs[0, 1])
sns.countplot(ax=ax2, data=df, x=categorical_data[5])
ax2.tick_params(axis='x', rotation=90)

ax3 = fig.add_subplot(gs[0, 2])
sns.countplot(ax=ax3, data=df, x=categorical_data[6])
ax3.tick_params(axis='x', rotation=90)


plt.show()


# In[157]:


fig = plt.figure(figsize=(10,12),layout='constrained')
fig.suptitle('Relationship between the Categorical features and the Salary')

gs = fig.add_gridspec(3,3)

sns.set_palette("pastel")

ax1 = fig.add_subplot(gs[0, 0])
sns.barplot(data=df, x=df['SEX'], y=df['SALARY'])
ax1.tick_params(axis='x', rotation=0)

ax2 = fig.add_subplot(gs[0, 1])
sns.barplot(data=df, x=df['DESIGNATION'], y=df['SALARY'])
ax2.tick_params(axis='x', rotation=90)

ax3 = fig.add_subplot(gs[0, 2])
sns.barplot(data=df, x=df['UNIT'], y=df['SALARY'])
ax3.tick_params(axis='x', rotation=90)

plt.show()


# In[41]:


################ Target Distribution #########
#plt.figure(figsize=(11,8))
sns.set(rc={'figure.figsize':(11,8),'figure.dpi':90})
sns.histplot(df['SALARY'], kde=True, color="#0072B2")
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()


# In[30]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df['SALARY'])
plt.title('Box Plot of Salary')
plt.xlabel('Salary')
plt.show()


# #### After checking the Salary distribution, I found out that the distribution is skewed which means that most of the salaries are in the range of 50000. 
# #### Also it has outliers that their salary are higher than 200000. So, I checked these outliers.

# In[45]:


### Checking the outliers
high_salary_employees = df[df['SALARY'] > 200000]

high = pd.DataFrame(high_salary_employees)

# Display the rows of employees with salary greater than 200,000
high = high.drop(columns=['FIRST NAME', 'LAST NAME'])
high


# #### After Checking the outliers by filtering the data to display only the professions whose salaries are higher than 200000, I found out all of them are Directors, most of them aged above 40, and their experience is more than ten years. 
# #### So, based on the previous distributions of the features and their correlation between the salary which means they are not considered outliers.

# # Feature Engineering

# Since the "Past Experience" feature has a strong relation with salary. So, the total time in the position from the date of joining to the Current date is considered experience. Thus, I calculated the total experience by adding the duration (in years) of joining the position to the experience. And it has a strong relation with salary.

# In[46]:


# Convert date columns to datetime
df['DOJ'] = pd.to_datetime(df['DOJ'])
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])


# In[48]:


# Calculate the duration in days
df['DURATION'] = df['CURRENT DATE'] - df['DOJ']
df['DURATION_YEARS'] = df['DURATION'].dt.days / 365.25


# In[51]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='DURATION_YEARS', y='SALARY', data=df)
sns.regplot(x='DURATION_YEARS', y='SALARY', data=df, scatter=False, color='red')
plt.title('Duration vs. Salary')
plt.xlabel('Duration (Years)')
plt.ylabel('Salary')
plt.show()   #There is a strong relation between the Duration and the Salary


# In[53]:


df["EXP"] = df["PAST EXP"] + df["DURATION_YEARS"]  # The total Experience of each data profession
df.head()


# In[54]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='EXP', y='SALARY', data=df)
sns.regplot(x='EXP', y='SALARY', data=df, scatter=False, color='red')
plt.title('Total Experience vs. Salary')
plt.xlabel('Total Experience (Years)')
plt.ylabel('Salary')
plt.show()


# In[56]:


# Choosing the best features based on their relation between them and the Salary
data = df[['EXP', 'PAST EXP', 'SEX', 'UNIT', 'DESIGNATION', 'AGE', 'RATINGS', 'SALARY']]
data.head()


# ### Splitting the Data and Encoding the data for model training

# In[135]:


# Define features and target
X = data.drop(columns=['SALARY'])
y = data['SALARY']


# In[136]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[137]:


X_train


# ### Training various machine learning regression models for choosing the best model based on the MSE, RMSE, and R-squared metrics.
# 
# 
# 

# #### Applying transformers for some features before training the model to be able to train:
# * Ordinal Encoder: For encoding the categorical features that have ordinal values.
# *  OneHot Encoder: For encoding the categorical features (transforming each unique value to a binary column).
# *  Standard Scaler: For scaling the numeric features.

# In[159]:


# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(categories=[['Analyst', 'Senior Analyst', 'Associate', 'Manager', 'Senior Manager', 'Director']]), ['DESIGNATION']),
        ('onehot', OneHotEncoder(), ["UNIT","SEX"]),
        ('scaler', StandardScaler(), ["EXP", "PAST EXP", "AGE", "RATINGS"])
    ],
    remainder='passthrough'  
)


# In[163]:


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regression": RandomForestRegressor(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
    "K-Nearest Neighbors Regression": KNeighborsRegressor(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net": ElasticNet()
}

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Fit and evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ("s1", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae, rmse, r2 = evaluate_model(y_test, y_pred)
    print(f"{name} -> MAE: {mae}, RMSE: {rmse}, R2: {r2}")


# ### Based on the MAE, RMSE, and R squared of all the models. The Random Forest Regression model gives the highest values. So, Tuning its parameters using GridSearchCV will give higher values.

# In[164]:


preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(categories=[['Analyst', 'Senior Analyst', 'Associate', 'Manager', 'Senior Manager', 'Director']]), ['DESIGNATION']),
        ('onehot', OneHotEncoder(), ["UNIT","SEX"]),
        ('scaler', StandardScaler(), ["EXP", "PAST EXP", "AGE", "RATINGS"])
    ],
    remainder='passthrough'  
)

pipeline = Pipeline([
        ("s1", preprocessor),
        ("model", RandomForestRegressor())
    ])


# In[168]:


# Define the parameter grid
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
}

# Perform grid search
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_


# In[171]:


pipeline = Pipeline([
        ("s1", preprocessor),
        ("model", RandomForestRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200))
    ])
pipeline


# In[173]:


# Fit the best model on the training data
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the mode
mae, rmse, r2 = evaluate_model(y_test, y_pred)
print(f"Random Forest Regression -> MAE: {mae}, RMSE: {rmse}, R2: {r2}")


# In[174]:


# Save the model
joblib.dump(pipeline, 'model.joblib')


# ### Now, I have a saved pipeline that includes the transformers and the tuned Random Forest Regression model.

# ### After Saving the model, we can use it for unseen data. 

# In[175]:


def load_model():
    return joblib.load('model.joblib')

def get_user_input():
    user_data = {
        'SEX': input("Enter Gender (M/F): "),
        'DESIGNATION': input("Enter Designation: "),
        'UNIT': input("Enter Business Unit: "),
        'AGE': int(input("Enter Age: ")),
        'LEAVES USED': int(input("Enter Leaves Used: ")),
        'LEAVES REMAINING': int(input("Enter Leaves Remaining: ")),
        'RATINGS': float(input("Enter Ratings: ")),
        'PAST EXP': int(input("Enter Past Experience: ")),
        'DOJ': input("Enter Date of Joining (MM-DD-YYYY): "),
        'CURRENT DATE': input("Enter Current Date (MM-DD-YYYY): ")
    }
    return user_data


# In[176]:


def preprocess_input(user_data):
    # Convert the user data to a DataFrame
    df = pd.DataFrame([user_data])
    
    # Calculate duration and the total experience
    df['DOJ'] = pd.to_datetime(df['DOJ'])
    df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])
    df['DURATION_YEARS'] = (df['CURRENT DATE'] - df['DOJ']).dt.days / 365
    df["EXP"] = df["PAST EXP"] + df["DURATION_YEARS"]
    
    # Drop the original date columns
    df = df.drop(columns=['DOJ', 'CURRENT DATE', 'DURATION_YEARS', 'LEAVES USED', 'LEAVES REMAINING'])
    
    print(df)
    
    return df


# In[177]:


def predict_salary(model, user_data):
    processed_data = preprocess_input(user_data)
    prediction = model.predict(processed_data)
    return prediction[0]


# In[151]:


# Get user input
user_data = get_user_input()
    
# Predict salary
predicted_salary = predict_salary(pipeline, user_data)
print(f"Predicted Salary: {predicted_salary}")


# In[152]:


# Get user input
user_data = get_user_input()
    
# Predict salary
predicted_salary = predict_salary(pipeline, user_data)
print(f"Predicted Salary: {predicted_salary}")


# In[153]:


# Get user input
user_data = get_user_input()
    
# Predict salary
predicted_salary = predict_salary(pipeline, user_data)
print(f"Predicted Salary: {predicted_salary}")


# In[178]:


# Get user input
user_data = get_user_input()
    
# Predict salary
predicted_salary = predict_salary(pipeline, user_data)
print(f"Predicted Salary: {predicted_salary}")


# In[179]:


# Get user input
user_data = get_user_input()
    
# Predict salary
predicted_salary = predict_salary(pipeline, user_data)
print(f"Predicted Salary: {predicted_salary}")

