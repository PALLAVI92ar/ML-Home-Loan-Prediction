# Home loan application data
# Prediction of home loan application into Safe and Risky loan by 0 and 1 binary classification
# Data Validation, Data Exploration, classification technique, 
# Decision Trees using Gini and Entropy methods, Bagging, Random forests.

# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing dataset
df = pd.read_csv("loan_data_set.csv")

# Preview data
pd.set_option('display.max_columns', None)
df.head()

# Dataset dimensions - (rows, columns)
df.shape

# Features data-type
df.info()

# Checking blanks of all the columns 
df.isnull().sum()

# Statistical summary
df.describe()

df.groupby("Gender").size()
# Male are around 81%, so fill the missing values with "Male" for Gender column
df.Gender = df.Gender.fillna('Male')

df.groupby("Married").size()
# "Yes" has around 65% so, fill the missing values with "Yes"
df.Married = df.Married.fillna('Yes')

# Fill the missing values by its mode value
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df.groupby("Self_Employed").size()
# Since "No" is around 82% so, fill the missing values of this column with "No"
df.Self_Employed = df.Self_Employed.fillna('No')

# Here fill the missing values by its mean 
df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())

# Fill the 'Loan_Amount_Term' column missing values by its mode value
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

df.groupby("Credit_History").size()
# "1" is high when compare to 0 so, fill the missing values with "1" here
df.Credit_History = df.Credit_History.fillna('1.0')
df['Credit_History'] = df.Credit_History.astype(float)

df.groupby("Education").size()

df.groupby("Property_Area").size()

df.groupby("Loan_Status").size()

df.isnull().sum()
df.describe()
df.head()
df.tail()
df.dtypes

# Label encoding for object data
from sklearn.preprocessing import LabelEncoder
col = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
# Encode labels with value 0 and 1.
le = LabelEncoder() 
for i in col:
    # Fit label encoder and return encoded labels
    df[i] = le.fit_transform(df[i])
df.dtypes

# Heatmap- finds correlation between Independent and dependent attributes
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True)
plt.show()

# Selecting the variables using correlation
df.corr()

# Splitting  X and Y variables
X = df.iloc[:,1:12]
list(X)

Y = df['Loan_Status']
Y

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scale=scaler.fit_transform(X)

# Splitting up of X and Y variables into train and test cases
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test = train_test_split(X_scale,Y, test_size=0.30,random_state = 42)
X_train.shape,X_test.shape, Y_train.shape,Y_test.shape

# Implementing the Decision tree model by gini index method
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

# Graphical visualization of how nodes are distributed into tree
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(dt, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

dt.tree_.node_count
dt.tree_.max_depth

Y_pred_dt = dt.predict(X_test)

print(f"Decision tree has {dt.tree_.node_count} nodes with maximum depth covered up to {dt.tree_.max_depth}")

# Import the metrics class
from sklearn.metrics import confusion_matrix, accuracy_score

cm=confusion_matrix(Y_test,Y_pred_dt)
print("Confusion matrix for home loan sanction process risky and safe is:",cm)

acc=accuracy_score(Y_test,Y_pred_dt).round(3)
print("Accuracy in Home loan sanction is:",acc)

# Further tuning is required to decide about max depth value
# Apply grid search cv method and pass levels and
# Look out for the best depth at this place and from this can fit the model to get home loan sanctioning error rate

from sklearn.model_selection import GridSearchCV
levels = {'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}

DTgrid = GridSearchCV(dt, cv = 10, scoring = 'accuracy', param_grid = levels)

DTgridfit = DTgrid.fit(X_train,Y_train)

DTgridfit.fit(X_test,Y_test)

print(DTgridfit.best_score_)

print(DTgridfit.best_estimator_)

###############################################################################
# Implementing the model-->Decision tre by Entropy method

dt1 = DecisionTreeClassifier(criterion='entropy')

dt1.fit(X_train, Y_train)

dt1.tree_.node_count
dt1.tree_.max_depth

Y_pred_dt1 = dt1.predict(X_test)

print(f"Decision tree has {dt1.tree_.node_count} nodes with maximum depth covered up to {dt1.tree_.max_depth}")

from sklearn.metrics import confusion_matrix, accuracy_score

cm1=confusion_matrix(Y_test,Y_pred_dt1)
print("Confusion matrix for home loan sanction process risky and safe is:",cm1)

acc1=accuracy_score(Y_test,Y_pred_dt1).round(3)
print("Home loan sanctioning process accuracy result is:",acc1)

from sklearn.model_selection import GridSearchCV
levels = {'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}

DT1grid = GridSearchCV(dt1, cv = 10, scoring = 'accuracy', param_grid = levels)

DT1gridfit = DT1grid.fit(X_train,Y_train)

DT1gridfit.fit(X_test,Y_test)

print(DT1gridfit.best_score_)

print(DT1gridfit.best_estimator_)

###############################################################################

# DecisionTreeClassifier= dt--> base learner, To the baselearner dt apply Bagging(splits on the datapoints)

from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(base_estimator=dt,max_samples=0.6, n_estimators=500, random_state = 8)
bag.fit(X_train, Y_train)
Y_pred_bag = bag.predict(X_test)

from sklearn.metrics import accuracy_score 
accuracy_score(Y_pred_bag,Y_test).round(3)

# Grid Search Cv method
from sklearn.model_selection import GridSearchCV
samples = {'max_samples': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

grid = GridSearchCV(bag, cv = 10, scoring = 'accuracy', param_grid = samples)

gridfit = grid.fit(X_train,Y_train)

gridfit.fit(X_test,Y_test)

print(gridfit.best_score_)

print(gridfit.best_estimator_)

#################################################################################
# Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_features = 6,  n_estimators = 500, random_state =24)

RFC.fit(X_train, Y_train)
Y_pred_rf = RFC.predict(X_test)

from sklearn.metrics import accuracy_score 
accuracy_score(Y_pred_rf,Y_test).round(3)

# Grid Search Cv method
from sklearn.model_selection import GridSearchCV
samples = {'max_features': [1,2,3,4,5,6,7,8,9,10,11]}

RFgrid = GridSearchCV(RFC, cv = 10, scoring = 'accuracy', param_grid = samples)

RFgridfit = RFgrid.fit(X_train,Y_train)

RFgridfit.fit(X_test,Y_test)

print(RFgridfit.best_score_)

print(RFgridfit.best_estimator_)





