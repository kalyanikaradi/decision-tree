#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('HR-Employee-Attrition.csv')


# In[3]:


data.describe()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.head()


# In[7]:


data.describe(include=['O'])#to find inf about categorical


# In[8]:


#univariant analysis
get_ipython().system('pip install sweetviz')
#library


# In[9]:


#sweetviz is library which automatically gives all the digrams 
import sweetviz as sv
my_report = sv.analyze(data)
my_report.show_html()


# In[10]:


#create new df with cattegorical var
data1=data[['BusinessTravel',
    'Department',
    'EducationField',
    'Gender',
    'JobRole',
    'MaritalStatus',
     'Over18',
    'OverTime']]


# In[11]:


data1


# In[12]:


#to convert categorical data to numerical data
categorical_col=[]#list
for column in data.columns:#for loop access col
        if data[column].dtype == object and len (data[column].unique())<= 50:
            categorical_col.append(column)
            print(f"{column} : {data[column].unique()}")
            print("===================================")


# In[13]:


#plotting how categorical coreelated with target
plt.figure(figsize=(50,50),facecolor='white')
plotnumber = 1

for column in data1:
    if plotnumber<=16:
        ax = plt.subplot(4,4,plotnumber)
        sns.countplot(x=data1[column].dropna(axis=0),hue=data.Attrition)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Attrition',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[14]:


numerical_col = []#list for numerical
for column in data.columns:#accessing col from datasets
    if data[column].dtype == int and len(data[column].unique()) >= 10:
        numerical_col.append(column)#inserting those col in list


# In[15]:


numerical_col


# In[ ]:





# In[16]:


#discrete data separting
data3 = data[['Education',
             'EmployeeCount',
             'JobInvolvement',
             'JobLevel',
             'JobSatisfaction',
             'NumCompaniesWorked',
             'PerformanceRating',
             'RelationshipSatisfaction',
             'StandardHours',
             'StockOptionLevel',
             'TrainingTimesLastYear',
             'WorkLifeBalance']]


# In[17]:


data3


# In[18]:


plt.figure(figsize=(50,50),facecolor='white')
plotnumber = 1

for column in data3:
    if plotnumber<=16:
        ax = plt.subplot(4,4,plotnumber)
        sns.countplot(x=data3[column].dropna(axis=0),hue=data.Attrition)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Attrition',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[19]:


data2=data[['Age',
           'DailyRate',
           'DistanceFromHome',
           'EmployeeNumber',
           'HourlyRate',
           'MonthlyIncome',
           'MonthlyRate',
           'NumCompaniesWorked',
           'PercentSalaryHike',
           'TotalWorkingYears',
           'YearsAtCompany',
           'YearsInCurrentRole',
           'YearsSinceLastPromotion',
           'YearsWithCurrManager']]


# In[20]:


data2


# In[21]:


plt.figure(figsize=(50,50),facecolor='white')
plotnumber = 1

for column in data2:
    if plotnumber<=16:
        ax = plt.subplot(4,4,plotnumber)
        sns.histplot(x=data2[column].dropna(axis=0),hue=data.Attrition)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Attrition',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[22]:


data.isnull().sum()


# In[23]:


data1.head()


# In[24]:


data.Attrition.unique()


# In[25]:


#manual encoding attrition(where we give value)
data.Attrition=data.Attrition.map({'Yes':1,'No':0})
data.head()


# In[26]:


data.BusinessTravel.unique()


# In[27]:


#encoding businessTravel manual encoding
data.BusinessTravel=data.BusinessTravel.map({'Travel_Frequently':2,'Travel_Rarely':1,'Non-Travel':3})
data.head()


# In[28]:


#department
data.Department.unique()


# In[29]:


data.Department=data.Department.map({'Research & Development':2,'Sales':1,'Human Resources':0})


# In[30]:


data.head()


# In[31]:


#education field
data.EducationField.unique()


# In[32]:


data.EducationField=data.EducationField.map({'Life Sciences':5, 'Other':6, 'Medical':3, 'Marketing':1,
       'Technical Degree':2, 'Human Resources':4})


# In[33]:


data.head()


# In[34]:


data.Gender.value_counts()


# In[35]:


#encoding gender
data.Gender=pd.get_dummies(data.Gender,drop_first=True)


# In[36]:


data.Gender


# In[37]:


#job role
data.JobRole.value_counts()


# In[38]:


data.JobRole=data.JobRole.map({'Sales Executive':8,'Research Scientist':6,'Laboratory Technician':3,  'Manufacturing Director':7,
'Healthcare Representative':4,
'Manager':2,                   
'Sales Representative':6,
'Research Director': 5,            
'Human Resources':1 })


# In[39]:


data.head()


# In[40]:


#encoding marital status usiing label 
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
data.MaritalStatus=label.fit_transform(data.MaritalStatus)


# In[41]:


data.MaritalStatus


# In[42]:


data.OverTime.value_counts()


# In[43]:


data.OverTime=label.fit_transform(data.OverTime)


# In[44]:


data.head()


# In[45]:


#feature selection
plt.figure(figsize=(30,30))
sns.heatmap(data2.corr(), annot=True, cmap='RdYlGn', annot_kws={'size':15})


# In[46]:


data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis='columns',inplace=True)


# In[47]:


data.describe()


# In[48]:


#model Creation
#creating independent and dependent variable
X = data.drop('Attrition', axis=1)#independent
Y= data.Attrition#dependent


# In[49]:


#smote is used balance data which before data is overfitting
#balancing data
from collections import Counter
from imblearn.over_sampling import SMOTE#data balance
sm = SMOTE()#obj creating
print(Counter(Y))
X_sm,Y_sm=sm.fit_resample(X,Y)
print(Counter(Y_sm))


# In[50]:


#!pip uninstall scikit-learn --yes


# In[51]:


#!pip uninstall imblearn --yes


# In[52]:


#!pip install scikit-learn==1.2.2


# In[53]:


#!pip install imblearn


# In[54]:


#preparing train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_sm, Y_sm, test_size=0.25, random_state=42)


# In[55]:


#imp
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
Y_hat=dt.predict(X_test)
Y_hat


# In[56]:


Y_train_predict=dt.predict(X_train)#predicting data
Y_train_predict


# In[57]:


#evalute model
from sklearn.metrics import accuracy_score, classification_report, f1_score

Y_train_predict=dt.predict(X_train)
acc_train=accuracy_score(Y_train,Y_train_predict)
acc_train


# In[58]:


print(classification_report(Y_train,Y_train_predict))


# In[59]:


pd.crosstab(Y_train,Y_train_predict)


# In[60]:


#test accuracy
test_acc=accuracy_score(Y_test,Y_hat)
test_acc


# In[61]:


#test score
test_f1=f1_score(Y_test,Y_hat)
test_f1


# In[62]:


print(classification_report(Y_test,Y_hat))


# In[63]:


pd.crosstab(Y_test,Y_hat)


# In[64]:


#hyper parameter tuning
from sklearn.model_selection import GridSearchCV


# In[65]:


#creating dict
params={
    'criterion':('gini','entropy'),
    'splitter':('best','random'),
    'max_depth':(list(range(1,20))),
    'min_samples_split':[2,3,4],
    'min_samples_leaf':list(range(1,20)),
}

tree_clf=DecisionTreeClassifier(random_state=3)
tree_cv = GridSearchCV(tree_clf, params, scoring='f1',n_jobs=-1, verbose=1, cv=3)


tree_cv.fit(X_train,Y_train)
best_params = tree_cv.best_params_
print(f"Best parameters:{best_params})")


# # Random Forest Implementation

# In[66]:


X_train


# In[67]:


Y_train


# In[68]:


from sklearn.ensemble import RandomForestClassifier#importing random forest
rf_clf = RandomForestClassifier(n_estimators=100)#taking 100 decision
rf_clf.fit(X_train,Y_train)


# In[70]:


from sklearn.ensemble import RandomForestClassifier#ensemble techique is used to run multiple tress


# In[71]:


RandomForestClassifier()


# In[72]:


Y_predict=rf_clf.predict(X_test)


# In[73]:


print(classification_report(Y_test,Y_predict))


# In[74]:


f_score=f1_score(Y_test,Y_predict)
f_score


# In[76]:


#hyperparameter tunning
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]#list comprehension using loop
max_features= ['auto','sqrt']#max num feature allowed
max_depth = [int(x) for x in np.linspace(10,110, num=11)]
max_depth.append(None)
min_samples_split = [2,5,10]#min num samples required to internal node
min_samples_leaf = [1,2,4]#min num samples for leaf node
bootstrap = [True, False]#sampling
#dict for hyperparmeter
random_grid ={'n_estimators':n_estimators,'max_features':max_features,
             'max_depth':max_depth,'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf,'bootstrap':bootstrap}

rf_clf1 = RandomForestClassifier(random_state=42)

rf_cv = RandomizedSearchCV(estimator=rf_clf1, scoring='f1', param_distributions=random_grid, n_iter=100, cv=3,verbose=2,random_state=42, n_jobs=-1)


# In[77]:


rf_cv.fit(X_train, Y_train)
rf_best_params = rf_cv.best_params_
print(f"Best paramters:{rf_best_params}")


# In[83]:


rf_clf2 = RandomForestClassifier(**rf_best_params)
rf_clf2.fit(X_train, Y_train)#traing
Y_predict=rf_clf2.predict(X_test)#testing
f1_score=f1_score(Y_test,Y_predict)#checking performace


# In[84]:


f1_score


# In[85]:


print(classification_report(Y_test,Y_predict))


# In[ ]:




