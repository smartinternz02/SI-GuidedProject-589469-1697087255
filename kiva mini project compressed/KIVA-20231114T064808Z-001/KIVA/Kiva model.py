#!/usr/bin/env python
# coding: utf-8

# # <font color=green> Importing the libraries

# In[1]:


import numpy as np
import pandas as pd # data processing

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.simplefilter('ignore') #Stop showing warning messages


# # <font color=green> Importing dataset

# In[2]:


data = pd.read_csv("C:/Users/Bobby/Downloads/Implemetiting Model/kiva_loans.csv")


# In[3]:


data


# In[4]:


def preprocess_inputs(df):
    df = df.copy()
    
    return df


# In[5]:


Z = preprocess_inputs(data)


# In[6]:


Z


# # <font color=green> Getting the Preliminary Information about the Dataset

# In[7]:


Z.shape


# In[8]:


Z.info()


# In[9]:


Z.isna()


# # <font color=green>Checking for Missing Value in Each Column

# In[10]:


Z.isna().sum()


# # <font color=green> Checking percentage of Missing values
#    

# In[11]:


Z.isna().mean()*100


# In[12]:


total=Z.isnull().sum()
Percentnull=round((Z.isnull().sum()/list(Z.shape)[0])*100,5)
missing_values=pd.concat([total,Percentnull],axis=1,keys=['Total', 'Percent'])
missing_values


# # <font color=green> Findind the unique values from the dataset

# In[13]:


Z.nunique()


# # Unique Value in Each Column

# In[14]:


{column:len(Z[column].unique()) for column in Z.select_dtypes('object').columns }


# # <font color =green> Creating Preprocessing Function
#    # <font color =green>  Finding number of male and female borrower in the given loan

# In[15]:


def get_male_count(z):
    count = 0
    for gender in str(z).split(', '):
        if gender == 'male':
            count += 1
    return count

def get_female_count(z):
    count = 0
    for gender in str(z).split(', '):
        if gender == 'female':
            count += 1
    return count


# In[16]:


def preprocess_inputs(df):
   
    
    # Drop id column
    df = df.drop('id', axis=1)
    
    # Drop use and tags columns (avoiding NLP)
    df = df.drop(['use', 'tags'], axis=1)
    
    # Drop country_code and date columns (redundant information)
    df = df.drop(['country_code','currency','partner_id', 'date'], axis=1)
    
    # Drop region column (high-cardinality)
    df = df.drop(['region','activity','sector'], axis=1)
    
    df=df.drop (['posted_time','disbursed_time','funded_time'], axis=1)
    
     # Engineer gender count features
    df['male_count'] = df['borrower_genders'].apply(get_male_count)
    df['female_count'] = df['borrower_genders'].apply(get_female_count)
    df = df.drop('borrower_genders', axis=1)
    

    
    # Split df into X and y
    y = df['repayment_interval']
    Z = df.drop('repayment_interval', axis=1)
    
    # Encode target labels labels
    label_mapping = {
        'bullet': 0,
        'weekly': 1,
        'monthly': 2,
        'irregular': 3
    }
    y = y.replace(label_mapping)
    
   
    
    return df


# In[17]:


Z = preprocess_inputs(data)
Z


# In[18]:


def preprocess_inputs(df):
    
    # Split df into X and y
    y = df['repayment_interval']
    X = Z.drop('repayment_interval', axis=1)
    
    # Encode labels
    label_mapping = {
        'bullet': 0,
        'weekly': 1,
        'monthly': 2,
        'irregular': 3
    }
    y = y.replace(label_mapping)
    
    
     # create a dictionary mapping country names
    Country = {"Pakistan":0,"India":1,"Kenya":2,"Nicaragua":3,"El Salvador":4,"Tanzania":5,"Philippines":6,"Peru":7,
               "Senegal":8,"Cambodia":9,"Liberia":10,"Vietnam":11,"Iraq":12,"Honduras":13,"Palestine":14,"Mongolia":15,
               "United States":16,"Mali":17,"Colombia":18,"Tajikistan":19,"Guatemala":20,"Ecuador":21,"Bolivia":22,
               "Yemen":23,"Ghana":24,"Sierra Leone":25,"Haiti":26,"Chile":27,"Jordan":28,"Uganda":29,"Burundi":30,
               "Burkina Faso":31,"Timor-Leste":32,"Indonesia":33,"Georgia":34,"Ukraine":35,"Kosovo":36,"Albania":37,
               "The Democratic Republic of the Congo":38,"Costa Rica":39,"Somalia":40,"Zimbabwe":41,"Cameroon":42,
               "Turkey":43,"Azerbaijan":44,"Dominican Republic":45,"Brazil":46,"Mexico":47,"Kyrgyzstan":48,"Armenia":49,
               "Paraguay":50,"Lebanon":51,"Samoa":52,"Israel":53,"Rwanda":54}

    # create a dataframe with a country column
    X['country_labell'] = X['country'].map(Country)

    X = X.drop(['country'], axis=1)
    
    return X,y


# In[19]:


X,y = preprocess_inputs(data)


# In[20]:


X


# In[21]:


y


# # <font color=green> Fining if their is any missing values after all preprocessing is done

# In[22]:


X.isna().sum().sum()


# In[23]:


X.isnull()


# In[24]:


X.isnull().sum()


# In[25]:


def preprocess_inputs(df):
   #Train-test split
   X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
   
   # Scale X with a standard scaler
   scaler = StandardScaler()
   scaler.fit(X_train)
   
   X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
   X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
   
   return X_train, X_test, y_train, y_test


# In[26]:


X_train, X_test, y_train, y_test = preprocess_inputs(data)


# In[27]:


X_train


# In[28]:


X_train.mean()


# In[29]:


X_train.var()


# In[30]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train.shape)


# In[31]:


print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# In[ ]:





# # <font color=green> Prediction using SVC 

# In[32]:


from sklearn.svm import SVC 
svm=SVC(kernel='rbf',random_state=0)
svm.fit(X_train,y_train)


# In[33]:


y_pred_1=svm.predict(X_test)


# In[34]:


# check the accuracy on the training set
print('Training set : ',svm.score(X_train, y_train))
print('Testing set  : ',svm.score(X_test, y_test))


# In[35]:


accuracy_1=svm.score(X_test,y_test)
print ("Accuracy_SVM:",accuracy_1*100)


# #  <font color=green>Prediction using Decision Tree

# In[36]:


dt = DecisionTreeClassifier()


# In[37]:


# Fit the model on the training data
dt.fit(X_train, y_train)


# In[38]:


# Predict the labels of the test data
y_pred = dt.predict(X_test)


# In[39]:


# check the accuracy on the training set
print('Training set : ',dt.score(X_train, y_train))
print('Testing set  : ',dt.score(X_test, y_test))


# In[40]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy_DT:', accuracy*100)


# #  <font color=green>Prediction using Random Forest

# In[41]:


rand_forest = RandomForestClassifier(random_state=42)


# In[42]:


# Fit the model to the data
rand_forest.fit(X_train, y_train)


# In[43]:


predictionRF= rand_forest .predict(X_test)
# check the accuracy on the training set
print('Training set : ',rand_forest.score(X_train, y_train))
print('Testing set  : ',rand_forest.score(X_test, y_test))


# In[44]:


accuracy_RF=rand_forest.score(X_test, y_test)
print ("Accuracy_RF:",accuracy_RF*100)


# # <font color=green> Prediction using LogisticRegression

# In[45]:


# create a logistic regression model
lr = LogisticRegression()


# In[46]:


# fit the model on the training data
lr.fit(X_train, y_train)


# In[47]:


# use the model to make predictions on the test data
y_pred = lr.predict(X_test)


# In[48]:


# calculate the accuracy of the predictions
accuracy_LR = np.mean(y_pred == y_test)


# In[49]:


# print the accuracy
print("Accuracy_LR:", accuracy*100)


# # <font color=green> Applying KNN

# In[50]:


from sklearn.neighbors import KNeighborsClassifier  
knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn.fit(X_train, y_train)  


# In[51]:


#Predicting the test set result  
y_pred= knn.predict(X_test)  


# In[52]:


# Calculate accuracy of the model

from sklearn.metrics import accuracy_score
accuracy_KNN = accuracy_score(y_test, y_pred)
print(f'Accuracy_KNN: {accuracy*100}')


# # <font color=green> Comparison of Accuracies

# In[53]:


classifiers = [svm,dt,rand_forest,lr,knn]
classifiers


# In[54]:


# Create a table to compare the accuracies of each model
accuracy_df = pd.DataFrame({
    'Model': ['SVM','DecisionTree','Random Forest',' LogisticRegression', 'KNN'],
    'Accuracy': [accuracy_1*100, accuracy*100,accuracy_RF*100, accuracy_LR*100,accuracy_KNN*100]
})
print(accuracy_df)


# # <font color=green> Plot the comparison bar graph

# In[55]:


models = ['SVM','DecisionTree','Randomforest',' LogisticRegression', 'KNN']

accuracies = [accuracy_1*100, accuracy*100,accuracy_RF*100, accuracy_LR*100,accuracy_KNN*100]
plt.bar(models, accuracies, color='blue')

# Add title and axis labels
plt.title('Comparison of Model Accuracies')
plt.xlabel('Models')
plt.ylabel('Accuracy')


# # <font color=green> Exporting the model

# In[56]:


import joblib
classifier=joblib.dump(rand_forest,'Kiva_classification_RF.pkl')

