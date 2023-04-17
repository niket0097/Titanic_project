#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


print("****IMPORTING DATA****")


# In[3]:


test=pd.read_csv('test.csv',index_col=0)
train=pd.read_csv('train.csv',index_col=0)


# In[4]:


print("first 5 rows of train data:-\n",train.head())


# In[5]:


print("****UNIVARIATE ANALYSIS****")


# In[6]:


print("Information about columns in train data:-\n")
print(train.info())


# In[7]:


print("Numbers of NaN values in train data\n\n",train.isnull().sum())


# In[8]:


print("Obervation:")
print("1. The Cabin column is having almost 80% NaN values,so guessing values can affect the prediction,so should be dropped.")
print("2. Name, Ticket number, Embarked column can also be dropped as it is less likely to affect survivality.")
print("3. Pclass, Sex should be converted using One hot encoder.")


# In[9]:


sns.kdeplot(data=train['Fare'])
plt.title('KDE of Fare')
plt.show()


# In[37]:


print('Description of Fare data:\n',train['Fare'].describe())


# In[11]:


print("Observation: ")
print("Fare data is very scattered(std=46) and very large outliers, so it should be converted to categories & drop original data")


# In[12]:


print("****MULTIVARIATE ANALYSIS****")


# In[13]:


sns.countplot(data=train,x='Parch',hue='Survived')
plt.title("No. of people survived/not-survived wrt parch")
plt.show()


# In[14]:


sns.countplot(data=train,x='SibSp',hue='Survived')
plt.title("No. of people survived/not-survived wrt SibSp")
plt.show()


# In[15]:


print("Observation: It can be concluded that People with parents/children aboard has more chance of survival, and a new feature which shows whether the person has parents/children aboard can be added.")
print("Similar observation can drawn from SibSp graph, and a new feature whether the person has spouse/siblings aboarded can be added.")


# In[16]:


sns.kdeplot(data=train,x='Age',hue='Survived')
plt.xticks([0,10,20,30,40,50,60,70,80])
plt.title("KDE plot of Age showing survived & not-survived people")
plt.show()


# In[17]:


print("Observation: It can be obsevred that children & aged person has more chance of survival.")
print("A new feature wrt Age group can be made- 0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60+")


# In[18]:


print("****Feature engg & data processing****")


# In[19]:


x=train.copy(deep=True)
xt=test.copy(deep=True)


# In[20]:


y=x.pop('Survived')
drop_col=['Name','Ticket','Cabin','Embarked']
x.drop(drop_col,axis=1,inplace=True)
xt.drop(drop_col,axis=1,inplace=True)


# In[21]:


from sklearn.compose import ColumnTransformer as ct
from sklearn.pipeline import Pipeline as pl
from sklearn.impute import SimpleImputer as si
from sklearn.preprocessing import OneHotEncoder as ohc, StandardScaler
from sklearn.model_selection import cross_val_score as cvs
from xgboost import XGBClassifier as xg


# In[ ]:





# In[22]:


class newFeature1:
    def __init__(self, feature_names):
        self.feature = feature_names   
    def fit( self, x, y = None ):
        return self
    def transform(self, x, y=None):
        new_col=[]
        for col in self.feature:
            colname=str(col)+'_new'
            new_col.append(colname)
            if col=='Age':
                bins = [-1, 10, 20, 30, 40, 50, 60, 100]
                names = ['<10', '10-20', '20-30','30-40','40-50','50-60', '60+']
            if col=='Fare':
                bins = [-1,8,14.5,31,65.5,513]
                names = ['0-8', '8-14.5', '14.5-31', '31-65.5', '65.5-513']                
            x[colname] = pd.cut(x[col], bins, labels=names)
        x.drop(['Fare'],axis=1,inplace=True)
        return self
    


# In[23]:


class newFeature2:
    def __init__(self, feature_names):
        self.feature = feature_names   
    def fit( self, x, y = None ):
        return self
    def transform(self, x, y=None):
        for col in self.feature:
            x[col+"_new"]=x[col].astype(bool)
        return self


# In[24]:


obj1=newFeature1(['Age','Fare'])
obj1.transform(x)
obj1.transform(xt)


# In[25]:


obj2=newFeature2(['SibSp','Parch'])
obj2.transform(x)
obj2.transform(xt)


# In[26]:


num_col=['Age','SibSp','Parch']
cat_col=['Pclass','Sex','Age_new','Fare_new','SibSp_new','Parch_new']


# In[27]:


num_t=pl(steps=[('impute1',si(strategy='median'))])


# In[28]:


cat_t=pl(steps=[('impute',si(strategy='most_frequent')),('Ohot',ohc(handle_unknown='ignore'))])


# In[29]:


prep=ct(transformers=[('num',num_t,num_col),('cat',cat_t,cat_col)])


# In[30]:


d_final=5
n_final=50
score_final=0
for d in [5,10,15]:
    for n in [100,200,500,1000]:    
        model=xg(n_estimators=n,learning_rate=0.05,max_depth=d)
        my_model=pl(steps=[('pre',prep),('model',model)])
        score = cvs(my_model,x,y,cv=5,scoring='roc_auc')
        print(score.mean())
        if score.mean()>score_final:
            d_final=d
            n_final=n
            score_final=score.mean()


# In[31]:


model2=xg(n_estimators=n_final,learning_rate=0.05,max_depth=d_final)
my_model2=pl(steps=[('pre',prep),('model',model2)])


# In[32]:


my_model2.fit(x,y)


# In[33]:


prediction=my_model2.predict(xt)


# In[34]:


output=pd.DataFrame({'PassengerID':test.index,'Survived':prediction})
output.to_csv("Submission.csv",index=False)


# In[ ]:





# In[ ]:




