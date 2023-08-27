#!/usr/bin/env python
# coding: utf-8

# In[74]:


#importing library
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#creating data
x,y=make_classification(n_samples=100, n_features=2, n_informative=2,n_redundant=0,  n_classes=2, n_clusters_per_class=1, class_sep=0.3, random_state=10)


# In[3]:


#plot scatterplot
sns.scatterplot(x[:,0],x[:,1],hue=y)


# In[18]:


#support vector machine
svc=SVC()
svc.fit(x,y)
y_pred=svc.predict(x)
y_pred


# In[19]:


#confusion matrix
confusion_matrix(y,y_pred)


# In[20]:


#ploting decision boundary
plot_decision_regions(x,y,clf=svc,legend=2)


# In[21]:


#logistic regression
lr=LogisticRegression()
lr.fit(x,y)
y_pred=lr.predict(x)
y_pred


# In[128]:


#confusion matrix
confusion_matrix(y,y_pred)


# In[76]:


#ploting s=decision boundary
plot_decision_regions(x,y,clf=lr,legend=2)


# In[39]:


#Decision tree
dt=DecisionTreeClassifier(max_depth=None)
dt.fit(x,y)
y_pred=dt.predict(x)
y_pred


# In[40]:


#confusion matrix
confusion_matrix(y,y_pred)


# In[41]:


#ploting s=decision boundary
plot_decision_regions(x,y,clf=dt,legend=2)


# In[42]:


from sklearn.tree import plot_tree
fig=plt.figure(figsize=(25,20))
plot_tree(dt,filled=True)


# In[47]:


#KNN
knn=KNeighborsClassifier()
knn.fit(x,y)
y_pred=knn.predict(x)
y_pred


# In[48]:


#confusion matrix
confusion_matrix(y,y_pred)


# In[50]:


#ploting decision boundary
plot_decision_regions(x,y,clf=knn,legend=2)


# In[68]:


#Naive bayes
nb=GaussianNB()
nb.fit(x,y)
y_pred=nb.predict(x)
y_pred


# In[69]:


#confusion matrix
confusion_matrix(y,y_pred)


# In[70]:


#ploting decision boundary
plot_decision_regions(x,y,clf=nb,legend=2)


# In[94]:


#Random Forest
rf=RandomForestClassifier()
rf.fit(x,y)
y_pred=rf.predict(x)
y_pred


# In[95]:


#confusion matrix
confusion_matrix(y,y_pred)


# In[96]:


#ploting s=decision boundary
plot_decision_regions(x,y,clf=rf,legend=2)


# In[ ]:




