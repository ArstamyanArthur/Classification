#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[1]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)


# In[2]:


import pandas as pd
pd.DataFrame(X).head()


# In[3]:


y


# In[4]:


pd.DataFrame(X).describe()


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[6]:


X_train.shape


# In[7]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=10000)
classifier.fit(X_train, y_train)


# In[8]:


y_pred = classifier.predict(X_test)


# In[9]:


y_pred


# In[10]:


probs_y = classifier.predict_proba(X_test)


# In[11]:


for x in probs_y[:10]:
    print([round(y,2) for y in x])


# In[12]:


import numpy as np
probs_y = np.round(probs_y, 2)
res = "{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("y_test",
                                                         "y_pred", 
                                                         "Setosa(%)",
                                                         "versicolor(%)",
                                                         "virginica(%)\n")
res += "-"*65+"\n"
res += "\n".join("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c) 
                 for x, y, a, b, c in zip(y_test, y_pred, probs_y[:,0], probs_y[:,1], probs_y[:,2]))
res += "\n"+"-"*65+"\n"
print(res)


# In[13]:


classifier.score(X_test, y_test)


# # Confusion Matrix

# In[14]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[15]:


df_cm = confusion_matrix(y_test, y_pred, normalize = "true")
df_cm


# In[16]:


# Plot confusion matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# confusion matrix sns heatmap 
ax = plt.axes()
df_cm = (confusion_matrix(y_test, y_pred, normalize="true")*100).astype(int)

sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = ax )
ax.set_title('Confusion Matrix')
plt.show()


# ## Implement gradient descent on logistic regression

# In[17]:


# example of updating one weight
# epochs = 50
# lr = 0.1
# for _ in range(epochs):
#     w1 -= lr * x1 * (y_hat - y) # y_hat is sigmoid function


# #### Remember that the derivative of loss function has the following formula
# <img src="Loss.png">

# In[29]:


import numpy as np


# In[30]:


class BinaryLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
    
    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.zeros((X.shape[1]))

        for i in range(self.n_iters):
            y_pred = self.sigmoid(X.dot(self.weights))
            dw = ((y_pred-y).dot(X))/len(y)
            self.weights -= self.learning_rate*dw

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(self.weights.dot(X.T))>0.5
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def score(self, X_t, y_t):
        y_pred = self.predict(X_t)
        return (y_pred == y_t).sum() / len(y_t)
    def coeff_(self):
        return self.weights


# # Questions
# 
# 1. How does logistic regression handle categorical variables?
# 2. Can logistic regression be used for classification problems with more than two classes?
# 3. Is there a way to do regularization in Logistic Regression?
# 4. How can you deal with imbalanced data in logistic regression?

# # Homework

# # 1. Solve classification problem using 'classification.csv' dataset

# In[40]:


import numpy as np
import pandas as pd
df = pd.read_csv('classification.csv')
# df.head(5)


# #### visualise the data, do some EDA

# In[41]:


# df.info()
df.describe(include='all')


# In[42]:


df.duplicated().sum()


# In[43]:


df.boxplot()


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns
# plt.figure(figsize=(7, 4))
# sns.countplot(x='ed', data=df, order=df['ed'].value_counts().index)
# sns.countplot(x='ed', data=df[df['default']==1], color='black', order=df['ed'].value_counts().index)
# plt.show()
plt.figure(figsize=(7, 4))
sns.barplot(x='ed', y='default', data=df)
plt.show()
print(df['ed'].value_counts())
# df.loc[df['default']==1, 'ed'].value_counts()/df['ed'].value_counts()


# In[45]:


df['default'].value_counts()


# In[46]:


sns.heatmap(df.corr(), annot=True, cmap='Blues')


# ##### target variable is 'default'. Apply feature selection, feature scaling, cross validation etc. (anything you think is needed)

# In[52]:


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
y = df.default
X = df.drop('default', axis=1)
n = np.random.randint(1, 1000)
print(n)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=n)
print(X_train.shape, X_test.shape)


# ## sklearn.preprocessing.OneHotEncoder

# In[53]:


ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(X_train[['ed']])
print(ohe.get_feature_names_out())
X_train[ohe.get_feature_names_out()] = ohe.transform(X_train[['ed']])
X_train.drop('ed', axis=1, inplace=True)

X_test[ohe.get_feature_names_out()] = ohe.transform(X_test[['ed']])
X_test.drop('ed', axis=1, inplace=True)
cols = list(X_train.columns.values)


# ##  sklearn.preprocessing.MinMaxScaler

# In[54]:


scaler = MinMaxScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # 2. Print accuracy, confusion matrix, precision and recall on train and test (and maybe validation) datasets.

# ##### do not use any libraries for metrics, implement yourself

# In[62]:


lg = BinaryLogisticRegression(learning_rate=0.1, n_iters=30000) # need to increase n_iters otherwise predicts all 0(False)
lg.fit(X_train, y_train)
y_test = np.array(y_test)
y_pred = lg.predict(X_test)
print(lg.score(X_test, y_test))
# print(lg.predict(X_train))
print(lg.predict(X_test), y_test.sum())
print(pd.Series(index=[1]+cols, data=lg.coeff_()))


# In[63]:


def vals(org, pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(org)):
        if org[i] == 1:
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred[i] == 1:
                fp += 1
            else:
                tn += 1
    return np.array([tp, fp, tn, fn])
def accuracy(tp, fp, tn, fn):
    numerator = tp + tn
    denominator = tp + tn + fp + fn
    return numerator / denominator
def recall(tp, fn):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return float('nan')
def ppv(tp, fp):        #         precision
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return float('nan')


# In[64]:


tp, fp, tn, fn = vals(y_test, y_pred)
cnf_mat_ontest = pd.DataFrame(data=[[tp, fn], [fp, tn]], columns=['Positive', 'Negative'], index=['True', 'False'])
print(cnf_mat_ontest, '\n')
print(f'Accuraccy = {accuracy(tp, fp, tn, fn)}\nRecall = {recall(tp, fn)}\nPrecision(ppv) = {ppv(tp, fp)}')


# In[ ]:




