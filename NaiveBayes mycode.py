#!/usr/bin/env python
# coding: utf-8

# # Homework

# In[83]:


# Please implement 2 classes below


# In[6]:


import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

df = pd.read_csv('classification.csv')
y = df.default
X = df.drop(columns = ['default', 'ed'])

n = np.random.randint(1, 1000)
print(n)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=n)
y_train = np.array(y_train)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler() 
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# ##  Naive Bayes

# In[7]:


class NaiveBayes_mine:                           
    def __init__(self):
        self.classes = None
        self.n_feat = None
        self.prior = None
        self.dist = {}
        
    def fit(self, X_train, y_train):   
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.n_feat = X_train.shape[1]
        self.classes, counts = np.unique(y_train, return_counts=True)
        self.prior = dict(zip(self.classes, counts/len(y_train)))
        for cls in self.classes:
            new = X_train[y_train == cls]
            for index in range(self.n_feat):
                u = new[:, index].mean()
                std = np.std(new[:, index])
                self.dist[(cls, index)] = (u, std)
#                 print(cls, index, self.dist)
        
    def predict(self, X_test):
        y_pred = []
        for x in np.array(X_test):
            probs = []
            for cls in self.classes:
                result = np.log(self.prior[cls])
#                 print(x, cls, result)
                for i in range(self.n_feat):
#                     print(x[i])
                    u, std = self.dist[(cls, i)]
                    result += -np.log(std)- 0.5 * ((x[i]-u)/std)**2
                probs.append(result)
#                 print(probs)
            y_pred.append(self.classes[np.argmax(probs)])
        return np.array(y_pred)
    def score(self, X_test, y_test):
        return (self.predict(X_test) == y_test).sum()/len(y_test)
    
class NaiveBayes:
    def __init__(self):
        self.means = []
        self.variances = []
        self.priors = []
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        for c in np.unique(y):
            X_c = X[y==c]
            self.means.append(X_c.mean(axis=0))
            self.variances.append(X_c.var(axis=0))
            self.priors.append(len(X_c)/len(X))
            
        self.means = np.array(self.means)
        self.variances = np.array(self.variances)
        self.priors = np.array(self.priors)

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for i, x in enumerate(X):
            y = self.pdf(x, self.means, self.variances)
            y = np.log(y).sum(axis=1)
            y += np.log(self.priors)
            y_pred.append(np.argmax(y))
        return np.array(y_pred)

    def score(self, X, y):
        return (self.predict(X)==y).sum() / len(y)
        
    def pdf(self, x, mean, var):
        return norm.pdf(x, mean, var)
    
nb_mine = NaiveBayes_mine()
nb_mine.fit(X_train, y_train)

nb = NaiveBayes()
nb.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB
nbg = GaussianNB()
nbg.fit(X_train, y_train)

print('Minimum accuracy - ', 517/700, '\nMyclass accuracy - ', nb_mine.score(X_test, y_test), 
      '\nSklearn GaussianNB - ', nbg.score(X_test, y_test), '\nNairis class accuracy - ', nb.score(X_test, y_test))

# real distributions
df_0 = df[df['default']==0]
df_1 = df[df['default']==1]
# df_0.hist()
# df_1.hist()


# ## LinearDiscriminantAnalysis 

# In[8]:


class LDA_mine:  
    def __init__(self):
        self.classes = None
        self.prior = None
        self.means = {}
        self.E = None
        
    def fit(self, X_train, y_train):   
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        n_feat = X_train.shape[1]
        self.classes, counts = np.unique(y_train, return_counts=True)
        self.prior = dict(zip(self.classes, counts/len(y_train)))
        
        self.E = np.zeros((n_feat, n_feat))
        for cls in self.classes:
            new = X_train[y_train == cls]
            mean = np.mean(new, axis=0)
            self.means[cls] = mean
            self.E += (new - mean).T.dot((new - mean))
        self.E /= (len(y_train)-len(self.classes))
#         print(self.E)
    def predict(self, X_test):
        y_pred = []
        for x in np.array(X_test):
            probs = []
            for cls in self.classes:
                mean = self.means[cls]
                result = self.prior[cls] * np.exp(-((x-mean).dot(np.linalg.inv(self.E)).dot((x-mean).T))/2) 
                probs.append(result)
            y_pred.append(self.classes[np.argmax(probs)])
        return np.array(y_pred)
    def score(self, X_test, y_test):
        return (self.predict(X_test) == y_test).sum()/len(y_test)  
    
class LDA:
    def __init__(self):
        self.means = []
        self.cov_mtrx = None
        self.priors = []
        
    def fit(self, X, y):
        self.cov_mtrx = np.zeros((X.shape[1], X.shape[1]))

        for c in np.unique(y):
            X_c = X[y==c]
            self.means.append(X_c.mean(axis=0))
            self.priors.append(len(X_c)/len(X))
            new_diff = X_c - X_c.mean(axis=0)
            self.cov_mtrx += new_diff.T @ new_diff
        
        self.cov_mtrx = self.cov_mtrx / (len(X) - len(self.means))
        self.means = np.array(self.means)
        self.priors = np.array(self.priors)
        
    def predict(self, X):
        y_pred = []
        for x in X:
            y = []
            for p, mean in zip(self.priors, self.means):
                w1 = np.linalg.inv(self.cov_mtrx) @ mean
                w0 = np.log(p) -0.5 * mean @ w1
                y.append(x @ w1 + w0)
            y_pred.append(np.argmax(y))
        return np.array(y_pred)

    def score(self, X, y):
        return (self.predict(X)==y).sum() / len(y)
        
    def pdf(self, x, mean, var):
        return norm.pdf(x, mean, var)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
lda = LDA_mine()
lda.fit(X_train, y_train)                                                  
sklearn_lda = LinearDiscriminantAnalysis()
sklearn_lda.fit(X_train, y_train)
print('Minimum accuracy - ', 517/700, '\nMyclass accuracy - ',
      lda.score(X_test, y_test), '\nSklearn  - ', sklearn_lda.score(X_test, y_test))


# In[ ]:





# In[ ]:




