#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', n_components=0.97, variance_t=0.001, corr_t= 0.9, random_state=None):
        """
        Initialize the MyModel with hyperparameters.

        Parameters:
        - C: Regularization parameter
        - kernel: Kernel function for the SVM ('linear', 'rbf', 'poly', etc.)
        - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        - random_state: Seed for random number generation
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.n_components = n_components
        self.variance_t = variance_t
        self.corr_t = corr_t
        self.model = None
        self.drop = None
        self.sscaler = None,
        self.pcac = None

    def fit(self, x_train, y_train):
        """
        Fit the SVM model to the training data.

        Parameters:
        - X: Training data features
        - y: Target labels
        """
        
        #using variance threshold
        self.drop = self.variance_treshould_invf(x_train)
        
        vx_train = x_train.drop(columns=self.drop,axis=1)
        
        # using correlation
        next_drop = self.correlation(vx_train)
        self.drop = self.drop + list(next_drop)
        
        cvx_train = vx_train.drop(columns=next_drop,axis=1)
        
        self.sscaler = StandardScaler()

        # fit the scaler
        scvx_train = pd.DataFrame(self.sscaler.fit_transform(cvx_train), columns=cvx_train.columns)
        
        # define the pca
        self.pcac = PCA(n_components= self.n_components, svd_solver="full")

        pscvx_train = self.pcac.fit_transform(scvx_train)
        
        
        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, random_state=self.random_state)
        self.model.fit(pscvx_train, y_train)

    def predict(self, X):
        """
        Make predictions using the trained SVM model.

        Parameters:
        - X: Input data for predictions

        Returns:
        - Predicted labels
        """
        
        if self.model is None:
            raise ValueError("Model has not been trained. Please call fit() first.")
            
        cvx_valid = X.drop(columns=self.drop,axis=1)
        
        scvx_valid = pd.DataFrame(self.sscaler.transform(cvx_valid), columns=cvx_valid.columns)
        
        pscvx_valid = self.pcac.transform(scvx_valid)
        
        return self.model.predict(pscvx_valid)
    
    def variance_treshould_invf(self, X):
        should_drop = []
        stds = X.describe().loc["std"]
        max_variance = max(stds)**2
        for i in range(0, len(stds)):
            if (stds[i]**2)< (self.variance_t):
                should_drop.append(f"feature_{i+1}")
        return should_drop
    
    def correlation(self, X):
      col_corr = set()
      corr_matrix = X.corr()
      for i in range(len(corr_matrix.columns)):
        for j in range(i):
          if abs(corr_matrix.iloc[i,j])>= self.corr_t:
            colname = corr_matrix.columns[i]
            col_corr.add(colname)
      return col_corr

    def getValidationSet(self):
        return self.x_valid


# In[2]:


# Importing necessary libraries
import pandas as pd


# In[3]:


train = pd.read_csv("./train.csv")
valid = pd.read_csv("./valid.csv")

# drop label_2, label_3 and label_4
dropping_labels = ["label_2","label_3", "label_4"]
train.drop(dropping_labels, axis=1, inplace= True)
valid.drop(dropping_labels, axis=1, inplace= True)


# In[4]:


# check whether any missing values in the train set
train.columns[train.isnull().any()]


# In[5]:


# splitting features and the label
x_train = train.drop(["label_1"], axis=1)
y_train = train["label_1"]
x_valid = valid.drop(["label_1"], axis=1)
y_valid = valid["label_1"]


# In[ ]:


# Create an instance of MyModel
my_model = MyModel()

# Fit the model to the training data
my_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = my_model.predict(x_valid)

# Print the accuracy of the model
accuracy = (y_pred == y_valid).mean()
print(f"Accuracy: {accuracy}")

# Example of using RandomizedSearchCV to tune hyperparameters
param_dist = {
    'C': uniform(0.1, 10.0),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'n_components': [0.97, 0.98, 0.99],
    'variance_t': [0.002, 0.003],
    'corr_t': [0.85,0.9]
}

random_search = RandomizedSearchCV(
    estimator=my_model,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=5,  # Number of cross-validation folds
    verbose=2,
    random_state=42,  # Set a random seed for reproducibility
    n_jobs=-1  # Use all available CPU cores for parallel computation
)
random_search.fit(x_train, y_train)

print("Best hyperparameters found by RandomizedSearchCV:")
print(random_search.best_params_)


# In[ ]:




