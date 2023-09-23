#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


# In[12]:


def append_to_file(text):
    with open("outputs_optimized_1.txt", "a") as file:
        # Write content to the file
        file.write(f"{text}\n")


# In[ ]:


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np

all_labels = ["label_1","label_2","label_3", "label_4"]
for label in all_labels:
    droping_labels = all_labels.copy()
    droping_labels.remove(label)

    train = pd.read_csv("./train.csv")
    valid = pd.read_csv("./valid.csv")

    train.drop(droping_labels, axis=1, inplace=True)
    valid.drop(droping_labels, axis=1, inplace=True)

    if(len(train.columns[train.isnull().any()])>0):
        print(f"{label} has missing values in train set")
        train.dropna(inplace=True)

    if(len(valid.columns[train.isnull().any()])>0):
        print(f"{label} has missing values in valid set")
        valid.dropna(inplace=True)

    # splitting features and the label
    x_train = train.drop([label], axis=1)
    y_train = train[label]
    x_valid = valid.drop([label], axis=1)
    y_valid = valid[label]
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)  # You can adjust the sampling strategy

    # Fit and transform the dataset
    rx_train, ry_train = smote.fit_resample(x_train, y_train)

    scaler = StandardScaler()

    # fit the scaler
    sx_train = pd.DataFrame(scaler.fit_transform(rx_train), columns=rx_train.columns)
    sx_valid = pd.DataFrame(scaler.transform(x_valid), columns=x_valid.columns)
    
    for n_comp in [0.95, 0.96, 0.98, 0.99, None]:
        
        if n_comp is not None:
            pca = PCA(n_components= n_comp)

            psx_train = pca.fit_transform(sx_train)
            psx_valid = pca.transform(sx_valid)
            
            new_len = len(psx_train[0])
            
            psx_train = pd.DataFrame(psx_train, columns=[f"new_label{i}" for i in range(1, len(psx_train[0])+1)])
            psx_valid = pd.DataFrame(psx_valid, columns=[f"new_label{i}" for i in range(1, len(psx_valid[0])+1)])
        else:
            psx_train = sx_train
            psx_valid = sx_valid

        # Create an instance of MyModel
#         init_model = SVC()

#         # Fit the model to the training data
#         init_model.fit(x_train, y_train)

#         # Make predictions on the test data
#         y_pred = init_model.predict(x_valid)

#         # Print the accuracy of the model
#         accuracy = (y_pred == y_valid).mean()
#         print(f"Accuracy for {label} with n_comp {n_comp}: {accuracy}")
#         append_to_file(f"Initial accuracy for {label} with n_comp {n_comp}: {accuracy}")

        # Example of using RandomizedSearchCV to tune hyperparameters
        param_dist = {
            'C': uniform(0.1, 100.0),
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': uniform(0.001, 0.1)
        }
        
        svc = SVC()

        random_search = RandomizedSearchCV(
            estimator=svc,
            param_distributions=param_dist,
            n_iter=20,  # Number of random combinations to try
            cv=5,  # Number of cross-validation folds
            verbose=2,
            random_state=42,  # Set a random seed for reproducibility
            n_jobs=-1  # Use all available CPU cores for parallel computation
        )
        
        full_x = pd.concat([psx_train,psx_valid], axis = 0)
        full_y = pd.concat([ry_train, y_valid], axis = 0)
        
        random_search.fit(full_x, full_y)

        print(f"Best hyperparameters found by RandomizedSearchCV for label {label} with n_comp {n_comp}:")
        print(random_search.best_params_)
        append_to_file(f"Best params for {label} with n_comp {n_comp}: {random_search.best_params_}")

        print(f"Best Score: for label {label} with n_comp {n_comp}", random_search.best_score_)
        append_to_file(f"Best score for {label} with n_comp {n_comp}: {random_search.best_score_}")

        # Perform cross-validation to evaluate the model with the best hyperparameters
    #     cross_val_scores = cross_val_score(random_search, X, y, cv=5, n_jobs=-1)

        # Print cross-validation scores
    #     print("Cross-Validation Scores:", cross_val_scores)
    #     append_to_file(f"Cross-Validation Scores for {label} : {cross_val_scores} \n")
    #     print("Mean CV Score:", np.mean(cross_val_scores))
    #     append_to_file(f"Mean CV Score for {label} : {np.mean(cross_val_scores)}")


# In[ ]:




