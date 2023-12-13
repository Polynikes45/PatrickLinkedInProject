#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Patrick Bockey
# ### 12 December 2023

# ***

# #### Q1

# #### Read in the data, call the dataframe "s"  and check the dimensions of the dataframe
# 

# In[103]:


import pandas as pd
s = pd.read_csv("social_media_usage.csv")
print(s.shape)
s.head(10)


# In[96]:


##Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

##Filter Warnings
import warnings
warnings.filterwarnings("ignore")


# ***

# #### Q2

# * Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. 
#     + If it is, make the value of x = 1, otherwise make it 0. Return x. 
# * Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[70]:


def clean_sm(x):
    x=np.where((x==1),1, 0)
    return x

import numpy as np
## create a toy dataframe
d ={
    'x': ['A', 'B', 'C'],
    'y': [1.5, 400, 1]
}
df = pd.DataFrame(d)
print(df)

clean_sm(df)


# ***

# #### Q3

# * Create a new dataframe called "ss". 
# * The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn
# * Use the following features: 
#     + income (ordered numeric from 1 to 9, above 9 considered missing)
#     + education (ordered numeric from 1 to 8, above 8 considered missing)
#     + parent (binary)
#     + married (binary)
#     + female (binary)
#     + age (numeric, above 98 considered missing)
# * Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.
# 
# 

# In[71]:


ss = pd.DataFrame({
    "sm_li": s["web1h"].apply(clean_sm),
    "income": np.where(s["income"]>9, np.nan, s["income"]),
    "education": np.where(s['educ2']>8, np.nan, s["educ2"]),
    "parent": np.where(s['par']==1, 1, 0),
    "married": np.where(s['marital'] ==1, 1, 0),
    "female": np.where(s['gender']==2,1, 0),
    "age": np.where(s['age'] >97, np.nan, s["age"])})

ss = ss.dropna()
ss.head(10)

#Week 5 supplemental material


# * **Target**: sm_li (LinkedIn User)
#     + Is a LinkedIn user (=1)
#     + Is not a LinkedIn user (=0)
# * **Features**:
#     + income (1 'Less than 10,000' -> 9 'Greater than 150,000')
#     + education (1 'Less than high school' -> 8 'Postgraduate or professional degree')
#     + parent (binary)
#     + married (binary)
#     + female (binary)
#     + age (numeric)

# In[72]:


#Exploratory Data Analysis
ss.info()
ss.describe()

# Assuming 'df' is your DataFrame
plt.figure(figsize=(15, 10))
 
# Using Seaborn to create a heatmap
sns.heatmap(ss.corr(), annot=True, fmt='.2f', linewidths=2)
 
plt.title('Correlation Heatmap')
plt.show()


# ***

# #### Q4

# #### Create a target vector (y) and feature set (X)

# In[73]:


y = ss["sm_li"]
x = ss.drop("sm_li", axis=1)

x.head()


# ***

# #### Q5

# #### Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[74]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                random_state=104,  
                                                test_size=0.20, 
                                                stratify=y,
                                                shuffle=True) 

print ("TRAINING SET")
print("xtrain.shape: ", xtrain.shape)
print("ytrain.shape: ", ytrain.shape)

print("TESTING SET")
print("xtest.shape: ", xtest.shape)
print("ytest.shape: ", ytest.shape)


# **xtrain**: this object contains 80% (1008 rows, 6 columns) of the x features that will be used to build and train the model
# 
# **xtest**: This object contains 20% (252 rows, 6 columns) of the x features that will be used to test the output of the model on unseen data
# 
# **ytrain**: this object contains 80% (1008 rows, 1 column) of the y, target that will be used to build and train the model
# 
# **ytest**: This object contains 20% (252 rows, 1 column) of the y, target that will be used to test the output of the model on unseen data to evaluate performance

# ***

# #### Q6

# #### Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.
# 

# In[75]:


Lr1 = LogisticRegression(class_weight='balanced')


# In[76]:


Lr1.fit(xtrain, ytrain)


# ***

# #### Q7
# #### Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.
# 
# #### Q8
# ####  Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents
# 

# In[77]:


y_pred = Lr1.predict(xtest) 
accuracy_score(ytest, y_pred)


# #### The model has 68% accuracy

# In[78]:


pd.DataFrame(confusion_matrix(ytest, y_pred),
             columns=["Predicted negative", "Predicted positive"],
             index=["Actual negative", "Actual positive"]).style.background_gradient(cmap="PiYG")


# **Predicted negative/Actual Negative**: The Model predicted that the user would not be a LinkedIn user, and was right
# 
# **Predicted negative/Actual Positive**: The Model predicted that the user would not be a LinkedIn user, and was wrong
# 
# **Predicted positive/Actual Negative**: The Model predicted that the user would be a LinkedIn user, and was wrong
# 
# **Predicted positive/Actual Positive**: The Model predicted that the user would be a LinkedIn user, and was right

# ***

# #### Q9
# 

# #### Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.
# 

# In[79]:


## Positive recall: TP/(TP+FN)
57/(57+27)


# **Recall** is how the model does at avoiding false negatives. This metric is important when missclassifying an instance as negative is severe, such as when detecting Fraud. If missclassified, the fraud would be missed.
# 

# In[80]:


## Positive precision: TP/(TP+FP)
57/(57+53)


# **Precision** is the positive predictive value of the model. It is useful when the cost of false positives, or missclassificaiton is high

# In[81]:


## Positive F1 Score: 2 * (precision * recall) / (precision + recall)
2* (.5181*.6786)/(.5181+.6786)


# **F1 Score** provides a way to combine both precision and recall into a single measure that captures both metrics. It is a good balanced way to evaluate a model. 

# In[82]:


# Get other metrics with classification_report
print(classification_report(ytest, y_pred))


# ***

# #### Q10

# * **Use the model to make predictions**
# 
#     + What is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent, who is married, female, and 42 years old uses LinkedIn? 
#     
#     + How does the probability change if another person is 82 years old, but otherwise the same?

# In[97]:


# New data for features: income, education, parent, married, female, age
person1 = [8, 7, 0, 1, 1, 42]
person2 = [8, 7, 0, 1, 1, 82]

# Predict class, given input features
predicted_class1 = Lr1.predict([person1])
predicted_class2 = Lr1.predict([person2])

# Generate probability of positive class (=1)
probs1 = Lr1.predict_proba([person1])
probs2 = Lr1.predict_proba([person2])


# In[92]:


# Print predicted class and probability of the 42 year old
print(f"Predicted class: {predicted_class1[0]}") # 0=not a LinkedIn User, 1=is a LinkedIn User
print(f"Probability that the 42 year old is a LinkedIn User: {probs1[0][1]}")


# In[95]:


# Print predicted class and probability of the 82 year old
print(f"Predicted class: {predicted_class2[0]}") # 0=not a LinkedIn User, 1=is a LinkedIn User
print(f"Probability that that the 82 year old is a LinkedIn User: {probs2[0][1]}")


# ***
git remote add origin https://github.com/Polynikes45/PatrickLinkedInProject.git
  git branch -M main
  git push -u origin main