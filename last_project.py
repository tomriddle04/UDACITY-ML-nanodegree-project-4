#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


cd C:\\Users\\Abhishek\\Downloads


# #### Getting Started:
# In this project, We will evaluate the performence and predictive power of a model that has been 
# trained and tested on data collected from Boston city.A model trained on this data that is seen as
# a good fit then it is used to make certain prediction about home .This model will very valueable
# for someone like real estate agent.
# 
# - The features 'LS','MEDV','LSTAT','PTRATIO' are essential.The remaining non-relevant features have
#   been excluded

# In[3]:


import pandas as pd
import numpy as np
import visuals  as vs

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from sklearn.cross_validation new name is sklearn.model_selection
data = pd.read_csv('housing.csv')
#data
price=data['MEDV']
feature=data.drop('MEDV',axis=1)
print(price)
print(feature)


# In[4]:


#DISTRIBUTION OF DATA
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(data['RM'],bins=15)
plt.title('Average number of room distribution')
plt.xlabel("RM")
plt.ylabel('frequency')
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(data['LSTAT'],bins=15)
plt.title('Homeowner distribution with low class')
plt.xlabel('LSTAT')
plt.ylabel('frequency')
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(data['PTRATIO'],bins=15)
plt.title('Student to teacher ratio distribution')
plt.xlabel('LSTAT')
plt.ylabel('frequency')
plt.show()


# #### Data Exploration
# 
# In this section of this project, we will make a cursory investigating about Boston housing data and
# provide our obeservation .familiarization ourself with the data through an explorative process is a 
# fundamental process practice to help us better understanding and justify our result.
# 
# Since the main goal of this project is to construct a working model which has the capablity of predicting 
# the value of house .we will need to seperate the dataset into features and the target variable.The
# features 'RM','LSTAT',and 'PTRATIO', give us quantitative information about each data point .The 
# target variable 'MEDV' will be the variable we seek to predict .These are stored in features and 
# prices respectively.

# In[5]:


#DATA EXPLORATION
min_price=np.min(price)
max_price=np.max(price)
mean_price=np.mean(price)
median_price=np.median(price)
std_price=np.std(price)
print("Minimum price: ${}".format(min_price))
print("Maximum price: ${}".format(max_price))
print('Mean or average price : ${}'.format(mean_price))
print("Meadian price: ${}".format(median_price))
print("standard deviation of price : ${}".format(std_price))

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(data['RM'],data['MEDV'])
plt.title('Average Selling price and average no of Rooms')
plt.xlabel('RM')
plt.ylabel('Price')
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(data['LSTAT'],data['MEDV'])
plt.title('Average selling price VS percentage of low class Homeowners')
plt.xlabel('Lstat')
plt.ylabel('Price')
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(data['PTRATIO'],data['MEDV'])
plt.title('Average selling price VS Ratio of Student to Teacher')
plt.xlabel('PT_ratio')
plt.ylabel('Price')
plt.show()


# #### Q.1 Feature Observation
# Ans 1.RM 
# - For a higher RM,one would expect to observe a high MEDV
# - this is because more rooms would imply more space.hence its cost is more
# 2.LSTAT
# - for a higher LSTAT one would expect to observe a lower MEDV
# 3.PTRATIO
# - For a higher PTRATIO,one would expect to observe a lower MEDV
# 

# In[6]:


#Define a Performance metric
from sklearn.metrics import r2_score
def perfomence_metric(y_true,y_predict):
    score=r2_score(y_true,y_predict)
    return(score)

score=perfomence_metric([3,-.5,2,7,4.2],[2.5,0.0,2.1,7.8,5.3])
print('Model has a cofficient of determination R^2 of {:.3f}'.format(score))


# #### Q.2 Goodnees of Fit
# Answer:
# - R^2=92.3%
# - This implies 92.3% of variation is explained by the target variable and it seems to be high.The 
# model has a fairly strong correlation and has successfully capture the variation of the target variable
# - we have only five points here,and it may be hard to draw conclusion that is statistically significant 
# .Some more data point may or perhaps helped to improve the model

# In[7]:


#implementation of shuffle and split data
X_train,X_test,y_train,y_test=train_test_split(feature,price,test_size=None,random_state=0)
print("Training and Testing split was successful ")


# #### Q.3 Training and Testing
#  What is the benefit to spliting a dataset into some ratio of training and testing subsets for a
# learning algorithm
# Answer:
# - If we are building a model and checking the performance of the model on the same data our model 
# will lead to overfitting and it will perform worst on the unseen data.That's why We use the training data
# to train the model.and then We use testing data to checking the performance of the model.
# it's important that two sets are independent from each other or result will be baised.

# In[8]:


#Analyzing Model perfomence
vs.ModelLearning(feature,price)


# #### Q.4 Learning the data
# - Choose one of the graphs above and state the maximum depth for the model
# - what happens to the score of the training curve as more training points are added?What about the
# testing curve
# - Would having more training points benefits the model
# 
# Answer:
# 
# 1. max_depth=1(high bias scenario)
#     - we can see how the testing score increases with the number of obsevation.
#     - However,the testing score only increases to approximately 0.4,a low score.
#     this indicates how the model does not generalizes well for new unseen data.
#     - moreover,the training score(red line) decreases with the number of observation .Also the 
#     training score decreases to a very low score of apprex 0.4.This indicates how the model does not 
#     seem to fit the data well.
#     - Thus,we can say this model is facing a high bias problem.consequently,having more trainig points
#     would not benefit the model as the model is underfitting the dataset.instead,one should inrcrese
#     the model complexity to better fit the dataset.morever,the testing score has reached a plateau 
#     suggesting the model may not improve from adding more training points .
#     
# 2.max_depth=3(Ideal Scenario)
# - Testing score(green line) increses with trainig points.
#     - Reaches~0.8
#     - high score 
#     - Generalize well
# -Training score(redline)decreases slightly with training points.
#     - Reachers~0.8
#     - High score
#     - Fit dataset well
# - There seems to be no high bias or high variance problem
#     - Model fits and generalizes well.
#     - Ideal
#     - more training points should help it become an even more ideal model.
#     
# 3.max_depth=10(High variance scenario)
# - Testing score(green line)increases with training points
#     - Reaches~0.7
#     - Not so high score
#     - Does not generalize well
# - Trainig score(red line)barely decreases with training points
#     - At~1.0
#     - Almost perfect score
#     - Overfitting dataset
# - There seems to be a high variance problem
#     - overfitting
#     - More training points might help
#     - This is getting close to the ideal scenario.

# In[9]:


vs.ModelComplexity(X_train,y_train)


# #### Q.5 Bias-variance tradeoff
# - When the model is trained with a maximum depth of 1,does the model suffer from high bias or from 
# high variance?
# - How about when the model is trained with a maximum depth of 10 ?What visual cues in the graph 
# justify your conclusion?
# 
# Answer:
# - It is easy to identify wheather the model is suffering from a high bias or a high variance.
#     - High variance models have a gap between the training andvalidation scores.
#     - This is because it is able to fit the model well but unable to generalize well resulting 
#     in a high training score but low validation score.
#     - High bias models have a small or no gap between the training and validation score.
#     - This is because it is unable to fit the model well and unable to generalize well resulting
#     in both score converging to a similar low score.
# - Maximum depth of 1:High Bias
#     - Both training and testing score is low.
#     - There is barely gap betwwen training and testing score.
#     - This indicate that model is not fitting the dataet well and not generalize well .Hence
#     model is suffering from high bias.
# - maximum depth of 10:High variance
#     - Both training and testing score is low.
#     - There is substantial gap betwwen training and testing scores.
#     - This indicate that model is  fitting the dataet well but not generalizing  well .Hence
#     model is suffering from high variance.
# 

# #### Q.6 Best-Guess optimal Model
# - Which maximum depth do you think results in a model that best generalizes to unseen data?
# - What intuition lead you to this answer?
# 
# Answer:
# - The maximum depth of 5.
# - The training score seems to plateau here,indicating the highest possible score for the model's
# ability to generalize to unseen data.
# - Gap between the training score and testing score does not seem to be substantial too,indicating that
# the model may not suffering from a high variance scenario.
#     

# #### Q.7 Grid Search
# - What is the grid search technique?
# - How it can be applied to optimize a learning algorithm
# 
# Answer:
#     
# For a family of models with different values of parameters,grid search allows to select the best 
# possible model for prediction by allowing us to specify which of those parameters we want to change,
# their corresponding rangues and the function score to be optimized.it then gives us a combination
# of values for those parameters that optimize the scoring function by searching each of those models 
# iteratively
# 
# grid search performs hyperparameter optimization by selecting a grid of values,evaluating them and 
# returning the result.This parameter sweep functionality of grid search can optimize a learning algorithm
# 

# #### Q.8 Cross-validation
# - What is the k-fold cross-validation training technique
# - What benefit does this technique provide for grid search when optimizing a model
# 
# Answer:
# 
# - K-fold cross-validation summary
#     - dataset is split into k 'folds' of equal size
#     - Each fold acts as a testing set 1 time, and acts as the training set k-1 times.
#     - Average testing performance is used as the estimate of out-of-sample performance.
#     - Also known as cross-validated performance.
#     
# - Benefits of k-fold cross-validation
#     - More reliable estimate of out of sample performance than train/test split.
#     - Reduce the variance of a single trail of a train/test split.
#     - Hence,the benefits of k-fold cross-validation,we are able to use the average testing accuracy
#     as a benchmark to decide which is the most optimal set of parameter for the learning algorithm.
#     - If we dont using of k-fold cross-validation set and we run Grid-search, we would have different
#     set of optimal parameters due to the fact that without a cross-validation set,the estimate
#     of out-of-sample performance would have a high variance
#     - in,summary,without k-fold cross-validation the risk is higher that Grid search will select
#     hyper-parameter value combinations that perform very well on a specific train-test split but 
#     poorly otherwise.
# - Limitation of k-fold cross-validation
#     - it does not work well when dataset is not uniformly distributed(e.g. sorted data)

# In[10]:


from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
def fit_model(X,y):
    cv_sets=ShuffleSplit(n_splits=10, test_size='default', random_state=None)
    #cv_sets=ShuffleSplit(X.shape[0],test_size='default',random_state=None)
    regressor= DecisionTreeRegressor(random_state=0)
    params={'max_depth':list(range(1,11))}
    scoring_fnc=make_scorer(perfomence_metric)
    grid=GridSearchCV(regressor,params,cv=cv_sets,scoring=scoring_fnc)
    grid=grid.fit(X,y)
    return(grid.best_estimator_)  


# #### Q.9 Optimal Model
# - What maximum depth does the optimal model have? How does this result compare to your guess in
# question 6?
# 

# In[11]:


reg=fit_model(X_train,y_train)
print(reg.get_params()['max_depth'])


# Answer:
# 
# - The optimal model has a maximum depth of 5.The max_depth is the same as my guess in question 6

# #### Q.10 Predicting selling prices
# imagine that you were a real estate agent in the Boston area looking to use this model to help prices 
# homes owned by your clients that they wish to sell. You have collected the following information from 
# three of your clients:
#     
# #### Feature-------------------- Client 1------------------Client 2------------------ Client 3
# Total no. of rooms in home        5 rooms                   4 rooms                     8 rooms
# 
# Neighbourhood poverty level         17%                        32%                        3%
# 
# Student/Teacher ratio of nearby     15:1                       22:1                      12:1
# school

# In[12]:


client_data=[[5,17,15],
            [4,32,22],
            [8,3,12]]
for i,price in enumerate(reg.predict(client_data)):
    print('predicted selling price for client {}  home : ${:.2f}'.format(i+1,price))


# In[ ]:





# In[13]:


print(feature)


# In[14]:


price=data['MEDV']
(price)


# In[15]:


vs.PredictTrials(feature,price,fit_model,client_data)


# #### Q.11 Applicability
# - In a few sentances,discuss whether the constructed model should or should not be used in a real 
# world setting
# - How relevant today is data that was collected from 1978?How important is inflation?
# - Is the model robust enough to make consistent predictions?
# - Would data collected in an urban city like Boston be applicable in a rural city?
# - Is it fair to judge the price of an individual home based on the characteristics of the entire 
# neighborhood?
# 
# Answer:
#     
# - Data collected from a rural city may not be applicable as the demographics would change and other 
# features may be better able to fit the dataset instead of a model with features that was learned using
# urban data.
# - The learning algorithm learned from a very old dataset that may not be relevent because demographics
# have changed a lot since 1978.
# - There are only 3 features currently,there are more features that can be included such as crime rates,
# nearby to city,public transport access and more.

# In[ ]:





# In[ ]:





# In[ ]:




