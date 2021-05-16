#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Run LR as a baseline
#Try variants (alternate feature representation, etc)

#Try nonlinear models (trees, random forest, gxboost)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import mean_squared_error

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

data = pd.read_csv('submission.csv')

data["Sector"] = data['Sector'].astype('category')
data = pd.get_dummies(data, columns=['Sector'])

split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(data, data['Class']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
                               
y_train = strat_train_set["2015 PRICE VAR [%]"]
y_test = strat_test_set["2015 PRICE VAR [%]"]

x_train = strat_train_set.drop(columns=["2015 PRICE VAR [%]", "Unnamed: 0", "Class"])
x_test = strat_test_set.drop(columns=["2015 PRICE VAR [%]", "Unnamed: 0", "Class"])

# #Baseline Model:
# from sklearn.linear_model import LinearRegression
# model1 = LinearRegression().fit(x_train,y_train)
#
# pred1 = model1.predict(x_test)
# error1 = mean_squared_error(y_test,pred1)
# importance1 = model1.coef_
#
# plt.bar([x for x in range(len(importance1))], importance1)
# plt.show()

# #Regularized Regressions: Ridge
from sklearn.linear_model import Ridge
# ridge_model = Ridge(alpha = 1).fit(x_train, y_train)
# ridge_pred = ridge_model.predict(x_test)
# ridge_error = mean_squared_error(y_test,ridge_pred)


#Regularized Regressions: Ridge Cross-Validation
from sklearn.model_selection import GridSearchCV
#
# lambda_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3]
# parameters = {'alpha' : lambda_list}
# grid_search = GridSearchCV(Ridge(), param_grid=parameters, cv = 5, scoring = "neg_mean_squared_error")
# grid_search.fit(x_train,y_train)
#
# best_ridge = grid_search.best_estimator_
# best_ridge_pred = best_ridge.predict(x_test)
# best_ridge_error = mean_squared_error(y_test,best_ridge_pred)
# importance2 = best_ridge.coef_
#
# plt.bar([x for x in range(len(importance2))], importance2)
# plt.show()
#
# #Fine-Grained Cross-Validation
# lambda_list2 = [5, 7, 10, 12, 15, 20, 50, 75, 95]
# parameters2 = {'alpha' : lambda_list2}
# grid_search2 = GridSearchCV(Ridge(), param_grid=parameters2, cv = 5, scoring = "neg_mean_squared_error")
# grid_search2.fit(x_train,y_train)
#
# best_ridge2 = grid_search2.best_estimator_
# best_ridge_pred2 = best_ridge2.predict(x_test)
# best_ridge_error2 = mean_squared_error(y_test,best_ridge_pred2)
#
# Regularized Regression: Lasso
#%%
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 0.1).fit(x_train, y_train)
lasso_pred = lasso_model.predict(x_test)
lasso_error = mean_squared_error(y_test,lasso_pred)

#Regularized Regressions: Lasso Cross-Validation

from sklearn.model_selection import GridSearchCV

lambda_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3]
parameters = {'alpha' : lambda_list}
grid_search = GridSearchCV(Lasso(), param_grid=parameters, cv = 5, scoring = "neg_mean_squared_error")
grid_search.fit(x_train,y_train)

best_lasso = grid_search.best_estimator_
best_lasso_pred = best_lasso.predict(x_test)
best_lasso_error = mean_squared_error(y_test,best_lasso_pred)
importance3 = best_lasso.coef_

feature_list = x_train.columns.values
imp_feature_index=[]
imp_feature_list=[]
imp_feature_values=[]
for i in range(0,20):
    imp_feature_values.append(-np.sort(-importance3)[i])
    imp_feature_index.append(np.where(importance3 == imp_feature_values[i])[0][0])
    imp_feature_list.append(feature_list[imp_feature_index[i]])

plt.figure(figsize=(10,6))
plt.title("Most important 20 features from Lasso")

plt.barh(imp_feature_list,imp_feature_values, color="steelblue",edgecolor="black", align="center")
plt.savefig("lasso_feature_imp")
plt.show()
#%%


# plt.bar([x for x in range(len(importance3))], importance3)
# plt.show()
#
# # #Regularized Regression: Elastic Net
# # from sklearn.linear_model import ElasticNet
# # EN_model = ElasticNet(alpha = 0.1, l1_ratio = 0.5).fit(x_train, y_train)
# # EN_pred = EN_model.predict(x_test)
# # EN_error = mean_squared_error(y_test,EN_pred)
#
# #Regularized Regression: Elastic Net Cross-Validation
# lambda_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3]
# parametersGrid = {'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3],
#                   'l1_ratio': np.arange(0.0, 1.1, 0.1)}
# grid_search = GridSearchCV(ElasticNet(), param_grid=parametersGrid, cv = 5, scoring = "neg_mean_squared_error")
# grid_search.fit(x_train,y_train)
#
# best_EN = grid_search.best_estimator_
# best_EN_pred = best_EN.predict(x_test)
# best_EN_error = mean_squared_error(y_test,best_EN_pred)
#
#
# # #Decision Tree Regressor
# # from sklearn.tree import DecisionTreeRegressor
# # tree_model = DecisionTreeRegressor(max_depth=5).fit(x_train,y_train)
# # tree_pred = tree_model.predict(x_test)
# # tree_error = mean_squared_error(y_test,tree_pred)
#
# #Decision Tree Regressor: Cross Validation
# parametersGrid = {'criterion': ["mse"],
#                   'min_samples_split': [10, 20],
#                   'max_depth' : [3, 5],
#                   'min_samples_leaf' : [20, 40, 100],
#                   'max_leaf_nodes' : [5, 20]
#                   }
# grid_search_tree = GridSearchCV(DecisionTreeRegressor(), param_grid=parametersGrid, cv = 5, scoring = "neg_mean_squared_error")
# grid_search_tree.fit(x_train,y_train)
#
# best_tree = grid_search_tree.best_estimator_
# best_tree_pred = best_tree.predict(x_test)
# best_tree_error = mean_squared_error(y_test,best_tree_pred)
#
#
#Random Forest
from sklearn.ensemble import RandomForestRegressor
# rf_model = RandomForestRegressor(max_depth=2).fit(x_train,y_train)
# rf_pred = rf_model.predict(x_test)
# importance4=rf_model.feature_importances_
#
# feature_list = x_train.columns.values
# imp_feature_index=[]
# imp_feature_list=[]
# imp_feature_values=[]
# for i in range(0,20):
#     imp_feature_values.append(-np.sort(-importance4)[i])
#     imp_feature_index.append(np.where(importance4 == imp_feature_values[i])[0][0])
#     imp_feature_list.append(feature_list[imp_feature_index[i]])
#
#
# plt.figure(figsize=(10,6))
# plt.title("Most important 20 features from Random Forest")
#
# plt.barh(imp_feature_list,imp_feature_values, color="steelblue",edgecolor="black", align="center")
# plt.show()


# #Random Forest Regressor: Cross Validation
# parametersGrid = {'bootstrap': ['True','False'],
#                   'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#                   'max_features' : ['auto', 'sqrt'],
#                   'min_samples_leaf' : [1, 2, 4],
#                   'min_samples_split': [2, 5, 10],
#                   'n_estimators' : [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
#                   }
#
# grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid=parametersGrid, cv = 5, scoring = "neg_mean_squared_error")
# grid_search_rf.fit(x_train,y_train)
# importance4=grid_search_rf.feature_importances_
#
# feature_list = x_train.columns.values
# imp_feature_index=[]
# imp_feature_list=[]
# imp_feature_values=[]
# for i in range(0,20):
#     imp_feature_values.append(-np.sort(-importance4)[i])
#     imp_feature_index.append(np.where(importance4 == imp_feature_values[i])[0][0])
#     imp_feature_list.append(feature_list[imp_feature_index[i]])
#
#
# plt.figure(figsize=(10,6))
# plt.title("Most important 20 features from Random Forest")
#
# plt.barh(imp_feature_list,imp_feature_values, color="steelblue",edgecolor="black", align="center")
# plt.show()


# #Random Forest Regressor: Randomized Cross Validation
# from sklearn.model_selection import RandomizedSearchCV
# parametersGrid = {'bootstrap': ['True','False'],
#                   'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#                   'max_features' : ['auto', 'sqrt'],
#                   'min_samples_leaf' : [1, 2, 4],
#                   'min_samples_split': [2, 5, 10],
#                   'n_estimators' : [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
#                   }
#
# grid_search_rf = RandomizedSearchCV(RandomForestRegressor(), param_distributions=parametersGrid, n_iter = 50, cv = 5, scoring = "neg_mean_squared_error", verbose=10)
# grid_search_rf.fit(x_train,y_train)
#
#
# best_rf = grid_search_rf.best_estimator_
# best_rf_pred = grid_search_rf.predict(x_test)
# best_rf_error = mean_squared_error(y_test,best_rf_pred)
#
# importance4 = grid_search_rf.feature_importances_
#
# feature_list = x_train.columns.values
# imp_feature_index=[]
# imp_feature_list=[]
# imp_feature_values=[]
# for i in range(0,20):
#     imp_feature_values.append(-np.sort(-importance4)[i])
#     imp_feature_index.append(np.where(importance4 == imp_feature_values[i])[0][0])
#     imp_feature_list.append(feature_list[imp_feature_index[i]])
#
#
# plt.figure(figsize=(10,6))
# plt.title("Most important 20 features from Random Forest")
#
# plt.barh(imp_feature_list,imp_feature_values, color="steelblue",edgecolor="black", align="center")
# plt.show()
#




 #best_rf_error
# #Out[39]: 913.5194956827593
#
best_rf=RandomForestRegressor(bootstrap='False', max_depth=90, min_samples_leaf=2,min_samples_split=10, n_estimators=1200,verbose=10).fit(x_train,y_train)
#%%
importance4 = best_rf.feature_importances_


feature_list = x_train.columns.values
imp_feature_index=[]
imp_feature_list=[]
imp_feature_values=[]
for i in range(0,20):
    imp_feature_values.append(-np.sort(-importance4)[i])
    imp_feature_index.append(np.where(importance4 == imp_feature_values[i])[0][0])
    imp_feature_list.append(feature_list[imp_feature_index[i]])

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.figure(figsize=(8,6))
plt.title("Most important 20 features from Random Forest")

plt.barh(imp_feature_list,imp_feature_values, color="steelblue",edgecolor="black", align="center")
plt.savefig("rf_feature_imp")
plt.show()





















