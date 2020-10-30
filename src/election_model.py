# Imports
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re
from textblob import TextBlob
from datetime import tzinfo, timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import ensemble
from scipy.stats import pearsonr


def load_data(csv_file):
  """
  Loads the cleaned tweets csv file
  
  Parameters
  ----------
  csv_file: str
    The file path of the .csv file to be loaded
    
  Outputs
  -------
  data: Pandas DataFrame
    The cleaned dataframe, ready to work on  
  """
  
  data = pd.read_csv(csv_file)
  return data


def baseline_polarity_model(data):
  """
  Runs the baseline Gradient Boosting Regressor model
  with Polarity as the target
  
  Parameters
  ----------
  data: Pandas DataFrame
    The DataFrame to use
    
  Outputs
  -------
  Returns nothing
  
  Saves a plot of the Feature Importance and
  Permutation Importance
  """
  
  X = data[['USA', 'tweet_length', 'hour', 'friends_count', 'followers_count']]
  y = data['polarity']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split data
  reg = ensemble.GradientBoostingRegressor()
  reg.fit(X_train, y_train)
  mse = mean_squared_error(y_test, reg.predict(X_test))
  mae = mean_absolute_error(y_test, reg.predict(X_test))
  print(f'The mean square error is {mse} and the mean absolute error is {mae}')
  feature_importance = reg.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  fig = plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, np.array(X.columns)[sorted_idx])
  plt.title('Feature Importance: Polarity')

  result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
  sorted_idx = result.importances_mean.argsort()
  plt.subplot(1, 2, 2)
  plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(X.columns)[sorted_idx])
  plt.title("Permutation Importance: Polarity (test set)")
  fig.tight_layout()
  plt.show()

def final_polarity_model(data):
  """
  Runs the enhanced Gradient Boosting Regressor model
  with Polarity as the target
  
  Parameters
  ----------
  data: Pandas DataFrame
    The DataFrame to use
    
  Outputs
  -------
  Returns nothing
  
  Saves a plot of the Feature Importance and
  Permutation Importance
  """
    
  X = data[['USA', 'tweet_length', 'hour', 'friends_count', 'followers_count', 'Biden', 'Trump', 'red', 'blue']]
  y = data['polarity']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split data
  reg = ensemble.GradientBoostingRegressor()
  reg.fit(X_train, y_train)
  mse = mean_squared_error(y_test, reg.predict(X_test))
  mae = mean_absolute_error(y_test, reg.predict(X_test))
  print(f'The mean square error is {mse} and the mean absolute error is {mae}')
  feature_importance = reg.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  fig = plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, np.array(X.columns)[sorted_idx])
  plt.title('Feature Importance: Polarity')

  result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
  sorted_idx = result.importances_mean.argsort()
  plt.subplot(1, 2, 2)
  plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(X.columns)[sorted_idx])
  plt.title("Permutation Importance: Polarity (test set)")
  fig.tight_layout()
  plt.show()

def baseline_subjectivity_model(data):
  """
  Runs the baseline Gradient Boosting Regressor model
  with Subjectivity as the target
  
  Parameters
  ----------
  data: Pandas DataFrame
    The DataFrame to use
    
  Outputs
  -------
  Returns nothing
  
  Saves a plot of the Feature Importance and
  Permutation Importance
  """
  
  X = data[['USA', 'tweet_length', 'hour', 'friends_count', 'followers_count']]
  y = data['subjectivity']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split data
  reg = ensemble.GradientBoostingRegressor()
  reg.fit(X_train, y_train)
  mse = mean_squared_error(y_test, reg.predict(X_test))
  mae = mean_absolute_error(y_test, reg.predict(X_test))
  print(f'The mean square error is {mse} and the mean absolute error is {mae}')
  feature_importance = reg.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  fig = plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, np.array(X.columns)[sorted_idx])
  plt.title('Feature Importance: Subjectivity')

  result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
  sorted_idx = result.importances_mean.argsort()
  plt.subplot(1, 2, 2)
  plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(X.columns)[sorted_idx])
  plt.title("Permutation Importance: Subjectivity (test set)")
  fig.tight_layout()
  plt.show()

def final_subjectivity_model(data):
  """
  Runs the enhanced Gradient Boosting Regressor model
  with Subjectivity as the target
  
  Parameters
  ----------
  data: Pandas DataFrame
    The DataFrame to use
    
  Outputs
  -------
  Returns nothing
  
  Saves a plot of the Feature Importance and
  Permutation Importance
  """
  
  X = data[['USA', 'tweet_length', 'hour', 'friends_count', 'followers_count', 'Biden', 'Trump', 'red', 'blue']]
  y = data['subjectivity']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split data
  reg = ensemble.GradientBoostingRegressor()
  reg.fit(X_train, y_train)
  mse = mean_squared_error(y_test, reg.predict(X_test))
  mae = mean_absolute_error(y_test, reg.predict(X_test))
  print(f'The mean square error is {mse} and the mean absolute error is {mae}')
  feature_importance = reg.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  fig = plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, np.array(X.columns)[sorted_idx])
  plt.title('Feature Importance: Subjectivity')

  result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
  sorted_idx = result.importances_mean.argsort()
  plt.subplot(1, 2, 2)
  plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(X.columns)[sorted_idx])
  plt.title("Permutation Importance: Subjectivity (test set)")
  fig.tight_layout()
  plt.show()

def final_overall_sentiment_model(data):
  """
  Runs the final Gradient Boosting Regressor model
  with overall sentiment as the target
  
  Parameters
  ----------
  data: Pandas DataFrame
    The DataFrame to use
    
  Outputs
  -------
  Returns nothing
  
  Saves a plot of the Feature Importance and
  Permutation Importance
  """

  X = data[['USA', 'tweet_length', 'hour', 'friends_count', 'followers_count', 'Biden', 'Trump', 'red', 'blue']]
  y = data['overall_sentiment']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split data
  reg = ensemble.GradientBoostingRegressor()
  reg.fit(X_train, y_train)
  mse = mean_squared_error(y_test, reg.predict(X_test))
  mae = mean_absolute_error(y_test, reg.predict(X_test))
  print(f'The mean square error is {mse} and the mean absolute error is {mae}')
  feature_importance = reg.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  fig = plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, np.array(X.columns)[sorted_idx])
  plt.title('Feature Importance: Overall Sentiment')

  result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
  sorted_idx = result.importances_mean.argsort()
  plt.subplot(1, 2, 2)
  plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(X.columns)[sorted_idx])
  plt.title("Permutation Importance: Overall Sentiment (test set)")
  fig.tight_layout()
  plt.show() 

if __name__ == "__main__":
    data = load_data('final_election.csv')
    final_polarity_model(data)
    final_subjectivity_model(data)
    final_overall_sentiment_model(data)
