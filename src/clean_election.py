import pandas as pd
import numpy as np
import json
import re
from textblob import TextBlob
from datetime import tzinfo, timedelta, datetime

pd.options.mode.chained_assignment = None

def load_data(file):
  df = pd.read_csv(csv_file)
  return df

def clean(df):
  df['screen_names'] = [ user['screen_name'] for user in df['user']] # Extract screen names
  df['location'] = [user['location'] for user in df['user']] # Extract locations
  df['name'] = [user['name'] for user in df['user']] # Extract name
  df['friends_count'] = [user['friends_count'] for user in df['user']] # Extract friend count
  df['followers_count'] = [user['followers_count'] for user in df['user']] # Extract follower count
  df['user_description'] = [user['description'] for user in df['user']] # Extract user description
  df['profile_creation_date'] = [user['created_at'] for user in df['user']] # Extract profile creation date
  # Drop unnecessary columns
  df.drop('id_str', axis=1, inplace=True)
  df.drop('geo', axis=1, inplace=True)
  df.drop('coordinates', axis=1, inplace=True)
  df.drop('contributors', axis=1, inplace=True)
  df.drop('withheld_in_countries', axis=1, inplace=True)
  df.drop('truncated', axis=1, inplace=True) # They are all false 
  df['created_at'] = df['created_at'].str.split('+').str[0] # Clean date column
  df['profile_creation_date'] = df['profile_creation_date'].str.split('+').str[0] # Clean date column
  df['profile_creation_date'] = df['profile_creation_date'].str[4:] # Clean date column
  states = {' AL': 'Alabama', ' AK': 'Alaska', ' AZ': 'Arizona', ' AR': 'Arkansas', ' CA': 
          'California', ' CO': 'Colorado', ' CT': 'Connecticut', ' DE': 'Delaware', 
          ' FL': 'Florida', ' GA': 'Georgia', ' HI': 'Hawaii', 
         ' ID': 'Idaho', ' IL': 'Illinois', ' IN': 'Indiana', ' IA': 'Iowa', 
         ' KS': 'Kansas', ' KY': 'Kentucky', ' LA': 'Louisiana', ' ME': 'Maine', 
         ' MD': 'Maryland', ' MA': 'Massachusetts', 
         ' MI': 'Michigan', ' MN': 'Minnesota', ' MS': 'Mississippi', ' MO': 'Missouri', 
         ' MT': 'Montana', ' NE': 'Nebraska', ' NV': 'Nevada', ' NH': 'New Hampshire', 
         ' NJ': 'New Jersey', ' NM': 'New Mexico', ' NY': 'New York', ' NC': 'North Carolina', 
         ' ND': 'North Dakota', ' OH': 'Ohio', ' OK': 'Oklahoma', ' OR' : 'Oregon', 
         ' PA': 'Pennsylvania', ' RI': 'Rhode Island', ' SC': 'South Carolina', 
         ' SD': 'South Dakota', ' TN': 'Tennessee', ' TX': 'Texas', 
         ' UT': 'Utah', ' VA': 'Virginia', ' VT': 'Vermont', ' WA': 'Washington', 
         ' DC': 'DC', ' WV': 'West Virginia', ' WI': 'Wisconsin', ' WY': 'Wyoming', 
         ' PR': 'Puerto Rico'}
  df['location'] = df['location'].str.strip() # Clean whitespaces
  df['location'] = df['location'].str.replace(',', ' ')
  df['location'] = df['location'].str.replace('.', '')
  df['location'] = df['location'].str.replace('/', ' ')
  df['state'] = '' # Create state 1 column
  df['state2'] = '' # Create state 2 column

  # Extract states
  for i, location in enumerate(df['location']):
      if len(location)!= 2:
          for k,v in states.items():
              if k.upper() in location.upper():
                  if len(df['state'][i]) == 0:
                      df['state'][i] = k
                  else:
                      df['state2'][i] = df['state2'][i] + k
              if v.upper() in location.upper():
                  if len(df['state'][i]) == 0:
                      df['state'][i] = k
                  else:
                      df['state2'][i] = df['state2'][i] + k
      else:
          for k,v in states.items():
              if k.strip() in location.upper():
                  df['state'][i] = df['state'][i] + k
  df['state'] = df['state'].str.strip() # Clean whitespaces
  df['state2'] = df['state2'].str.strip() # Clean whitespaces
  # Remove repeats of states
  for i,v in enumerate(df['state']):
    if df['state'][i] == df['state2'][i]:
        df['state2'][i] = ''
  
  cities = {'PHILLY': 'PA', 'PHILADELPHIA': 'PA', 'BOSTON': 'MA', 'NYC': 'NY', 
  'BALTIMORE': 'MD', 'SEATTLE': 'WA', 'PORTLAND': 'OR', 'CALI': 'CA', 'SAN FRANCISCO': 'CA', 
  ' LA ': 'CA', 'LOS ANGELES': 'CA', 'PHOENIX': 'AZ', 'OMAHA': 'NE', 'CHICAGO': 'IL', 
  'DALLAS': 'TX', 'HOUSTON': 'TX', 'SAN ANTONIO': 'TX', 'AUSTIN': 'TX', 'BOISE': 'ID', 
  'HONOLULU': 'HI', 'SLC': 'UT', 'Salt Lake': 'UT', 'ST LOUIS': 'MO', 'ATLANTA': 'GA', 
  'JERSEY': 'NJ', 'DETROIT': 'MI', 'BROOKLYN': 'NY', 'DENVER': 'CO'}
          
          
  for i,location in enumerate(df['location']): # Map cities to state
    if df['state'][i] == '':
        for k,v in cities.items():
            if k in location.upper():
                df['state'][i] = v
  american_terms = ['AMERICA', 'AMERICAN', 'MIDWEST', 'NORTHWEST', 'PACIFIC', 'MERICA', 
  'USA', ' US', ' USA', 'UNITED STATES', 'UNITED STATES OF AMERICA']
  df['USA'] = ''
  # Loop through US terms
  for i, v in enumerate(df['location']):
    if len(df['state'][i]) != 0:
        df['USA'][i] = 1
    elif df['location'][i].upper() in american_terms:
        df['USA'][i] = 1
    else:
        df['USA'][i] = 0 
  # Extract text length
  df['tweet_length'] = df.apply(lambda row: row.display_text_range[1], axis = 1)
  # Add Trump and Biden reference columns
  biden_reference = ['JOE', 'BIDEN']
  trump_reference = ['TRUMP', 'DONALD']
  df['Biden'] = 0
  df['Trump'] = 0
  for i, v in enumerate(df['full_text']):
    for word in biden_reference:
      if word in df['full_text'][i].upper():
        df['Biden'][i] = 1
    for word in trump_reference:
      if word in df['full_text'][i].upper():
        df['Trump'][i] = 1
  # Add retweet column
  word = 'RT @'
  df['retweet'] = 0
  for i, v in enumerate(df['full_text']):
    if word in df['full_text'][i]:
      df['retweet'][i] = 1
  # Add red and blue state columns
  blue = ['CA', 'CO', 'CT', 'DE', 'HI', 'IL', 'ME', 'ML', 'MD', 'MN', 'NV', 'NH', 'NJ', 'NM', 'NY', 'OR', 'RI', 'VT', 'VA', 'WA', 'PR']
  red = ['AL', 'AK', 'AZ', 'AR', 'FL', 'GA', 'ID', 'IN', 'IA', 'KS', 'KY', 'LA', 'MI', 'MS', 'MO', 'MT', 'NE', 'NC', 'ND', 'OH', 'OK', 'PA', 'SC', 'SD', 'TN', 'TX', 'UT', 'WV', 'WI', 'WY']
  df['blue'] = 0
  df['red'] = 0
  for i, v in enumerate(df['state']):
    for abbrev in blue:
        if abbrev in df['state'][i]:
            df['blue'][i] = 1
    for abbrev in red:
        if abbrev in df['state'][i]:
            df['red'][i] = 1
  # Time series
  df['day_created'] = ''
  df['time_created'] = ''
  # Day Created
  df['day_created'] = [row[4:10] for row in df['created_at']]
  # Time Created
  df['time_created'] = [row[11:-1] for row in df['created_at']]
  df['date_time_created'] = [row[4:10] + ' ' + '2020 ' + row[11:-1] for row in df['created_at']]
  df['date_time_created'] = [pd.to_datetime(row, format='%b %d %Y %X') for row in df['date_time_created']]
  df['hour'] = [row.hour for row in df['date_time_created']] # Create hour column
  # Change hours
  df['hour'][0:2] = df['hour'][0:2]*0 # Sept 29
  df['hour'][48537:92918] = df['hour'][48537:92918]+24 # Oct 1
  df['hour'][92918:] = df['hour'][92918:]+48 # Oct 2
  df['hour'][48537:50688] = 24
  df['hour'][92918:94843] = 47
  df['polarity'] = df.apply(lambda row: TextBlob(row.full_text).sentiment.polarity, axis = 1) # Create polarity column
  df['subjectivity'] = df.apply(lambda row: TextBlob(row.full_text).sentiment.subjectivity, axis = 1) # Create subjectivity column
  df['overall_sentiment'] = abs(df['polarity']) + abs(df['subjectivity'])
  df_for_model = df[['USA', 'tweet_length', 'hour', 'friends_count', 'followers_count', 
  'Biden', 'Trump', 'retweet', 'red', 'blue', 'polarity', 'subjectivity', 'overall_sentiment']]
  df_for_model.to_csv('final_election.csv') # Save to csv

if __name__ == "__main__":
    df = load_data('data/subset_data.csv')
    clean(df)