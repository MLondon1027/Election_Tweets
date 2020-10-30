# Election_Tweets

## Project Goal

The goal of this project was to predict sentiment as measured by polarity and subjectivity of 2020 election related tweets. Sentiment polarity determines whether the tweet is positive, negative, or neutral on a scale of -1 (most negative) to 1 (most positive). Subjectivity measures how much opinion is inserted into the tweet. It is generally believed that high opinion tweets (high subjectivity is closer to 1 and low subjectivity is closer to 0). We can then combine the absolute value of polarity and the value of subjectivity to get an overall sentiment score for the tweet.

Feature engineering included extracting location from the tweet if possible (including which state), determining if possible whether the tweet was from an American (if there was a state listed or other "American" related word in the locations), determining whether Trump and/or Biden were discussed in the tweet, determine whether it was a retweet, determining which hour the tweet was created (between 0 and 71), and measuring polarity, subjectivity, and overall sentiment of the tweet by combining subjectivity and polarity.

Because these tweets were made after the first Presidential debate, it will be especially interesting to determine how sentiment changes as the hours pass. Do people become more or less subjective or polarizing?

## Exploratory Data Analysis

## Baseline Models and Final Model Results



