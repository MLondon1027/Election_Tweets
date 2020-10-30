# Election_Tweets

## Project Goal and Process

The goal of this project was to predict sentiment as measured by polarity and subjectivity of 2020 election related tweets. Sentiment polarity determines whether the tweet is positive, negative, or neutral on a scale of -1 (most negative) to 1 (most positive). Subjectivity measures how much opinion is inserted into the tweet. It is generally believed that high opinion tweets (high subjectivity is closer to 1 and low subjectivity is closer to 0). We can then combine the absolute value of polarity and the value of subjectivity to get an overall sentiment score for the tweet.

Feature engineering included extracting location from the tweet if possible (including which state), determining if possible whether the tweet was from an American (if there was a state listed or other "American" related word in the locations), determining whether Trump and/or Biden were discussed in the tweet, determine whether it was a retweet, determining which hour the tweet was created (between 0 and 71), and measuring polarity, subjectivity, and overall sentiment of the tweet by combining subjectivity and polarity. Polarity and subjectivity were measured using the Python TextBlob library.

Because these tweets were made after the first Presidential debate, it will be especially interesting to determine how sentiment changes as the hours pass. Do people become more or less subjective or polarizing?

## Exploratory Data Analysis

We first examined general attributes of the data, such as the distribution of tweet lengths, how many retweets there were in the dataset, and the distribution of states the tweets came from (when available).

Some of the interesting things we noticed were:

*_"Trump" was mentioned almost twice as often as "Biden", though "Joe" was mentioned nearly twice as often as "Donald"_*
![Name mentions](/images/name_mentions.png)

*_Mean tweet length across Twitter is 34 characters.<sup>[1](https://slate.com/technology/2018/10/twitter-tweet-character-limits-280-140-effect.html)</sup> Mean tweet length for our dataset was 61 characters_*
![Tweet length distribution](/images/tweet_len_dist.png)

*_Retweets comprised 69% of our selected dataset_*
![Retweets vs original content](/images/retweets.png)


## Baseline Models and Final Model Results



