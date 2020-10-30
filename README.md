# Election_Tweets

## Project Goal and Process

The goal of this project was to predict sentiment as measured by polarity and subjectivity of 2020 election related tweets. Sentiment polarity determines whether the tweet is positive, negative, or neutral on a scale of -1 (most negative) to 1 (most positive). Subjectivity measures how much opinion is inserted into the tweet. It is generally believed that high opinion tweets (high subjectivity is closer to 1 and low subjectivity is closer to 0). We can then combine the absolute value of polarity and the value of subjectivity to get an overall sentiment score for the tweet.

Feature engineering included extracting location from the tweet if possible (including which state), determining if possible whether the tweet was from an American (if there was a state listed or other "American" related word in the locations), determining whether Trump and/or Biden were discussed in the tweet, determine whether it was a retweet, determining which hour the tweet was created (between 0 and 71), and measuring polarity, subjectivity, and overall sentiment of the tweet by combining subjectivity and polarity. Polarity and subjectivity were measured using the Python TextBlob library.

Because these tweets were made after the first Presidential debate, it will be especially interesting to determine how sentiment changes as the hours pass. Do people become more or less subjective or polarizing?

## Obtaining Data
Data Source: [GitHub Repo link provided @12PM Friday 10/29/2020]
1. Clone [GitHub Repo link provided @12PM Friday 10/29/2020]
2. Follow the instructions in the repository to install twarc and tqdm.
3. Apply for a twitter developer account.
4. Save api key, save api secret key, save bearer token.
5. Enter your twitter api information into twarc.
6. Use a mv command to move the contents of the desired days into a new single directory.
7. Look inside the cloned repository for the appropriate .txt files containing tweet ids. (ex. cat * >> file name.txt)
8. Concatenate those files into one file.
9. In the terminal, use awk 'NR % 100 == 0' <file.txt> > <result.txt> to systematically sample every 100th tweet id. These are the tweets you will hydrate.
10. Modify the hydrate.py script in the cloned repository and run the script to rehydrate tweets from your file of tweet ids.
11. Analyze tweets.

## Exploratory Data Analysis

## Baseline Models and Final Model Results



