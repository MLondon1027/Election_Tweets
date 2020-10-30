import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob


def top_words(term_freq, num=10):
    word_totals = np.zeros(len(word_list))

    for term in tf:
        word_totals += term.toarray()[0]

    word_dict = zip(word_list, word_totals)
    word_dict = sorted(word_dict, key=lambda x: x[1], reverse=True)[1:num+1]

    return word_dict


def make_bars(data):
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots(figsize=(10, 10), dpi=72)
    ax.bar(data)
    ax.set_title('Last name mentions', fontsize=30)
    ax.set_xlabel('')
    pass


def get_sentiment(df):
    df['polarity'] = df.apply(lambda row: round(
        TextBlob(row.full_text).sentiment.polarity, 2), axis=1)
    df['subjectivity'] = df.apply(lambda row: round(
        TextBlob(row.full_text).sentiment.subjectivity, 2), axis=1)

    return df


def main():
    df = pd.read_csv('data/final_election.csv')

    n_features = 1000
    stops = ENGLISH_STOP_WORDS.union({'000', '10', '11', '15', '150', '16', '19', '1987', '1st', '200', '2012', '2016', '2018', 'rt',
                                      '35',  '4d', '611', '750', '80s', '90', '90s', '9am', '9pm', 'https'})  # , 'joebiden', 'realdonaldtrump', 'joe', 'donald',
#                                  'just', 'let', 'amp', 'going', 'know', 've', 'got', 'debates2020', 'debate', 'tonight', 'did', 'said', 'don', 'chris', 'wallace',
#                                  'say', 'want', 'time', 'new', 'night', 'president'})

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words=stops)
    tfidf = tfidf_vectorizer.fit_transform(df.full_text)

    tf_vectorizer = CountVectorizer(max_df=1.0, min_df=1,
                                    max_features=n_features,
                                    stop_words=stops)
    tf = tf_vectorizer.fit_transform(df.full_text)

    top_twenty = top_words(tf, num=20)

    states_color = {'CA': 'blue', 'CO': 'blue', 'CT': 'blue', 'DE': 'blue', 'HI': 'blue',
                    'IL': 'blue', 'ME': 'blue', 'ML': 'blue', 'MD': 'blue', 'MN': 'blue',
                    'NV': 'blue', 'NH': 'blue', 'NJ': 'blue', 'NM': 'blue', 'NY': 'blue',
                    'OR': 'blue', 'RI': 'blue', 'VT': 'blue', 'VA': 'blue', 'WA': 'blue',
                    'AL': 'red', 'AK': 'red', 'AZ': 'red', 'AR': 'red', 'FL': 'red',
                    'GA': 'red', 'ID': 'red', 'IN': 'red', 'IA': 'red', 'KA': 'red',
                    'KY': 'red', 'LA': 'red', 'MI': 'red', 'MS': 'red', 'MO': 'red',
                    'MT': 'red', 'NE': 'red', 'NC': 'red', 'ND': 'red', 'OH': 'red',
                    'OK': 'red', 'PA': 'red', 'SC': 'red', 'SD': 'red', 'TN': 'red',
                    'TX': 'red', 'UT': 'red', 'WV': 'red', 'WI': 'red', 'WY': 'red'}


if __name__ == '__main__':
    main()
