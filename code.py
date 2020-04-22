import GetOldTweets3 as got
import pandas as pd
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix


 def GetSentCSV(text_query, since_date, until_date, filename):
        count = 50000000000000

        # Creation of query object
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setSince(since_date).setUntil(until_date) \
            .setMaxTweets(count)
        # Creation of list that contains all tweets
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        # Creating list of chosen tweet data
        text_tweets = [[tweet.date, tweet.text] for tweet in tweets]

        df = pd.DataFrame(data=text_tweets, columns=['Date', "Tweets"])
        df['Sentiment'] = df['Tweets'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        df.set_index('Date', inplace=True)
        df = df.groupby(df.index.date).mean()

        return (df.to_csv(filename))

def minmax(x):
    return((x-min(x)) / (max(x) - min(x)))

def clean(ticker):
    ticker['Scaled_Sentiment'] = minmax(ticker['Sentiment'])
    ticker['Scaled_Returns'] = minmax(ticker['Returns'])
    return(ticker)

def One_Year_NN(df):
    split_date = '2019-03-01'
    train = df.loc[df['Date'] < split_date]
    train_X = train.drop('Direction', axis = 1)
    train_Y = train['Direction']

    test = df.loc[df['Date'] >= split_date]
    test_X = test.drop('Direction', axis=1)
    test_Y = test['Direction']

    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(1,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(train_X['Sentiment'], train_Y, epochs=5, batch_size=10)

    predictions = model.predict_classes(test_X['Sentiment'])

    matrix = confusion_matrix(test_Y, predictions)

    score = model.evaluate(test_X['Sentiment'], test_Y, verbose=1)

    accuracy = score[1]

    return(print(matrix,accuracy))

def Two_Month_NN(df):
    split_date = '2019-12-31'
    train = df.loc[df['Date'] < split_date]
    train_X = train.drop('Direction', axis = 1)
    train_Y = train['Direction']

    test = df.loc[df['Date'] >= split_date]
    test_X = test.drop('Direction', axis=1)
    test_Y = test['Direction']

    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(1,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dropout(0.25))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(train_X['Sentiment'], train_Y, epochs=5, batch_size=10)

    predictions = model.predict_classes(test_X['Sentiment'])

    matrix = confusion_matrix(test_Y, predictions)


    score = model.evaluate(test_X['Sentiment'], test_Y, verbose=1)

    accuracy = score[1]

    return(print(matrix,accuracy))


if __name__ == '__main__':
    
    AGG = clean(pd.read_csv('AGG.csv'))
    SPY = clean(pd.read_csv('SPY.csv'))
    DBC = clean(pd.read_csv('DBC.csv'))
    IYR = clean(pd.read_csv('IYR.csv'))
    GLD = clean(pd.read_csv('GLD.csv'))


    One_Year_NN(AGG)
    One_Year_NN(SPY)
    One_Year_NN(DBC)
    One_Year_NN(IYR)
    One_Year_NN(GLD)

    Two_Month_NN(AGG)
    Two_Month_NN(SPY)
    Two_Month_NN(DBC)
    Two_Month_NN(IYR)
    Two_Month_NN(GLD)
