"""Text Analytics Module."""
import pandas as pd
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups


class TextAnalyzer(object):
    """TextAnalyzer class Module."""

    def __init__(self):
        """Default constructor."""
        super(TextAnalyzer, self).__init__()

    def remove_stop_words(self, sentence):
        """Function to remove stop words."""
        # Update stop words lists.
        upd = ['-', '--', '>', 'from:', '|', '|>', 're:', 'subject:',
               'would', 'writes:', "don't", 'one', 'x', 'b', 'c', 'd',
               'know', 'like', 'lines:', 'people', 'nntp-posting-host:',
               'organization:', 'think', 'use', '*', ':', '>>', '$', '?',
               '%', '@', '&', '^', '#', '!', '~', '`', '}', '{', ']', '[',
               '(', ')', 'could', 'get', "i'm", "it's", 'also', '1', '2',
               '3', '4', '5', '6', '7', '8', '9', '10', 'make', 'many',
               'may', 'see', 'two', 'even', 'it.', 'it', 'much', 'time',
               'want', 'us', 'way',
               "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'"]
        stop = set(stopwords.words('english'))
        # Updating stopwords lists.
        stop.update(set(upd))
        return [i for i in sentence if i.strip() not in stop]

    def calculate_frequency(self, sentence, top_10):
        """Function to calculate frequncy of a word in a sentence."""
        return {column: sentence.split().count(column) for column in top_10}

    def most_frequent(self):
        """Most Frequent."""
        # Loading textual data.
        twenty_train = fetch_20newsgroups(subset='train')
        data = pd.DataFrame({'textcol': twenty_train.data})
        # Tokenizing dataset.
        sentence = ' '.join(data['textcol']).lower().split()
        # Removing stopwords.
        sentence = self.remove_stop_words(sentence)
        # Getting top 10 most frequent words.
        top_10 = pd.Series(sentence).value_counts()[:10]
        top_10 = top_10.index.tolist()
        data = data.merge(data['textcol'].apply(
            lambda x: pd.Series(self.calculate_frequency(x.lower(), top_10))),
            left_index=True, right_index=True)
        return data

obj = TextAnalyzer()
obj.most_frequent()
