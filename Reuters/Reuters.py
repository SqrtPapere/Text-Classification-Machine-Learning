
import html
import codecs
import os
import re
from html.parser import HTMLParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from timeit import default_timer as timer
from collections import Counter
import pickle

# nome usato per i file salvati con pikle
filename = 'finalized_model_Reut.sav'
filename2 = 'finalized_vectorizer_Reut.sav'

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# Attivare per usare cross-validation
cross_validation = True

# divide il DataSet in size_steps
size_steps = 10

# numero di iterazioni per ogni train_size nella cross-validation
num_iterations = 100

# dimensione ad ogni passo del test_set ad ogni step
test_size = 0.1

# Attivare i due estimatori
multinomial = True
bernoulli = True

# Attiva test su file TestClassifier
test_classifier = True

# Percorso files reuters
REUTERS_FOLDER = dir_path+'/reuters21578/'


def print_options():
    # Stampa la configurazione
    print("\n")
    print("Reuters-NaiveBayes-learning curves\n\n")
    print("Make Bernoulli: " + str(bernoulli))
    print("Make Multinomial: " + str(multinomial) + "\n")
    print("Cross Validation active: " + str(cross_validation) + "\n")
    if cross_validation:
        print("Number of iterations for each train_size: " + str(num_iterations))
        print("Percentage of Test Size: " + str(test_size * 100)+"%" +"\n")
    print("Learning curves steps: " + str(size_steps) + "\n")


"""
implementazione del parser disponibile su scikitlearn at :
 http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
"""
class ReutersParser(HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True

    def handle_endtag(self, tag):
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append( (self.topics, self.body) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data


def filter_doc_list_through_topics(frequent_topics, docs):
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in frequent_topics:
                d_tup = (t, d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs


def get_frequent_topic_list(topics_, docs):
    # Ritorna una lista contenente le 10 labels più frequenti
    topics_occurrences = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in topics_:
                topics_occurrences.append(t)
                break
    occurences_dict = Counter(topics_occurrences)
    return list(dict(occurences_dict.most_common(10)).keys())


def create_bag_of_word(docs):
    # vettore di labels
    y_ = [d[0] for d in docs]

    # corpus list
    text_vect = [d[1] for d in docs]

    # oggetto vectorizer per trasformare la lista
    vectorizer_ = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words='english')

    X_ = vectorizer_.fit_transform(text_vect)
    print('Bag of Word dimensions: '+ str(X_.shape))
    return vectorizer_, X_, y_


def plot_learning_curve(estimator_, title_, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, size_steps)):
    # calcola i valori delle learning curve e plotta i risultati
    plt.figure()
    plt.title(title_)
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Classification Accuracy")

    # learning curve disponibile su scikitlearn
    train_sizes, train_scores, test_scores = learning_curve(estimator_, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    test_scores_avg = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    train_scores_avg = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    print(title_ + ": Score obtained in final step :" + str(test_scores_avg[len(test_scores_avg) - 1]))

    plt.grid()

    plt.fill_between(train_sizes, train_scores_avg - train_scores_std, train_scores_avg + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_avg - test_scores_std, test_scores_avg + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_avg, 'd-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_avg, 'd-', color="g", label="Test score")

    plt.legend(loc="best")
    return plt

def do_Bernoulli(DoIt, X, y, cv):
    if DoIt:
        bernoulli_start = timer()
        print("Bernoulli learning curve...")
        bernoulli_nb = BernoulliNB(alpha=.01)
        title = "Bernoulli - Reuters"
        plot_learning_curve(bernoulli_nb, title, X, y, cv=cv, n_jobs=4)  # n_jobs è il numero di thread che può usare
        bernoulli_end = timer()
        print("Bernoulli time : " + str("%.3f" % (bernoulli_end - bernoulli_start)) + " seconds" + "\n")


def do_Multinomial(DoIt, X, y, cv):
    if DoIt:
        multinomial_start = timer()
        print("Multinomial learning curve...")
        multinomial_nb = MultinomialNB(alpha=.01)
        title_ = "Multinomial - Reuters"
        plot_learning_curve(multinomial_nb, title_, X, y, cv=cv, n_jobs=4)
        multinomial_finish = timer()
        print("Multinomial time elapsed: " + str("%.3f" % (multinomial_finish - multinomial_start)) + " seconds")

def classifier(vect, text, naiveB):
    x = vect.transform([text])
    return naiveB.predict(x)

print_options()
start = timer()

# files è la lista di nomi utili per i file nel dataset
files = [REUTERS_FOLDER+("reut2-%03d.sgm" % (a,)) for a in range(22)]

Rparser = ReutersParser()

# crea una lista documents di elementi del tipo: (['<topic>', '<places>'], '<content>'),...
documents = []
for fn in files:
    for d in Rparser.parse(open(fn, 'rb')):
        documents.append(d) # crea una lista di elementi del tipo: (['<topic>', '<places>'], '<content>'),...

# cerco i topic dal file all-topics
topics = open(REUTERS_FOLDER + "all-topics-strings.lc.txt", "r").readlines()
topics = [t.strip() for t in topics] # remove \n

# calcola le 10 categorie più frequenti
ten_most_common_topics = get_frequent_topic_list(topics, documents)

final_doc = filter_doc_list_through_topics(ten_most_common_topics, documents)

vectorizer_start = timer()
vectorizer, X, y = create_bag_of_word(final_doc)
vectorizer_stop = timer()
print("Time used for vectorizer: " + str("%.3f" % (vectorizer_stop-vectorizer_start)) + " seconds")

cv = None
if cross_validation:
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=num_iterations, test_size=test_size, random_state=None)

do_Bernoulli(bernoulli, X, y, cv)
do_Multinomial(multinomial, X, y, cv)


end = timer()
print("Total time : " + str("%.3f" % (end - start)) + " seconds")

plt.show()

if test_classifier:
    print("\n")

    mul = MultinomialNB(alpha=0.1)
    mul.fit(X, y)
    pickle.dump(mul, open(filename, 'wb'))
    pickle.dump(vectorizer, open(filename2, 'wb'))


""" with codecs.open('TestClassifier', 'r', encoding='utf-8', errors='replace') as file:
        articolo = file.read()

    print("Result of classifier on TestClassifier: " + str(classifier(vectorizer, articolo, mul)))
"""

