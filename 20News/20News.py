
import codecs
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from timeit import default_timer as timer
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

# nome usato per i file salvati con pikle
filename = 'finalized_model_20News.sav'
filename2 = 'finalized_vectorizer_20News.sav'

# rimozione di non_letters e /n
non_letters = True
remove_backslashes = True

# rimozione quotes e regular_expressions
remove_quotes = True

# Attivare i due estimatori
multinomial = True
bernoulli = True

# Attivare per usare cross-validation
cross_validation = True

# Attiva test su file TestClassifier
test_classifier = True

# numero di iterazioni per ogni train_size nella cross-validation
num_iter = 100

# divide il DataSet in size_steps
size_steps = 10

#dimensione ad ogni passo del test_set ad ogni step
test_size = 0.1

# Percorso files 20News
news_groups_folder = dir_path + '/20news-18828'


def print_options():
    # Stampa la configurazione
    print("\n")
    print("20 News Groups-NaiveBayes-learning curves\n\n")
    print("Make Bernoulli: " + str(bernoulli))
    print("Make Multinomial: " + str(multinomial) + "\n")
    print("Want to remove Quotes: " + str(remove_quotes))
    print("Want to remove non letters: " + str(non_letters) + "\n")
    print("Cross Validation active: " + str(cross_validation))
    if cross_validation:
        print("Number of iterations for each train_size: " + str(num_iter))
        print("Percentage of Test Size: " + str(test_size * 100)+"%" + "\n")
    print("Learning curves steps: " + str(size_steps) + "\n")



def multiple_header(text):
    # rimozione header multipli
    if ":" in text.splitlines()[0]:
        if len(text.splitlines()[0].split(":")[0].split()) == 1:
            return True
    else:
        return False

def strip_newsgroup_header(text):
    # strip article header
    if multiple_header(text):
        _before, _blankline, after = text.partition('\n\n')
        if len(after) > 0 and multiple_header(after):
            after = strip_newsgroup_header(after)
        return after
    else:
        return text

def strip_newsgroup_quoting(text):
    # rimozione quotes
    regex_quotes = re.compile(r'(^In article|said:|writes in|writes:|wrote:|says:|^Quoted from|^\||^>)')
    good_lines = [line for line in text.split('\n')
                  if not regex_quotes.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    # reove article footer
    lines = text.strip().split('\n') # rimuovo spazi ad inizio e fine e poi divido in righe
    for num_line in range(len(lines) - 1, -1, -1):   # [range([start], stop[, step])]
        line = lines[num_line]
        if line.strip().strip('-') == '':
            break
    if num_line> 0:
        return '\n'.join(lines[:num_line])   # join rows con degli accapi ma mi fermo a line_num
    else:
        return text


def clean_text(raw_text):
    stripped_text = strip_newsgroup_header(raw_text)
    stripped_text = strip_newsgroup_footer(stripped_text)

    if remove_quotes:
        stripped_text = strip_newsgroup_quoting(stripped_text)

    if remove_backslashes:
        split_text = stripped_text.split('\n')
        stripped_text = ' '.join(split_text)
        split_text = stripped_text.split('^')
        stripped_text = ' '.join(split_text)
        split_text = stripped_text.split('/')
        stripped_text = ' '.join(split_text)
        split_text = stripped_text.split('\\')
        stripped_text = ' '.join(split_text)
        split_text = stripped_text.split('*')
        stripped_text = ' '.join(split_text)
        split_text = stripped_text.split('#')
        stripped_text = ' '.join(split_text)

    if non_letters:
        stripped_text = re.sub("[^a-zA-Z]", " ", stripped_text) # sostituisco le parole che non iniziano con una lettera [^a-zA-Z] con " " in stripped_text
    lower = stripped_text.lower()
    return lower


def extract_dataset(folder, articles, id_):
    # scansiono la cartella per raccogliere i files

    dir_list = [x for x in os.listdir(folder)[1:]]
    for each_dir in dir_list:
        path = os.path.join(folder, each_dir)
        each_file = os.listdir(path)
        for file in each_file:
            category= each_dir
            with codecs.open(os.path.join(path, file), 'r', encoding='utf-8', errors='replace') as file:
                raw_text = file.read()
            cleaned = clean_text(raw_text)
            id_ += 1
            articles.append((id_, category, cleaned))


# y è l'array di categorie di tutto articles
# text è l'array di testi
def create_bag_of_word(train):
    # labels
    y_ = [row[1] for row in train]

    # corpus list
    text_vect = [row[2] for row in train]

    # oggetto vectorizer per trasformare la lista
    vectorizer_ = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words='english')

    X_ = vectorizer_.fit_transform(text_vect)
    print('Bag of Word dimensions: ' + str(X_.shape))
    return vectorizer_, X_, y_


def plot_learning_curve(estimator_, title_, X_, y_, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, size_steps)):
    # calcola i valori e inserisce i dati nel grafico
    plt.figure()
    plt.title(title_)
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Classification Accuracy")

    # learning curve da sklearn
    train_sizes, train_scores, test_scores = learning_curve(estimator_, X_, y_, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)


    train_scores_avg = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print(title_ + ": Score obtained in final step = " + str(test_scores_mean[len(test_scores_mean) - 1]))

    plt.grid()

    plt.fill_between(train_sizes, train_scores_avg - train_scores_std, train_scores_avg + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_avg, 'D-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'D-', color="g", label="Test score")

    plt.legend(loc="best")
    return plt

def do_Bernoulli(DoIt, X, y, cv):
    if DoIt:
        bernoulli_start = timer()
        print("Bernoulli learning curve...")
        bernoulli_nb = BernoulliNB(alpha=.01)
        title = "Bernoulli - 20NewsGroups"
        plot_learning_curve(bernoulli_nb, title, X, y, cv=cv, n_jobs=4)  # n_jobs è il numero di thread che può usare
        bernoulli_end = timer()
        print("Bernoulli time elapsed: " + str("%.3f" % (bernoulli_end - bernoulli_start)) + " seconds" + "\n")


def do_Multinomial(DoIt, X, y, cv):
    if DoIt:
        multinomial_start = timer()
        print("Multinomial learning curve...")
        multinomial_nb = MultinomialNB(alpha=.01)
        title_ = "Multinomial - 20 News Groups"
        plot_learning_curve(multinomial_nb, title_, X, y, cv=cv, n_jobs=4)
        multinomial_finish = timer()
        print("Multinomial time : " + str("%.3f" % (multinomial_finish - multinomial_start)) + " seconds")

def classifier(vect, text, naiveB):
    text = clean_text(text)
    x = vect.transform([text])
    return naiveB.predict(x)

print_options()

start = timer()

# matrice dove salvo i risulati di extract_dataset
articles_data = []
id = 0

extract_dataset(news_groups_folder, articles_data, id)
vectorizerstart = timer()
vectorizer, X, y = create_bag_of_word(articles_data)

vectorizerstop = timer()
print("Time used for vectorizer: " + str("%.3f" % (vectorizerstop-vectorizerstart)) + " seconds")


# Cross-validation
cv = None
if cross_validation:
    cv = ShuffleSplit(n_splits=num_iter, test_size=test_size, random_state=None)

do_Bernoulli(bernoulli, X, y, cv)

do_Multinomial(multinomial, X, y, cv)

end = timer()
print("Total time : " + str("%.3f" % (end - start)) + " seconds")

plt.show()

if test_classifier:
    print("\n")

    mul = MultinomialNB(alpha=0.1)
    mul.fit(X, y)
    pickle.dump(mul, open('20News/'+filename, 'wb'))
    pickle.dump(vectorizer, open('20News/'+filename2, 'wb'))


"""   with codecs.open('20News/TestClassifier', 'r', encoding='utf-8', errors='replace') as content_file:
       articolo = content_file.read()

   print("Result of classifier on TestClassifier: " + str(classifier(vectorizer, articolo, mul)))

"""
