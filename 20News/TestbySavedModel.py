import pickle
import codecs
from timeit import default_timer as timer
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

test_file = "/TestClassifier"
saved_estimator = '/finalized_model_20News.sav'
saved_vector = '/finalized_vectorizer_20News.sav'

start = timer()
with codecs.open(dir_path + test_file, 'r', encoding='utf-8', errors='replace') as content_file:
    articolo = content_file.read()


loaded_model = pickle.load(open(dir_path + saved_estimator, 'rb'))
loaded_vect = pickle.load(open(dir_path +saved_vector, 'rb'))

x = loaded_vect.transform([articolo])

result = loaded_model.predict(x)
end = timer()
print("\n")
print("The resulting topic on prediction of TestClassifier is : " + str(result))
print("Time : "+"%.3f" %(end - start))
