import pickle
import codecs
import os
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))

saved_estimator = 'finalized_model_Reut.sav'
saved_vector = 'finalized_vectorizer_Reut.sav'

start = timer()
with codecs.open('TestClassifier', 'r', encoding='utf-8', errors='replace') as file:
    articolo = file.read()


loaded_model = pickle.load(open(saved_estimator, 'rb'))
loaded_vect = pickle.load(open(saved_vector, 'rb'))

x = loaded_vect.transform([articolo])

result = loaded_model.predict(x)
end = timer()
print("\n")
print("The resulting topic on prediction of TestClassifier is : " + str(result))
print("Time : "+"%.3f" %(end - start))