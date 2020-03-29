import os
from util import Util
util = Util()
import pandas as pd
import pickle
import nltk
#os.chdir("E://Study//GitLab//SMS_Classification//data//training_data.csv")

training_data = pd.read_csv("E://Study//GitLab//SMS_Classification//data//training_data.csv",encoding="utf-8")
#util.show_column_names(training_data)

training_data["message"] = training_data["message"].apply(util.clean_message)

training_data = util.label_encode(training_data,"category")
#print(training_data.head())

training_data["message"] = training_data["message"].str.lower()

training_data,cv,cv_vocab = util.text_features(training_data,"message")
src_path = os.chdir("E://Study//GitLab//SMS_Classification//src//")
#text_vectorizer_path = os.path.join(src_path,'text_vectorizer.pkl')
pickle.dump(cv_vocab,open("E://Study//GitLab//SMS_Classification//src//text_vectorizer_path.pkl",'wb'))

X_train,X_test,y_train,y_test = util.train_test_split(training_data,cv)

model = util.train_model(X_train,y_train)
#model_path = os.path.join(src_path,'model.pkl')
pickle.dump(model,open("E://Study//GitLab//SMS_Classification//src//model.pkl",'wb'))

y_pred = util.predict(X_test,model)

classification_report = util.classify_report(y_test,y_pred)
print(classification_report)
