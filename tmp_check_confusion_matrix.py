import os
import joblib
import pandas as pd
from train_model import DATA_PATH, MODELS_DIR, read_dataset, preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

raw = read_dataset(DATA_PATH)
raw = raw.dropna(subset=['content', 'label']).reset_index(drop=True)
raw = raw.drop_duplicates(subset=['content']).reset_index(drop=True)
raw['cleaned_text'] = raw['content'].astype(str).apply(preprocess_text)
raw = raw[raw['cleaned_text'].str.strip() != ''].reset_index(drop=True)

le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf.pkl'))
nb = joblib.load(os.path.join(MODELS_DIR, 'nb_model.pkl'))
svm = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))

X = tfidf.transform(raw['cleaned_text'])
y = le.transform(raw['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cm_nb = confusion_matrix(y_test, nb.predict(X_test))
cm_svm = confusion_matrix(y_test, svm.predict(X_test))
print('nrows', len(raw))
print('shape', X.shape)
print('y_test counts', pd.Series(y_test).value_counts().to_dict())
print('cm_nb', cm_nb.tolist())
print('cm_svm', cm_svm.tolist())
print('tp/tn/fp/fn maybe positive label order depends on classes? label classes', le.classes_.tolist())
