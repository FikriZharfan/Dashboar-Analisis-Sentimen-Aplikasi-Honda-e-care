import os
import time
from train_model import read_dataset, preprocess_text

path = os.path.join(os.getcwd(), "data", "ulasan_honda_ecare_5k.csv")
df = read_dataset(path)
df = df.dropna(subset=["content", "label"]).reset_index(drop=True)
print("valid labeled", len(df))
unique = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
print("unique content before cleaning", len(unique))
start = time.time()
unique["cleaned_text"] = unique["content"].astype(str).apply(preprocess_text)
end = time.time()
print("cleaned runtime", end - start)
clean = unique[unique["cleaned_text"].str.strip() != ""].reset_index(drop=True)
print("unique content after cleaning", len(clean))
print("label counts", clean["label"].value_counts().to_dict())
from sklearn.model_selection import train_test_split
X = list(range(len(clean)))
y = clean["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('train', len(X_train), 'test', len(X_test))
print('test label counts', y_test.value_counts().to_dict())
