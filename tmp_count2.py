import os
from train_model import read_dataset, preprocess_text

path = os.path.join(os.getcwd(), 'data', 'ulasan_honda_ecare_5k.csv')
raw = read_dataset(path)
print('valid labeled', len(raw))
unique = raw.drop_duplicates(subset=['content']).reset_index(drop=True)
print('unique content before cleaning', len(unique))
unique['cleaned_text'] = unique['content'].astype(str).apply(preprocess_text)
clean = unique[unique['cleaned_text'].str.strip() != ''].reset_index(drop=True)
print('clean count', len(clean))
print('label dist', clean['label'].value_counts().to_dict())
print('empty cleaned count', len(unique) - len(clean))
print('unique cleaned_text count', len(clean['cleaned_text'].unique()))
print('duplicate cleaned_text rows', clean['cleaned_text'].duplicated().sum())
