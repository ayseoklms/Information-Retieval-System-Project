import os
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


# Eğer tokenization için NLTK veritabanı inmemişse şunu da ekleriz (ilk sefer için):
# nltk.download('punkt')

def read_documents(folder_path):
    documents = []
    for label in ['pos', 'neg']:  # pozitif ve negatif klasörleri
        label_path = os.path.join(folder_path, label)
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append((content, label))
    return documents

def tokenize_documents(documents):
    tokenized_docs = []
    for content, label in documents:
        tokens = word_tokenize(content.lower())  # Küçük harfe çevirip tokenize ediyoruz
        tokenized_docs.append((tokens, label))
    return tokenized_docs

if __name__ == "__main__":
    data_path = "data/train"  # klasör yapımıza göre ayarla
    docs = read_documents(data_path)
    tokenized_docs = tokenize_documents(docs)

    print("Birinci dokümanın ilk 20 kelimesi:", tokenized_docs[0][0][:20])
