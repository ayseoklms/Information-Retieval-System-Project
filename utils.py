import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk # nltk.download çağrıları için

# NLTK kaynaklarının indirilmiş olduğundan emin olun
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')
try:
    WordNetLemmatizer().lemmatize("cats")
except LookupError:
    nltk.download('wordnet')
try:
    PorterStemmer().stem("running")
except LookupError: # Porter stemmer genelde punkt ile gelir ama emin olalım
    pass


stop_words_set = set(stopwords.words('english'))
stemmer_porter = PorterStemmer()
lemmatizer_wn = WordNetLemmatizer()

def load_documents_imdb(data_path_root):
    """
    IMDb veri setindeki dokümanları (train ve test altındaki pos/neg) yükler.
    data_path_root: 'aclImdb' klasörünün yolu.
    Returns: {doc_id: text_content} şeklinde bir sözlük.
    """
    documents = {}
    doc_id_counter = 0 # Basit bir sayaçla benzersiz ID

    for split in ['train', 'test']:
        split_path = os.path.join(data_path_root, split)
        if not os.path.exists(split_path):
            print(f"Uyarı: {split_path} bulunamadı.")
            continue
        for sentiment in ['pos', 'neg']:
            sentiment_path = os.path.join(split_path, sentiment)
            if not os.path.exists(sentiment_path):
                print(f"Uyarı: {sentiment_path} bulunamadı.")
                continue
            for filename in os.listdir(sentiment_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(sentiment_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Örnek ID: "train_pos_0_4.txt" -> "doc_0", "doc_1" ...
                        doc_id = f"doc_{doc_id_counter}"
                        documents[doc_id] = f.read()
                        doc_id_counter += 1
    if not documents:
        print(f"HATA: {data_path_root} altında hiç doküman yüklenemedi. Lütfen dosya yollarını kontrol edin.")
        print("IMDb veri setini 'data/aclImdb/' şeklinde çıkardığınızdan emin olun.")
    return documents

def preprocess_text(text, use_stemming=True, use_lemmatization=False):
    """
    Metni ön işler: lowercasing, HTML removal, punctuation removal, tokenization, stopword removal, stemming/lemmatization.
    """
    # 1. Lowercasing
    text = text.lower()
    # 2. HTML etiketlerini temizleme (<br /> vb.)
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text) # Diğer HTML etiketleri
    # 3. Noktalama işaretlerini ve sayıları kaldırma (sayılar opsiyonel, şimdilik kalsın, sadece harf ve boşluk)
    text = re.sub(r'[^a-z\s]', '', text) # Sadece harfleri ve boşlukları bırakır
    # text = re.sub(r'[^\w\s]', '', text) # Harf, sayı ve boşlukları bırakır, alt çizgiyi de korur
    # 4. Tokenization
    tokens = word_tokenize(text)
    # 5. Stopword removal & Stemming/Lemmatization
    processed_tokens = []
    for token in tokens:
        if token not in stop_words_set and len(token) > 1: # Tek harfli tokenları da atla
            if use_stemming:
                processed_tokens.append(stemmer_porter.stem(token))
            elif use_lemmatization: # Lemmatization genellikle daha iyi sonuç verir ama daha yavaştır
                processed_tokens.append(lemmatizer_wn.lemmatize(token)) # POS tag ile daha iyi olur ama basit tutalım
            else:
                processed_tokens.append(token)
    return processed_tokens

def create_vocabulary_report_detailed(documents, sample_size=3):
    """
    Bölüm I için daha detaylı kelime dağarcığı oluşturma adımlarını gösteren bir rapor üretir.
    """
    print("\n" + "="*70)
    print("BÖLÜM I: TERİMLERİN KELİME DAĞARCIĞININ BELİRLENMESİ (ÖRNEK RAPOR)")
    print("="*70 + "\n")

    all_original_tokens_sample = []
    all_processed_tokens_sample_stem = []
    # all_processed_tokens_sample_lemma = [] # İstenirse eklenebilir

    doc_ids = list(documents.keys())
    if not doc_ids:
        print("Rapor için doküman bulunamadı.")
        return {}, {}

    sample_doc_ids = doc_ids[:sample_size]
    print(f"İlk {len(sample_doc_ids)} doküman için ön işleme adımları gösteriliyor:\n")

    # Adım adım raporlama için
    report_steps = []

    for i, doc_id in enumerate(sample_doc_ids):
        text = documents[doc_id]
        report_steps.append(f"\n--- Doküman ID: {doc_id} ---")
        report_steps.append(f"Orijinal Metin (ilk 100 karakter): {text[:100].strip()}...")

        # 1. Word segmentation/tokenization (terimlere ayırma) - Orijinal haliyle (lowercase sonrası)
        # HTML ve noktalama temizliği yapılmadan tokenizasyon yanıltıcı olabilir,
        # Bu yüzden lowercase + basit temizlik sonrası tokenleri "orijinal" kabul edelim.
        cleaned_lower_text = re.sub(r'<br\s*/?>', ' ', text.lower())
        cleaned_lower_text = re.sub(r'<[^>]+>', '', cleaned_lower_text)
        cleaned_lower_text_for_tokenization = re.sub(r'[^a-z\s]', '', cleaned_lower_text)
        
        original_tokens_for_doc = word_tokenize(cleaned_lower_text_for_tokenization)
        all_original_tokens_sample.extend(original_tokens_for_doc)
        report_steps.append(f"  a) Tokenize Edilmiş (temizlenmiş, lowercase): {original_tokens_for_doc[:10]}...")

        # 2. Yaygın kelimeleri çıkarma: Stop words/Durak kelimeleri
        tokens_after_stopwords = [t for t in original_tokens_for_doc if t not in stop_words_set and len(t) > 1]
        report_steps.append(f"  b) Stopword'ler Çıkarıldıktan Sonra: {tokens_after_stopwords[:10]}...")

        # 3. Normalizasyon (terimlerin denklik sınıflandırması) - Bu adım genellikle stemming/lemmatization ile birleşir
        # Burada ayrıca bir normalizasyon adımı göstermek yerine stemming/lemmatization'ı normalizasyonun bir parçası olarak düşünebiliriz.

        # 4. Stemming-Lemmatization (Kök bulma) - Porter Stemmer
        stemmed_tokens_for_doc = [stemmer_porter.stem(t) for t in tokens_after_stopwords]
        all_processed_tokens_sample_stem.extend(stemmed_tokens_for_doc)
        report_steps.append(f"  c) Stemming (Porter) Uygulandıktan Sonra: {stemmed_tokens_for_doc[:10]}...")
        report_steps.append(f"     Bu doküman için nihai işlenmiş terimler: {stemmed_tokens_for_doc[:10]}...")

    # Orijinal kelimeleri alfabetik sırayla göster (örnek üzerinden)
    sorted_original_sample_tokens = sorted(list(set(t for t in all_original_tokens_sample if len(t)>1 and t not in stop_words_set))) # Stopwordsüz ve tek harfsiz
    print("\nÖrnek Dokümanlardan Elde Edilen 'Orijinal' Kelimeler (Stopwordsuz, Alfabetik, İlk 20):")
    for token in sorted_original_sample_tokens[:20]:
        stemmed_version = stemmer_porter.stem(token)
        print(f"  Orijinal: {token:<20} -> Stemming Sonrası: {stemmed_version}")
    if len(sorted_original_sample_tokens) > 20: print("  ...")

    # Tüm dokümanlar üzerinden nihai sözlüğü oluştur (Stemming ile)
    print("\nNihai Sözlük Oluşturuluyor (Tüm Dokümanlar Üzerinden - Stemming ile)...")
    processed_docs_full = {}
    final_vocabulary_stem_full = set()
    for doc_id, text_content in documents.items():
        processed_tokens_for_doc = preprocess_text(text_content, use_stemming=True, use_lemmatization=False)
        processed_docs_full[doc_id] = processed_tokens_for_doc
        final_vocabulary_stem_full.update(processed_tokens_for_doc)

    sorted_final_vocabulary = sorted(list(final_vocabulary_stem_full))

    print("\nNihai Sözlük (Stemming ile, Alfabetik, İlk 20 Terim):")
    print(sorted_final_vocabulary[:20])
    print(f"Nihai Sözlük Boyutu (Stemming): {len(sorted_final_vocabulary)}")

    # Rapor metnini yazdır
    for step_text in report_steps:
        print(step_text)
    print("\n" + "="*70 + "\n")

    return sorted_final_vocabulary, processed_docs_full


if __name__ == '__main__':
    # Test için
    IMDB_DATA_PATH = r'C:\Users\ayseo\OneDrive\Masaüstü\bilgi-erisim-sistemleri-proje\data' # Bu yolu kendi sisteminize göre ayarlayın
    
    # Küçük bir alt küme ile test etmek daha hızlı olur
    raw_docs_full = load_documents_imdb(IMDB_DATA_PATH)
    if raw_docs_full:
        print(f"Toplam {len(raw_docs_full)} doküman yüklendi.")
        
        # Rapor için küçük bir örnek set alalım
        sample_doc_ids_for_report = list(raw_docs_full.keys())[:3]
        sample_docs_for_report_content = {doc_id: raw_docs_full[doc_id] for doc_id in sample_doc_ids_for_report}

        # Rapor fonksiyonu tüm dokümanları da işleyebilir veya sadece rapor için örnekleri işleyebilir.
        # create_vocabulary_report_detailed hem raporu basar hem de TÜM dokümanların işlenmiş halini döndürür.
        final_vocab, processed_docs_all = create_vocabulary_report_detailed(raw_docs_full, sample_size=2)
        
        print(f"\nFinal vocabulary (stemmed) first 10: {final_vocab[:10]}")
        # print(f"Processed docs (sample from all): {list(processed_docs_all.items())[:1]}")
    else:
        print("Test için doküman yüklenemedi.")