# Basit Bilgi Erişim Sistemi Projesi

Bu proje, BİL 302: Bilgi Erişim Sistemleri dersi kapsamında geliştirilmiş basit bir bilgi erişim sistemidir. Sistem, IMDb film yorumları veri seti üzerinde metin ön işleme, ters indeks oluşturma, Boolean arama ve TF-IDF tabanlı sıralı arama yeteneklerini içermektedir. Arayüz, komut satırı üzerinden çalışmakta ve `colorama` kütüphanesi ile renklendirilmiştir.

## Özellikler

*   **Metin Ön İşleme:**
    *   Küçük harfe çevirme
    *   HTML etiketlerinin temizlenmesi
    *   Noktalama işaretlerinin kaldırılması
    *   Tokenizasyon (NLTK `word_tokenize`)
    *   Stopword (durak kelimeleri) çıkarılması (NLTK İngilizce stopword listesi)
    *   Kök bulma (Stemming - NLTK Porter Stemmer)
*   **Ters İndeks Oluşturma:**
    *   Her bir terim için postings listesi (doküman ID ve terim frekansı)
    *   Doküman frekansı (DF) ve toplam korpus frekansı bilgileri
*   **Boolean Arama:**
    *   AND ve OR operatörlerini destekler.
    *   AND sorguları için doküman frekansına göre optimize edilmiş merge algoritması.
*   **TF-IDF Sıralı Arama:**
    *   Logaritmik normalize edilmiş TF ve standart IDF kullanılarak TF-IDF skorları hesaplanır.
    *   [EĞER KULLANDIYSANIZ: Proje dökümanındaki `tf(w,d) * log(D/df(w))` formülü de TF için ham frekans kullanılarak bir seçenek olarak değerlendirilmiştir.]
    *   Dokümanlar sorguya göre TF-IDF skorlarına göre sıralanır.
*   **Değerlendirme:**
    *   Precision@K, Recall@K, F1-Score@K
    *   Average Precision (AP) ve Mean Average Precision (MAP)
    *   (Not: Değerlendirme metrikleri, `main.py` içerisinde örnek sorgular ve **manuel olarak oluşturulması gereken bir ground truth** seti üzerinden hesaplanmaktadır. `app.py` içindeki dinamik sorgulama sırasında gösterilen metrikler, sorgudaki tüm terimleri içeren dokümanları "ilgili" kabul eden bir pseudo-ground truth'a dayanır ve sadece anlık bir fikir verir.)
*   **Komut Satırı Arayüzü (`app.py`):**
    *   Kullanıcı dostu, renkli menüler ve çıktılar (`colorama` ile).
    *   Boolean veya TF-IDF arama seçeneği.
    *   Boolean için operatör (AND/OR) seçimi.
    *   Arama sonuçlarının (doküman ID ve içerik önizlemesi) listelenmesi.

## Kullanılan Veri Seti

*   **IMDb Movie Reviews (Large Movie Review Dataset v1.0)**
    *   Kaynak: `https://ai.stanford.edu/~amaas/data/sentiment/`
    *   50.000 adet film yorumu (25.000 train, 25.000 test).

## Kurulum

1.  **Proje Dosyalarını İndirin:**
    Bu repoyu klonlayın veya ZIP olarak indirin.

2.  **Python Ortamı:**
    Python 3.8 veya üzeri bir sürümün kurulu olduğundan emin olun.

3.  **Gerekli Kütüphaneleri Yükleyin:**
    Proje ana dizininde bir terminal açın ve aşağıdaki komutu çalıştırın:
    ```bash
    pip install -r requirements.txt
    ```
    `requirements.txt` dosyası şunları içermelidir:
    ```
    nltk
    numpy
    scikit-learn # (Değerlendirme metriklerinde sklearn kullanılmadıysa bu satır gereksiz olabilir, numpy yeterli olabilir)
    colorama
    ```

4.  **NLTK Kaynaklarını İndirin:**
    Bir Python interpretörü açın (veya bir script oluşturup çalıştırın) ve aşağıdaki komutları girin:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet') # Lemmatization için (eğer kullanılıyorsa) veya bazı NLTK bağımlılıkları için
    ```

5.  **Veri Setini Hazırlayın:**
    *   IMDb veri setini (`aclImdb_v1.tar.gz`) [Stanford AI Group](https://ai.stanford.edu/~amaas/data/sentiment/) adresinden indirin.
    *   Proje ana dizininde `data` adında bir klasör oluşturun.
    *   İndirdiğiniz `aclImdb_v1.tar.gz` dosyasını açın ve içinden çıkan `aclImdb` klasörünü oluşturduğunuz `data` klasörünün içine kopyalayın.
    *   Sonuç olarak dosya yapınız şu şekilde olmalıdır:
        ```
        bilgi-erisim-sistemleri-proje/
        ├── data/
        │   └── aclImdb/
        │       ├── train/
        │       ├── test/
        │       └── ... (diğer dosyalar)
        ├── app.py
        ├── main.py
        ├── utils.py
        ├── inverted_index.py
        ├── search.py
        ├── tfidf.py
        ├── evaluation.py
        └── requirements.txt
        ```
    *   `main.py` ve `utils.py` dosyalarındaki `IMDB_DATA_PATH` değişkeninin `r'data/aclImdb'` (veya sisteminize göre doğru yol) olarak ayarlandığından emin olun. Eğer `aclImdb` klasörünün içeriğini doğrudan `data` altına açtıysanız (`data/train`, `data/test` şeklinde), bu yolu `r'data'` olarak güncelleyin.

## Çalıştırma

Sistemin interaktif komut satırı arayüzünü başlatmak için proje ana dizininde bir terminal açın ve aşağıdaki komutu çalıştırın:

```bash
python app.py