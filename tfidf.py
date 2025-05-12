# tfidf.py

import math

def calculate_tf_log_normalized(term_frequency_in_doc):
    """
    Logaritmik normalize edilmiş Terim Frekansı (TF) hesaplar.
    TF(t,d) = 1 + log(tf(t,d)) if tf(t,d) > 0 else 0
    """
    if term_frequency_in_doc > 0:
        return 1 + math.log(term_frequency_in_doc)
    return 0.0

def calculate_idf(total_documents, document_frequency_of_term):
    """
    Ters Doküman Frekansı (IDF) hesaplar.
    IDF(t) = log(N / df(t))
    N: toplam doküman sayısı
    df(t): terimin geçtiği doküman sayısı
    """
    if total_documents == 0 or document_frequency_of_term == 0:
        return 0.0
    # df(t) > N olamaz ama df(t) == N ise IDF 0 olur (log(1)=0)
    # Eğer df(t) > N gibi bir durum oluşursa (yanlış veri girişi), negatif log oluşmaması için
    if document_frequency_of_term > total_documents:
        # Bu durum normalde olmamalı. Bir uyarı basılıp 0 döndürülebilir.
        print(f"Uyarı: df({document_frequency_of_term}) > N({total_documents}). IDF 0 olarak ayarlandı.")
        return 0.0
    return math.log(total_documents / document_frequency_of_term)

def calculate_tfidf_term_doc(term_frequency_in_doc, total_documents, document_frequency_of_term):
    """
    Bir dokümandaki bir terim için TF-IDF skorunu hesaplar.
    TF-IDF(t,d) = TF_log_normalized(t,d) * IDF(t)
    Proje dökümanındaki formül: tf(w,d) * log(D/df(w))
    Eğer tf(w,d) ham frekans ise ve logD/df(w) IDF ise, bu formüle göre:
    """
    tf_val = term_frequency_in_doc # Ham frekans
    idf_val = calculate_idf(total_documents, document_frequency_of_term)
    
    # Dökümandaki formül (tf(w,d) * log(D/df(w)))
    # return tf_val * idf_val

    # Daha yaygın kullanılan log-normalize TF ile:
    tf_log_norm = calculate_tf_log_normalized(term_frequency_in_doc)
    return tf_log_norm * idf_val


def calculate_doc_score_for_query(query_terms, doc_id, inverted_index, total_documents_N):
    """
    Bir dokümanın belirli bir sorguya göre TF-IDF skorunu hesaplar.
    score(doc, query) = ∑_{t in query ∩ doc} TF-IDF(t,doc)
    (Sorgu terimleri eşit ağırlıklı (1) varsayılıyor)
    Proje dökümanındaki formül: tfidf(d, q) = ∑ q * tfidf(d,w)
    Eğer q (sorgu terim ağırlığı) 1 ise: ∑ tfidf(d,w)
    """
    score = 0.0
    processed_query_terms = list(set(query_terms)) # Benzersiz sorgu terimleri

    for term in processed_query_terms:
        # Terimin dokümandaki frekansını al (tf(w,d))
        postings_for_term = inverted_index.get_postings_with_tf(term) # {doc_id: tf}
        term_freq_in_this_doc = postings_for_term.get(doc_id, 0)

        if term_freq_in_this_doc > 0: # Terim bu dokümanda geçiyorsa
            df_term = inverted_index.get_df(term)
            
            # TF-IDF(t,d) hesapla
            # Dökümandaki formüle göre mi, yoksa log-normalize TF ile mi?
            # Şimdilik log-normalize TF ile devam edelim (search.py'deki gibi)
            tfidf_val_term_doc = calculate_tfidf_term_doc(
                term_freq_in_this_doc,
                total_documents_N,
                df_term
            )
            score += tfidf_val_term_doc # q=1 varsayımıyla
            
    return score


if __name__ == '__main__':
    # Örnek Kullanım (Bu kısmı çalıştırmak için inverted_index ve utils gerekir)
    
    # Örnek değerler
    tf_wd = 5  # "word" terimi "docX" içinde 5 kez geçiyor
    N_docs = 1000 # Toplam 1000 doküman var
    df_w = 50   # "word" terimi 50 farklı dokümanda geçiyor

    tf_log_norm_val = calculate_tf_log_normalized(tf_wd)
    idf_val = calculate_idf(N_docs, df_w)
    tfidf_score_term_doc = calculate_tfidf_term_doc(tf_wd, N_docs, df_w)

    print(f"Örnek TF (log-normalized) for tf={tf_wd}: {tf_log_norm_val:.4f}")
    print(f"Örnek IDF for N={N_docs}, df={df_w}: {idf_val:.4f}")
    print(f"Örnek TF-IDF(term, doc) (log-norm TF): {tfidf_score_term_doc:.4f}")

    # Dökümandaki formüle göre (ham TF * IDF):
    tfidf_score_raw_tf = tf_wd * idf_val
    print(f"Örnek TF-IDF(term, doc) (raw TF * IDF): {tfidf_score_raw_tf:.4f}")

    # calculate_doc_score_for_query testi için mock inverted_index lazım
    class MockInvertedIndex:
        def get_postings_with_tf(self, term):
            if term == "elma":
                return {"doc1": 3, "doc2": 1}
            if term == "armut":
                return {"doc1": 1, "doc2": 4}
            return {}
        def get_df(self, term):
            if term == "elma": return 2
            if term == "armut": return 2
            return 0

    mock_ii = MockInvertedIndex()
    total_docs = 2
    query = ["elma", "armut"]
    
    score_doc1 = calculate_doc_score_for_query(query, "doc1", mock_ii, total_docs)
    score_doc2 = calculate_doc_score_for_query(query, "doc2", mock_ii, total_docs)
    
    print(f"\nMock Test: TF-IDF(doc1, query): {score_doc1:.4f}")
    print(f"Mock Test: TF-IDF(doc2, query): {score_doc2:.4f}")
    # Beklenen (log-norm TF, IDF=log(2/2)=0 olduğu için tüm skorlar 0 çıkacak, bu IDF'yi düzeltelim)
    # Eğer N=10, df_elma=2, df_armut=2 deseydik:
    # IDF_elma = log(10/2) = log(5) = 1.609
    # IDF_armut = log(10/2) = log(5) = 1.609
    # doc1:
    #   elma: tf_log(3)=1+log(3)=2.098. tfidf_elma_d1 = 2.098 * 1.609 = 3.376
    #   armut: tf_log(1)=1+log(1)=1. tfidf_armut_d1 = 1 * 1.609 = 1.609
    #   score_doc1 = 3.376 + 1.609 = 4.985
    
    # IDF'yi düzeltmek için mock_ii'yi ve total_docs'u güncelleyelim
    class MockInvertedIndexV2:
        def get_postings_with_tf(self, term):
            if term == "elma": return {"doc1": 3, "doc2": 1}
            if term == "armut": return {"doc1": 1, "doc2": 4}
            return {}
        def get_df(self, term):
            if term == "elma": return 2
            if term == "armut": return 2
            return 0
    
    mock_ii_v2 = MockInvertedIndexV2()
    total_docs_v2 = 10 # Total doküman sayısını artıralım ki IDF > 0 olsun
    
    score_doc1_v2 = calculate_doc_score_for_query(query, "doc1", mock_ii_v2, total_docs_v2)
    score_doc2_v2 = calculate_doc_score_for_query(query, "doc2", mock_ii_v2, total_docs_v2)
    
    print(f"\nMock Test v2 (N=10): TF-IDF(doc1, query): {score_doc1_v2:.4f}") # Beklenen ~4.985
    print(f"Mock Test v2 (N=10): TF-IDF(doc2, query): {score_doc2_v2:.4f}")
    # doc2:
    #   elma: tf_log(1)=1. tfidf_elma_d2 = 1 * 1.609 = 1.609
    #   armut: tf_log(4)=1+log(4)=2.386. tfidf_armut_d2 = 2.386 * 1.609 = 3.839
    #   score_doc2 = 1.609 + 3.839 = 5.448