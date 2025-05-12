# search.py

import math # calculate_doc_score_for_query için tfidf.py içinde kullanılabilir, burada doğrudan gerekmez
from collections import defaultdict

# utils.py'den sorgu ön işleme için fonksiyonu import ediyoruz
from utils import preprocess_text

# tfidf.py'den doküman skorlama fonksiyonunu import ediyoruz
# Bu importun çalışması için tfidf.py dosyasının aynı dizinde olması gerekir.
try:
    from tfidf import calculate_doc_score_for_query
except ImportError:
    print("UYARI: tfidf.py modülü bulunamadı veya calculate_doc_score_for_query fonksiyonu eksik.")
    print("TF-IDF sıralaması düzgün çalışmayabilir.")
    # Alternatif olarak, TF-IDF mantığını buraya geri taşıyabilirsiniz veya hatayı düzeltin.
    def calculate_doc_score_for_query(*args, **kwargs): # Placeholder
        return 0.0

class SearchEngine:
    def __init__(self, inverted_index_obj):
        """
        inverted_index_obj: Oluşturulmuş InvertedIndex sınıfının bir örneği.
        """
        self.ii = inverted_index_obj

    def _merge_postings_and(self, postings_set1, postings_set2):
        """
        İki postings listesinin (doküman ID setleri) kesişimini (AND) alır.
        """
        return postings_set1.intersection(postings_set2)

    def _merge_postings_or(self, postings_set1, postings_set2):
        """
        İki postings listesinin (doküman ID setleri) birleşimini (OR) alır.
        """
        return postings_set1.union(postings_set2)

    def boolean_search(self, query_string, operator='AND'):
        """
        Basit Boolean arama yapar.
        Sorgudaki terimleri işler, postings listelerini alır.
        AND operasyonu için, en düşük doküman frekansına (DF) sahip terimden
        başlayarak kesişim alır (optimizasyon).
        OR operasyonu için birleşim alır.

        Args:
            query_string (str): Kullanıcının girdiği sorgu.
            operator (str): 'AND' veya 'OR'. Varsayılan 'AND'.

        Returns:
            list: Eşleşen doküman ID'lerinin listesi.
        """
        processed_query_terms = list(set(preprocess_text(query_string))) # Benzersiz ve işlenmiş sorgu terimleri
        
        if not processed_query_terms:
            print("Sorgu işlenemedi veya boş. Sonuç döndürülmüyor.")
            return []

        # İndekste bulunan ve postings listesi olan terimleri ve postings'lerini al
        term_postings_map = {} # {term: set_of_doc_ids}
        valid_terms_for_query = []

        for term in processed_query_terms:
            # InvertedIndex'ten sadece doc_id'leri içeren seti al
            postings_set_for_term = self.ii.get_postings_list(term) 
            if postings_set_for_term: # Eğer terim indekste varsa ve en az bir dokümanda geçiyorsa
                term_postings_map[term] = postings_set_for_term
                valid_terms_for_query.append(term)
            else:
                # Eğer AND işlemi yapılıyorsa ve bir terim bile indekste yoksa, sonuç boş kümedir.
                if operator.upper() == 'AND':
                    # print(f"'{term}' terimi indekste bulunamadı veya hiçbir dokümanda geçmiyor. AND sorgusu boş sonuç döndürecek.")
                    return []
        
        if not valid_terms_for_query:
            # print("Sorgudaki geçerli terimlerin hiçbiri indekste bulunamadı.")
            return []
        
        # Terimleri postings listelerinin uzunluğuna (DF'ye eşdeğer) göre sırala (küçükten büyüğe).
        # Bu, AND operasyonunda en kısıtlayıcı (en az dokümanda geçen) terimle başlamayı sağlar.
        sorted_terms_by_df_length = sorted(valid_terms_for_query, key=lambda t: len(term_postings_map[t]))
        
        # İlk terimin doküman ID seti ile başla
        result_doc_ids_set = term_postings_map[sorted_terms_by_df_length[0]]

        # Kalan terimlerle merge işlemi yap
        if operator.upper() == 'AND':
            for i in range(1, len(sorted_terms_by_df_length)):
                current_term = sorted_terms_by_df_length[i]
                result_doc_ids_set = self._merge_postings_and(result_doc_ids_set, term_postings_map[current_term])
                if not result_doc_ids_set:  # Eğer kesişim herhangi bir noktada boşalırsa, daha fazla devam etmeye gerek yok.
                    break
        elif operator.upper() == 'OR':
            for i in range(1, len(sorted_terms_by_df_length)):
                current_term = sorted_terms_by_df_length[i]
                result_doc_ids_set = self._merge_postings_or(result_doc_ids_set, term_postings_map[current_term])
        else:
            raise ValueError("Desteklenmeyen operatör. Lütfen 'AND' veya 'OR' kullanın.")
            
        return list(result_doc_ids_set)

    def tfidf_rank(self, query_string, top_n=10):
        """
        TF-IDF ağırlıklandırması kullanarak dokümanları sorguya göre sıralar.
        Sıralama, tfidf.py modülündeki calculate_doc_score_for_query fonksiyonu ile yapılır.

        Args:
            query_string (str): Kullanıcının girdiği sorgu.
            top_n (int): Döndürülecek en iyi sonuç sayısı.

        Returns:
            list: (doc_id, score) çiftlerinden oluşan sıralı bir liste.
        """
        processed_query_terms = list(set(preprocess_text(query_string)))
        if not processed_query_terms:
            print("TF-IDF için sorgu işlenemedi veya boş.")
            return []

        N = self.ii.total_docs 
        if N == 0:
            print("TF-IDF için indekste hiç doküman bulunmuyor.")
            return []
            
        doc_scores = defaultdict(float)

       
        candidate_docs = set()
        for term in processed_query_terms:
            candidate_docs.update(self.ii.get_postings_list(term))

        if not candidate_docs:
    
            return []

        for doc_id in candidate_docs:
        
            score = calculate_doc_score_for_query(
                processed_query_terms, 
                doc_id,                
                self.ii,               
                N                      
            )
            if score > 0: 
                doc_scores[doc_id] = score
        
        
        sorted_docs_with_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        
        return sorted_docs_with_scores[:top_n]


if __name__ == '__main__':
  

    class MockInvertedIndexForSearch:
        def __init__(self):
            self.postings_store = {
                "iyi": {"doc1", "doc2", "doc3"},
                "film": {"doc1", "doc3", "doc4"},
                "aksiyon": {"doc2", "doc3", "doc5"},
                "korkunc": {"doc4", "doc6"},
                "zaman": {"doc4", "doc6"},
                "kaybi": {"doc6"}
            }
            self.tf_store = { # {term: {doc_id: tf_val}}
                "iyi": {"doc1": 2, "doc2": 1, "doc3": 3},
                "film": {"doc1": 1, "doc3": 2, "doc4":1},
                "aksiyon": {"doc2": 3, "doc3": 1, "doc5":2},
                "korkunc": {"doc4": 2, "doc6":1},
                "zaman": {"doc4":1, "doc6":1},
                "kaybi": {"doc6":3}
            }
            self.df_store = {term: len(docs) for term, docs in self.postings_store.items()}
            self.total_docs = 6 # Toplam doküman sayısı (doc1'den doc6'ya)

        def get_postings_list(self, term):
            return self.postings_store.get(term, set())

        def get_postings_with_tf(self, term):
            return self.tf_store.get(term, {})
            
        def get_df(self, term):
            return self.df_store.get(term, 0)

    print("Search.py doğrudan çalıştırılıyor (Test Modu)...\n")
    


    mock_ii_instance = MockInvertedIndexForSearch()
    search_engine_instance = SearchEngine(mock_ii_instance)

    # Boolean Arama Testi
    print("--- Boolean Arama Testleri ---")
    query_bool1 = "iyi film"
    results_and = search_engine_instance.boolean_search(query_bool1, operator='AND')
    print(f"Sorgu (AND) '{query_bool1}': {results_and} (Beklenen: ['doc1', 'doc3'] veya ['doc3', 'doc1'])") 

    query_bool2 = "aksiyon veya korkunc" 
    # preprocess_text("aksiyon veya korkunc") -> ['aksiyon', 'korkunc'] (eğer 'veya' stopword ise)
    results_or = search_engine_instance.boolean_search("aksiyon korkunc", operator='OR')
    print(f"Sorgu (OR) 'aksiyon korkunc': {results_or} (Beklenen: doc2, doc3, doc5, doc4, doc6 içeren bir set)")

    query_bool_nonexist = "bilinmeyen terim film"
    results_nonexist_and = search_engine_instance.boolean_search(query_bool_nonexist, operator='AND')
    print(f"Sorgu (AND) '{query_bool_nonexist}': {results_nonexist_and} (Beklenen: [])")
    
    results_nonexist_or = search_engine_instance.boolean_search(query_bool_nonexist, operator='OR')
    print(f"Sorgu (OR) '{query_bool_nonexist}': {results_nonexist_or} (Beklenen: ['doc1', 'doc3', 'doc4'])")


    print("\n--- TF-IDF Arama Testleri ---")
    query_tfidf1 = "iyi aksiyon filmi" 
    
   
    try:
        from tfidf import calculate_doc_score_for_query
        if 'calculate_doc_score_for_query' not in globals() or globals()['calculate_doc_score_for_query'].__name__ == 'calculate_doc_score_for_query': # Placeholder kontrolü
             pass 
        else:
            print("UYARI: TF-IDF için gerçek 'calculate_doc_score_for_query' yüklenemedi. Test sonuçları yanıltıcı olabilir.")
    except ImportError:
        print("UYARI: tfidf.py import edilemedi. TF-IDF test sonuçları yanıltıcı olabilir.")


    ranked_results_tfidf = search_engine_instance.tfidf_rank(query_tfidf1, top_n=3)
    print(f"Sorgu (TF-IDF) '{query_tfidf1}':")
    if ranked_results_tfidf:
        for doc_id, score in ranked_results_tfidf:
            print(f"  Doc ID: {doc_id}, Skor: {score:.4f}")
    else:
        print("  Sonuç bulunamadı veya skorlar 0.")
        
    
    
    ranked_results_kaybi = search_engine_instance.tfidf_rank("kaybi", top_n=1)
    print(f"Sorgu (TF-IDF) 'kaybi':")
    if ranked_results_kaybi:
        for doc_id, score in ranked_results_kaybi:
            print(f"  Doc ID: {doc_id}, Skor: {score:.4f} (Beklenen doc6)")