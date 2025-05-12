import os
from utils import load_documents_imdb, preprocess_text, create_vocabulary_report_detailed
from inverted_index import InvertedIndex
from search import SearchEngine
from evaluation import precision_recall_f1, average_precision, mean_average_precision
import random # Rastgele doküman seçimi için

def main():
    # --- Veri Yükleme ve Ön İşleme ---
    IMDB_DATA_PATH = r'C:\Users\ayseo\OneDrive\Masaüstü\bilgi-erisim-sistemleri-proje\data' # Bu yolu kendi sisteminize göre ayarlayın
    
    print("IMDb dokümanları yükleniyor...")
    raw_documents_all = load_documents_imdb(IMDB_DATA_PATH)
    if not raw_documents_all:
        return # Hata mesajı load_documents_imdb içinde veriliyor.
    
    print(f"{len(raw_documents_all)} doküman başarıyla yüklendi.")

    # --- Bölüm I: Kelime Dağarcığı Oluşturma Raporu ---
    # Bu fonksiyon aynı zamanda tüm dokümanları işler ve raporu basar.
    # sample_size, raporun ne kadar detaylı olacağını (kaç doküman örneği göstereceğini) belirler.
    # Tüm dokümanları işleyeceği için biraz zaman alabilir.
    print("\nKelimeler dağarcığı raporu oluşturuluyor ve metinler ön işleniyor...")
    final_vocabulary, processed_documents_all = create_vocabulary_report_detailed(raw_documents_all, sample_size=2)
    print("Ön işleme ve kelime dağarcığı raporu tamamlandı.")

    # --- Bölüm II: Ters İndeks Oluşturma ---
    print("\nTers indeks oluşturuluyor...")
    inv_index = InvertedIndex()
    inv_index.build_index(processed_documents_all)
    
    # Rapor için örnek bir terimin bilgilerini göster
    if final_vocabulary:
        # Ortanca bir terim seçelim veya belirli bir terim
        report_term_example = final_vocabulary[len(final_vocabulary) // 2] if final_vocabulary else None
        if report_term_example:
            print("\n" + "="*70)
            print("BÖLÜM II: TERS İNDERS YAPISI KURMA (ÖRNEK TERİM BİLGİSİ)")
            print("="*70)
            print(inv_index.get_term_data_for_report(report_term_example))
            print("="*70 + "\n")
    print("Ters indeks oluşturma tamamlandı.")

    # --- Arama Motoru ---
    search_engine = SearchEngine(inv_index)

    # --- BÖLÜM V: DEĞERLENDİRME ---
    # !!! BU KISIM MANUEL OLARAK OLUŞTURULMALI VEYA GELİŞTİRİLMELİDİR !!!
    # Aşağıdakiler sadece bir ÖRNEKTİR. Gerçek ground truth için dokümanları
    # okuyup sorgularla eşleştirmeniz gerekir.
    print("\n" + "="*70)
    print("BÖLÜM III, IV, V: BOOLEAN, TF-IDF ARAMA VE DEĞERLENDİRME")
    print("="*70 + "\n")

    # Örnek Sorgular ve MANUEL OLARAK BELİRLENMİŞ İlgili Doküman ID'leri (Ground Truth)
    # Doküman ID'leri "doc_0", "doc_1", ... şeklinde gidiyor.
    # Rastgele birkaç dokümanı ilgili kabul edelim (bu doğru bir GT değil, sadece örnek!)
    all_doc_ids_list = list(raw_documents_all.keys())
    
    # Eğer az doküman varsa (örn. test için sadece 100 doküman yüklendiyse)
    # ground truth oluştururken dikkatli olun.
    if len(all_doc_ids_list) < 50:
        print("UYARI: Değerlendirme için çok az doküman yüklü. Ground truth anlamsız olabilir.")
        # Bu durumda örnek sorguları ve GT'yi boş bırakabilir veya daha basit tutabilirsiniz.
        example_queries_ground_truth = {}
    else:
        # Daha fazla doküman olduğunda rastgele seçimler daha anlamlı olabilir ama yine de manuel olmalı.
        gt_doc_ids_q1 = set(random.sample(all_doc_ids_list, k=min(5, len(all_doc_ids_list)))) if len(all_doc_ids_list) > 0 else set()
        gt_doc_ids_q2 = set(random.sample(all_doc_ids_list, k=min(5, len(all_doc_ids_list)))) if len(all_doc_ids_list) > 0 else set()
        gt_doc_ids_q3 = set(random.sample(all_doc_ids_list, k=min(5, len(all_doc_ids_list)))) if len(all_doc_ids_list) > 0 else set()

        example_queries_ground_truth = {
            "amazing suspense thriller movie": gt_doc_ids_q1,
            "classic love story romance": gt_doc_ids_q2,
            "funny comedy hilarious actor": gt_doc_ids_q3,
            # Proje için daha fazla ve daha spesifik sorgular ekleyin!
            # Örneğin: "alien invasion science fiction", "world war two documentary", "animated children film"
        }
    
    print("ÖNEMLİ UYARI: Aşağıdaki değerlendirme, örnek ve rastgele seçilmiş 'ilgili dokümanlar' (ground truth) ile yapılmaktadır.")
    print("Gerçek bir değerlendirme için bu 'ground truth' setlerinin MANUEL olarak dikkatlice oluşturulması gerekir.\n")

    all_boolean_and_results_ranked = []
    all_boolean_or_results_ranked = [] # OR için de MAP hesaplayabiliriz, sıralama önemli
    all_tfidf_results_ranked = []
    all_relevant_sets_for_map = []

    # Değerlendirme için kullanılacak K değeri (örn. P@K, R@K)
    K_FOR_PRF = 10 # İlk 10 sonuç üzerinden P, R, F1

    for query, relevant_docs_gt_set in example_queries_ground_truth.items():
        if not relevant_docs_gt_set and len(all_doc_ids_list) >=50 : # Eğer GT boşsa ve yeterli doküman varsa uyarı ver (rastgele seçim başarısız olmuş olabilir)
            print(f"Uyarı: '{query}' sorgusu için ground truth seti boş. Bu sorgu değerlendirme dışında bırakılacak.")
            continue
        
        all_relevant_sets_for_map.append(relevant_docs_gt_set)

        print(f"\n--- Sorgu: '{query}' ---")
        print(f"  İlgili Dokümanlar (Ground Truth - Örnek): {list(relevant_docs_gt_set)[:5]}...") # İlk 5'i göster

        # --- Boolean Arama (AND) ---
        retrieved_bool_and = search_engine.boolean_search(query, operator='AND')
        # Boolean sonuçları sıralı olmadığı için P,R,F1'i tüm sonuçlar üzerinden veya K üzerinden hesaplayabiliriz.
        # MAP için, Boolean sonuçlarını (eğer bir sıralama mantığı yoksa) olduğu gibi kullanmak AP'yi etkiler.
        # Genellikle Boolean sistemler sıralama yapmaz, bu yüzden AP/MAP onlar için ana metrik değildir.
        # Ama proje dökümanı "hem (3) hem de (4). maddede yer alan yöntemleri kullanarak...değerlendirme yapınız" diyor.
        # Bu yüzden Boolean için de AP/MAP hesaplamaya çalışalım.
        # Eğer Boolean sonuçları için bir sıralama kriteri belirlenirse (örn. doküman ID'sine göre) MAP daha anlamlı olabilir.
        # Şimdilik, döndürülen sırayla (genellikle rastgele veya içsel bir sırayla) alalım.
        all_boolean_and_results_ranked.append(retrieved_bool_and)
        
        print(f"\n  Boolean (AND) Sonuçları ({len(retrieved_bool_and)} doküman bulundu):")
        for i, doc_id in enumerate(retrieved_bool_and[:K_FOR_PRF]):
            print(f"    {i+1}. {doc_id} - Başlık: {raw_documents_all.get(doc_id, 'N/A')[:50]}...")
        if len(retrieved_bool_and) == 0: print("    Hiç doküman bulunamadı.")
        
        p_bool_and, r_bool_and, f1_bool_and = precision_recall_f1(retrieved_bool_and[:K_FOR_PRF], relevant_docs_gt_set)
        print(f"  Boolean (AND) - P@{K_FOR_PRF}: {p_bool_and:.4f}, R (based on top {K_FOR_PRF}): {r_bool_and:.4f}, F1: {f1_bool_and:.4f}")

        # --- Boolean Arama (OR) ---
        retrieved_bool_or = search_engine.boolean_search(query, operator='OR')
        all_boolean_or_results_ranked.append(retrieved_bool_or)
        print(f"\n  Boolean (OR) Sonuçları ({len(retrieved_bool_or)} doküman bulundu):")
        for i, doc_id in enumerate(retrieved_bool_or[:K_FOR_PRF]):
            print(f"    {i+1}. {doc_id} - Başlık: {raw_documents_all.get(doc_id, 'N/A')[:50]}...")
        if len(retrieved_bool_or) == 0: print("    Hiç doküman bulunamadı.")

        p_bool_or, r_bool_or, f1_bool_or = precision_recall_f1(retrieved_bool_or[:K_FOR_PRF], relevant_docs_gt_set)
        print(f"  Boolean (OR) - P@{K_FOR_PRF}: {p_bool_or:.4f}, R (based on top {K_FOR_PRF}): {r_bool_or:.4f}, F1: {f1_bool_or:.4f}")


        # --- TF-IDF Arama ---
        # TF-IDF için tüm sonuçları alıp (top_n=len(raw_documents_all)) değerlendirebiliriz
        # veya belirli bir top_n için (örn. top_n=100 veya daha fazla)
        # AP/MAP için tüm sıralı liste gerekir.
        tfidf_ranked_results_scores = search_engine.tfidf_rank(query, top_n=len(raw_documents_all))
        tfidf_retrieved_doc_ids_ranked = [doc_id for doc_id, score in tfidf_ranked_results_scores]
        all_tfidf_results_ranked.append(tfidf_retrieved_doc_ids_ranked)

        print(f"\n  TF-IDF Sıralı Sonuçlar (ilk {K_FOR_PRF} gösteriliyor):")
        if not tfidf_ranked_results_scores:
            print("    Hiç doküman bulunamadı.")
        for i, (doc_id, score) in enumerate(tfidf_ranked_results_scores[:K_FOR_PRF]):
             print(f"    {i+1}. {doc_id} (Skor: {score:.4f}) - Başlık: {raw_documents_all.get(doc_id, 'N/A')[:50]}...")
        
        p_tfidf, r_tfidf, f1_tfidf = precision_recall_f1(tfidf_retrieved_doc_ids_ranked[:K_FOR_PRF], relevant_docs_gt_set)
        print(f"  TF-IDF - P@{K_FOR_PRF}: {p_tfidf:.4f}, R (based on top {K_FOR_PRF}): {r_tfidf:.4f}, F1: {f1_tfidf:.4f}")
        
        ap_tfidf = average_precision(tfidf_retrieved_doc_ids_ranked, relevant_docs_gt_set)
        print(f"  TF-IDF - Average Precision (AP): {ap_tfidf:.4f}")

    # --- Genel MAP Hesaplamaları ---
    print("\n--- Genel Ortalama Metrikler (Tüm Örnek Sorgular Üzerinden) ---")
    if all_boolean_and_results_ranked and all_relevant_sets_for_map:
        map_boolean_and = mean_average_precision(all_boolean_and_results_ranked, all_relevant_sets_for_map)
        print(f"  Genel MAP (Boolean AND): {map_boolean_and:.4f} (Boolean sonuçları sırasızsa AP düşük olabilir)")
    
    if all_boolean_or_results_ranked and all_relevant_sets_for_map:
        map_boolean_or = mean_average_precision(all_boolean_or_results_ranked, all_relevant_sets_for_map)
        print(f"  Genel MAP (Boolean OR): {map_boolean_or:.4f} (Boolean sonuçları sırasızsa AP düşük olabilir)")

    if all_tfidf_results_ranked and all_relevant_sets_for_map:
        map_tfidf = mean_average_precision(all_tfidf_results_ranked, all_relevant_sets_for_map)
        print(f"  Genel MAP (TF-IDF): {map_tfidf:.4f}")
    
    print("="*70)

  # main.py dosyasının sonundaki while True döngüsü

    # --- Kullanıcıdan Dinamik Sorgu Alma ---
    print("\n--- DİNAMİK SORGULAMA MODU ---")
    K_FOR_DYNAMIC_PRF = 10 # Dinamik sorgulama için P@K, R@K, F1@K için K değeri

    while True:
        user_query = input("\nSorgunuzu girin (çıkmak için 'exit' yazın): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            continue

        # --- Otomatik (Pseudo) Ground Truth Oluşturma (Bu kısım eleştirel değerlendirilmeli) ---
        # Sorgudaki tüm terimleri içeren dokümanları "ilgili" kabul edelim.
        # Bu, gerçek bir ground truth DEĞİLDİR, sadece bir gösterimdir.
        processed_user_query_terms = list(set(preprocess_text(user_query)))
        pseudo_relevant_docs_dynamic = set()
        if processed_user_query_terms: # Eğer sorgu boş değilse
            # İlk terimin geçtiği dokümanlarla başla
            if processed_user_query_terms[0] in inv_index.index:
                current_relevant_set = inv_index.get_postings_list(processed_user_query_terms[0])
                # Diğer terimlerle kesişim al (AND mantığı gibi)
                for i in range(1, len(processed_user_query_terms)):
                    if processed_user_query_terms[i] in inv_index.index:
                        current_relevant_set.intersection_update(inv_index.get_postings_list(processed_user_query_terms[i]))
                    else: # AND mantığında bir terim yoksa sonuç boştur
                        current_relevant_set = set()
                        break
                pseudo_relevant_docs_dynamic = current_relevant_set
        
        print(f"  (Dinamik sorgu için 'pseudo' ilgili doküman sayısı: {len(pseudo_relevant_docs_dynamic)})")
        #------------------------------------------------------------------------------------

        # Boolean Arama
        bool_operator = input("Boolean operatörü (AND/OR) [Varsayılan: AND]: ").upper()
        if bool_operator not in ['AND', 'OR']:
            bool_operator = 'AND' 
        
        bool_results_user = search_engine.boolean_search(user_query, operator=bool_operator)
        print(f"\nBoolean ({bool_operator}) Arama Sonuçları ({len(bool_results_user)} doküman):")
        if not bool_results_user:
            print("  Sonuç bulunamadı.")
        for i, doc_id in enumerate(bool_results_user[:K_FOR_DYNAMIC_PRF]):
            print(f"  {i+1}. {doc_id} - İçerik (ilk 70 krk): {raw_documents_all.get(doc_id, '')[:70]}...")
        if len(bool_results_user) > K_FOR_DYNAMIC_PRF: print("  ...")

        # Dinamik Boolean için P, R, F1
        if pseudo_relevant_docs_dynamic: # Eğer pseudo GT varsa hesapla
            p_bool_dyn, r_bool_dyn, f1_bool_dyn = precision_recall_f1(
                bool_results_user[:K_FOR_DYNAMIC_PRF], 
                pseudo_relevant_docs_dynamic
            )
            print(f"  Boolean ({bool_operator}) - P@{K_FOR_DYNAMIC_PRF}: {p_bool_dyn:.4f}, R (based on top {K_FOR_DYNAMIC_PRF}): {r_bool_dyn:.4f}, F1: {f1_bool_dyn:.4f}")
        else:
            print(f"  Boolean ({bool_operator}) - P, R, F1 hesaplanamadı (pseudo ilgili doküman yok).")


        # TF-IDF Arama
        tfidf_results_user_scores = search_engine.tfidf_rank(user_query, top_n=K_FOR_DYNAMIC_PRF)
        tfidf_results_user_ids = [doc_id for doc_id, score in tfidf_results_user_scores]

        print(f"\nTF-IDF Sıralı Arama Sonuçları (ilk {len(tfidf_results_user_scores)}):")
        if not tfidf_results_user_scores:
            print("  Sonuç bulunamadı.")
        for i, (doc_id, score) in enumerate(tfidf_results_user_scores): # Zaten top_n ile K_FOR_DYNAMIC_PRF kadar gelecek
            print(f"  {i+1}. {doc_id} (Skor: {score:.4f}) - İçerik (ilk 70 krk): {raw_documents_all.get(doc_id, '')[:70]}...")
        
        # Dinamik TF-IDF için P, R, F1 ve AP
        if pseudo_relevant_docs_dynamic:
            p_tfidf_dyn, r_tfidf_dyn, f1_tfidf_dyn = precision_recall_f1(
                tfidf_results_user_ids, # Zaten K_FOR_DYNAMIC_PRF kadar
                pseudo_relevant_docs_dynamic
            )
            print(f"  TF-IDF - P@{K_FOR_DYNAMIC_PRF}: {p_tfidf_dyn:.4f}, R (based on top {K_FOR_DYNAMIC_PRF}): {r_tfidf_dyn:.4f}, F1: {f1_tfidf_dyn:.4f}")
            
            # AP için tüm sıralı listeyi (veya daha uzun bir K'yı) almak daha iyi olurdu,
            # ama pseudo GT ile sadece ilk K üzerinden AP de bir fikir verebilir.
            # Daha doğru AP için tfidf_rank'ı daha büyük top_n ile çağırıp,
            # o listeyi AP fonksiyonuna vermek gerekir. Şimdilik K üzerinden yapalım.
            ap_tfidf_dyn = average_precision(tfidf_results_user_ids, pseudo_relevant_docs_dynamic)
            print(f"  TF-IDF - Average Precision (AP@{K_FOR_DYNAMIC_PRF}): {ap_tfidf_dyn:.4f}")
        else:
            print(f"  TF-IDF - P, R, F1, AP hesaplanamadı (pseudo ilgili doküman yok).")
        user_query = input("\nSorgunuzu girin (çıkmak için 'exit' yazın): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            continue

        # Boolean Arama
        bool_operator = input("Boolean operatörü (AND/OR) [Varsayılan: AND]: ").upper()
        if bool_operator not in ['AND', 'OR']:
            bool_operator = 'AND' 
        
        bool_results_user = search_engine.boolean_search(user_query, operator=bool_operator)
        print(f"\nBoolean ({bool_operator}) Arama Sonuçları ({len(bool_results_user)} doküman):")
        if not bool_results_user:
            print("  Sonuç bulunamadı.")
        for i, doc_id in enumerate(bool_results_user[:10]):
            print(f"  {i+1}. {doc_id} - İçerik (ilk 70 krk): {raw_documents_all.get(doc_id, '')[:70]}...")
        if len(bool_results_user) > 10: print("  ...")

        # TF-IDF Arama
        tfidf_results_user = search_engine.tfidf_rank(user_query, top_n=10)
        print(f"\nTF-IDF Sıralı Arama Sonuçları (ilk {len(tfidf_results_user)}):")
        if not tfidf_results_user:
            print("  Sonuç bulunamadı.")
        for i, (doc_id, score) in enumerate(tfidf_results_user):
            print(f"  {i+1}. {doc_id} (Skor: {score:.4f}) - İçerik (ilk 70 krk): {raw_documents_all.get(doc_id, '')[:70]}...")
        if len(tfidf_results_user) > 10: print("  ...") # Bu zaten top_n=10 ile sınırlı

if __name__ == '__main__':
    main()
    # main.py dosyasının sonlarına doğru (mevcut main() fonksiyonunun sonrasında veya içinde)

# ... (main fonksiyonunun geri kalanı) ...

def initialize_search_engine():
    # Bu fonksiyon main() içindeki adımların bir özeti olacak
    # ve app.py tarafından çağrılacak.
    
    IMDB_DATA_PATH = r'data' # VEYA SİZİN DOĞRU MUTLAK YOLUNUZ
    print("IMDb dokümanları yükleniyor (Bu işlem biraz zaman alabilir)...")
    raw_documents_all = load_documents_imdb(IMDB_DATA_PATH)
    if not raw_documents_all:
        print("HATA: Dokümanlar yüklenemedi. app.py başlatılamıyor.")
        return None, None, None # Hata durumunda None döndür

    print(f"{len(raw_documents_all)} doküman başarıyla yüklendi.")

    print("\nMetinler ön işleniyor ve kelime dağarcığı oluşturuluyor...")
    # Rapor detayını app.py'de göstermeyeceğiz, o yüzden sample_size'ı küçük tutabiliriz veya raporu kapatabiliriz.
    # Veya create_vocabulary_report_detailed'ın rapor basmayan bir versiyonunu kullanabiliriz.
    # Şimdilik utils.py'deki preprocess_text'i doğrudan çağıralım.
    
    # HATALI SATIR:
    # processed_documents_all = {
    #     doc_id: preprocess_text(text, use_stemming=True, use_lemmatization=False) 
    #     for doc_id, text_content in raw_documents_all.items() # text_content olmalıydı, text değil
    # }

    # DÜZELTİLMİŞ SATIR:
    processed_documents_all = {
        doc_id: preprocess_text(text_content, use_stemming=True, use_lemmatization=False) 
        for doc_id, text_content in raw_documents_all.items() # 'text' yerine 'text_content' kullanılıyor.
    }
    # Basit bir sözlük alalım (app.py'de lazım olmayabilir ama InvertedIndex için gerekebilir)
    # final_vocabulary = sorted(list(set(token for tokens_list in processed_documents_all.values() for token in tokens_list)))

    # main.py -> initialize_search_engine fonksiyonu içinde

# ... (önceki kodlar, processed_documents_all oluşturulduktan sonra) ...

    print("Ön işleme tamamlandı.")

    # --- Ters İndeks Oluşturma ---
    print("\nTers indeks oluşturuluyor...")
    inv_index = InvertedIndex() # inv_index burada tanımlanıyor
    inv_index.build_index(processed_documents_all)
    print("Ters indeks oluşturma tamamlandı.")

    # --- Arama Motoru Oluşturma ---
    # HATALI YERDEKİ SATIR BURADA OLMALIYDI:
    # search_engine_instance = SearchEngine(inv_index) # inv_index şimdi tanımlı

    # Bu satır doğru yerde, inv_index tanımlandıktan sonra SearchEngine oluşturuluyor.
    search_engine_instance = SearchEngine(inv_index) 
    
    # Fonksiyonun sonunda döndürülen değerler
    return search_engine_instance, raw_documents_all, inv_index

# Eğer main.py'yi doğrudan çalıştırdığınızda hala değerlendirme vb. yapmak istiyorsanız:
if __name__ == '__main__':
    # main() fonksiyonu içindeki değerlendirme ve raporlama kısımları burada kalabilir.
    # Ya da main() fonksiyonu sadece initialize_search_engine'i çağırıp bir şey yapmayabilir
    # ve tüm arayüz app.py'den çalıştırılır.
    # Şimdilik, main.py'nin eski işlevini koruduğunu varsayalım
    # ve app.py'nin initialize_search_engine'i ayrıca çağıracağını düşünelim.
    
    # Önceki main() fonksiyonunuzun içeriğini burada çalıştırabilirsiniz (arayüz kısmı hariç)
    # ... (Bölüm I, II, III, IV, V değerlendirme kodlarınız) ...
    
    print("\nAna veri işleme ve değerlendirme tamamlandı.")
    print("Arayüzü çalıştırmak için 'python app.py' komutunu kullanın.")

    # VEYA main() fonksiyonunu doğrudan çağırabilirsiniz:
    # main() # Bu, eski main fonksiyonunuzun tüm adımlarını çalıştırır (arayüz hariç)