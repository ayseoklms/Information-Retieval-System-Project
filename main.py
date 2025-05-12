# main.py

import os
import random 

from utils import (
    load_documents_imdb,
    preprocess_text,
    create_vocabulary_report_detailed
)
from inverted_index import InvertedIndex
from search import SearchEngine
from evaluation import (
    precision_recall_f1,
    average_precision,
    mean_average_precision
)
from colorama import Fore, Style, init as colorama_init 

colorama_init(autoreset=True) 

def run_project_and_interactive_demo():
    """
    Proje dökümanındaki tüm gereksinimleri (Bölüm I-V) sırayla çalıştırır,
    önceden tanımlanmış test sorguları ve ground truth ile değerlendirme yapar,
    çıktıları konsola üretir ve ardından interaktif bir sorgulama modu başlatır.
    """

    # --- VERİ YÜKLEME ---
    IMDB_DATA_PATH = r'C:\Users\ayseo\OneDrive\Masaüstü\bilgi-erisim-sistemleri-proje\data' 
    
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "PROJE BAŞLATILIYOR: IMDb DOKÜMANLARI YÜKLENİYOR...")
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)
    raw_documents_all = load_documents_imdb(IMDB_DATA_PATH)
    if not raw_documents_all:
        print(Fore.RED + Style.BRIGHT + "HATA: Doküman yükleme başarısız. Lütfen IMDB_DATA_PATH ve veri seti konumunu kontrol edin." + Style.RESET_ALL)
        return
    print(Fore.GREEN + Style.BRIGHT + f"\n{len(raw_documents_all)} doküman başarıyla yüklendi.\n" + Style.RESET_ALL)

    # --- BÖLÜM I: TERİMLERİN KELİME DAĞARCIĞININ BELİRLENMESİ ---
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "BÖLÜM I: KELİME DAĞARCIĞI OLUŞTURMA VE METİN ÖN İŞLEME")
    print("Bu işlem veri setinin büyüklüğüne göre biraz zaman alabilir...")
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)
    final_vocabulary, processed_documents_all = create_vocabulary_report_detailed(
        raw_documents_all, 
        sample_size=2
    )
    print(Fore.GREEN + Style.BRIGHT + "\nBölüm I tamamlandı.\n" + Style.RESET_ALL)

    # --- BÖLÜM II: TERS İNDEKS YAPISI KURMA ---
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "BÖLÜM II: TERS İNDEKS OLUŞTURMA")
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)
    inv_index = InvertedIndex()
    inv_index.build_index(processed_documents_all)
    
    if final_vocabulary:
        report_term_example_idx = len(final_vocabulary) // 3
        report_term_example = final_vocabulary[report_term_example_idx] if len(final_vocabulary) > report_term_example_idx else None
        if report_term_example:
            print(Fore.YELLOW + "\n--- Örnek Ters İndeks Kaydı (Rapor için) ---" + Style.RESET_ALL)
            # inv_index.get_term_data_for_report çıktısı zaten string, doğrudan print edilebilir
            print(inv_index.get_term_data_for_report(report_term_example)) 
    print(Fore.GREEN + Style.BRIGHT + "\nBölüm II tamamlandı.\n" + Style.RESET_ALL)

    # --- ARAMA MOTORU NESNESİNİ OLUŞTUR ---
    search_engine = SearchEngine(inv_index)

    # --- TEST SORGULARI VE GROUND TRUTH (MANUEL OLUŞTURULMALI!) ---
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "BÖLÜM V İÇİN TEST SORGULARI VE GROUND TRUTH TANIMLAMA")
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)
    all_doc_ids_list = list(raw_documents_all.keys())
    
    example_queries_ground_truth = {
        "amazing suspense thriller movie": {'doc_1818', 'doc_35507', 'doc_9005', 'doc_1815', 'doc_37614'},
        "classic love story romance": {'doc_27888', 'doc_2585', 'doc_26942', 'doc_33857', 'doc_43'},
        "funny comedy hilarious actor": {'doc_36910', 'doc_9209', 'doc_3510', 'doc_1709', 'doc_30708'},
        "world war two documentary germany": {'doc_12000', 'doc_12001', 'doc_28000', 'doc_28005', 'doc_41000'}, # TAMAMEN ÖRNEK ID'LER!
        "space alien invasion film": {'doc_3000', 'doc_13001', 'doc_18765', 'doc_23456', 'doc_47654'}  # TAMAMEN ÖRNEK ID'LER!
    }
    print(Fore.RED + Style.BRIGHT + "UYARI: Yukarıdaki Ground Truth setleri örnektir. En iyi sonuçlar için MANUEL olarak oluşturulmalıdır!" + Style.RESET_ALL)
    print("Tanımlanmış Ground Truth Setleri:\n")
    for q_text, gt_set in example_queries_ground_truth.items():
        print(Fore.MAGENTA + f"Sorgu: '{q_text}'" + Style.RESET_ALL)
        print(f"  İlgili Dokümanlar (Ground Truth): {list(gt_set)[:5]}...")
    print(Fore.BLUE + "-" * 50 + Style.RESET_ALL)

    K_FOR_PRF_EVAL = 10 

    all_boolean_and_retrieved_for_map = []
    all_boolean_or_retrieved_for_map = []
    all_tfidf_retrieved_ranked_for_map = []
    all_relevant_sets_for_map_eval = []

    # --- BÖLÜM III: BOOLE ERİŞİMİ (Sonuçlar) ---
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "BÖLÜM III: BOOLE ERİŞİMİ SONUÇLARI")
    print(Fore.YELLOW + "Proje Dökümanı Gereksinim 3: 'Aşağıda belirtilen sorgular için Boolean retrieval sonuçlarını gösteriniz...'" + Style.RESET_ALL)
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)

    for query_text, relevant_docs_gt_set in example_queries_ground_truth.items():
        print(Fore.MAGENTA + f"\n--- Sorgu: '{query_text}' ---" + Style.RESET_ALL)
        
        retrieved_bool_and = search_engine.boolean_search(query_text, operator='AND')
        print(Fore.GREEN + f"\n  Boolean (AND) Sonuçları:" + Style.RESET_ALL)
        print(f"    Alınan Doküman Sayısı: {len(retrieved_bool_and)}")
        print(f"    Doküman Kimlikleri (ilk {K_FOR_PRF_EVAL}): {retrieved_bool_and[:K_FOR_PRF_EVAL]}")
        if len(retrieved_bool_and) > K_FOR_PRF_EVAL: print("    ...")
        if relevant_docs_gt_set: all_boolean_and_retrieved_for_map.append(retrieved_bool_and)

        retrieved_bool_or = search_engine.boolean_search(query_text, operator='OR')
        print(Fore.GREEN + f"\n  Boolean (OR) Sonuçları:" + Style.RESET_ALL)
        print(f"    Alınan Doküman Sayısı: {len(retrieved_bool_or)}")
        print(f"    Doküman Kimlikleri (ilk {K_FOR_PRF_EVAL}): {retrieved_bool_or[:K_FOR_PRF_EVAL]}")
        if len(retrieved_bool_or) > K_FOR_PRF_EVAL: print("    ...")
        if relevant_docs_gt_set: all_boolean_or_retrieved_for_map.append(retrieved_bool_or)
        
        if relevant_docs_gt_set: all_relevant_sets_for_map_eval.append(relevant_docs_gt_set)

    print(Fore.GREEN + Style.BRIGHT + "\nBölüm III tamamlandı.\n" + Style.RESET_ALL)

    # --- BÖLÜM IV: TF-IDF (Sonuçlar) ---
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "BÖLÜM IV: TF-IDF SIRALI ERİŞİM SONUÇLARI")
    print(Fore.YELLOW + "Proje Dökümanı Gereksinim 4: '...aynı sorgu seti için TF-IDf puanlamasını kullanarak belgeleri çıkarınız...'" + Style.RESET_ALL)
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)

    for query_text, relevant_docs_gt_set in example_queries_ground_truth.items():
        print(Fore.MAGENTA + f"\n--- Sorgu: '{query_text}' ---" + Style.RESET_ALL)
        tfidf_ranked_scores_full = search_engine.tfidf_rank(query_text, top_n=len(raw_documents_all))
        
        print(Fore.GREEN + f"\n  TF-IDF Sıralı Sonuçlar (TF-IDF ağırlıklarına göre, ilk {K_FOR_PRF_EVAL} gösteriliyor):" + Style.RESET_ALL)
        if not tfidf_ranked_scores_full: 
            print("    Hiç doküman bulunamadı.")
        else:
            for i, (doc_id, score) in enumerate(tfidf_ranked_scores_full[:K_FOR_PRF_EVAL]):
                 print(f"    {i+1}. Doküman ID: {doc_id} (Skor: {score:.4f}) - Başlık (ilk 50 krk): {raw_documents_all.get(doc_id, 'N/A')[:50]}...")
        
        if relevant_docs_gt_set:
            tfidf_retrieved_ranked_ids = [doc_id for doc_id, score in tfidf_ranked_scores_full]
            all_tfidf_retrieved_ranked_for_map.append(tfidf_retrieved_ranked_ids)

    print(Fore.GREEN + Style.BRIGHT + "\nBölüm IV tamamlandı.\n" + Style.RESET_ALL)

    # --- BÖLÜM V: DEĞERLENDİRME (Metrik Hesaplamaları) ---
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "BÖLÜM V: SİSTEM DEĞERLENDİRMESİ (METRİKLER)")
    # ... (Proje dökümanı gereksinimleri metni) ...
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)
    print(Fore.YELLOW + f"\nDeğerlendirme Metrikleri (ilk {K_FOR_PRF_EVAL} sonuç veya tüm sıralı liste üzerinden):\n" + Style.RESET_ALL)

    idx_counter = 0
    for query_text, relevant_docs_gt_set in example_queries_ground_truth.items():
        if not relevant_docs_gt_set: continue 

        print(Fore.MAGENTA + f"\n--- Sorgu Değerlendirmesi: '{query_text}' ---" + Style.RESET_ALL)
        print(f"  Ground Truth İlgili Doküman Sayısı: {len(relevant_docs_gt_set)}")

        retrieved_b_and = all_boolean_and_retrieved_for_map[idx_counter] 
        p_b_and, r_b_and, f1_b_and = precision_recall_f1(retrieved_b_and[:K_FOR_PRF_EVAL], relevant_docs_gt_set)
        ap_b_and = average_precision(retrieved_b_and, relevant_docs_gt_set)
        print(Fore.GREEN + f"  Boolean (AND):" + Style.RESET_ALL)
        print(f"    P@{K_FOR_PRF_EVAL}: {p_b_and:.4f}, R@{K_FOR_PRF_EVAL}: {r_b_and:.4f}, F1@{K_FOR_PRF_EVAL}: {f1_b_and:.4f}")
        print(f"    Average Precision (AP): {ap_b_and:.4f}")

        retrieved_b_or = all_boolean_or_retrieved_for_map[idx_counter]
        p_b_or, r_b_or, f1_b_or = precision_recall_f1(retrieved_b_or[:K_FOR_PRF_EVAL], relevant_docs_gt_set)
        ap_b_or = average_precision(retrieved_b_or, relevant_docs_gt_set)
        print(Fore.GREEN + f"  Boolean (OR):" + Style.RESET_ALL)
        print(f"    P@{K_FOR_PRF_EVAL}: {p_b_or:.4f}, R@{K_FOR_PRF_EVAL}: {r_b_or:.4f}, F1@{K_FOR_PRF_EVAL}: {f1_b_or:.4f}")
        print(f"    Average Precision (AP): {ap_b_or:.4f}")
        
        retrieved_tfidf = all_tfidf_retrieved_ranked_for_map[idx_counter]
        p_tfidf, r_tfidf, f1_tfidf = precision_recall_f1(retrieved_tfidf[:K_FOR_PRF_EVAL], relevant_docs_gt_set)
        ap_tfidf = average_precision(retrieved_tfidf, relevant_docs_gt_set)
        print(Fore.GREEN + f"  TF-IDF:" + Style.RESET_ALL)
        print(f"    P@{K_FOR_PRF_EVAL}: {p_tfidf:.4f}, R@{K_FOR_PRF_EVAL}: {r_tfidf:.4f}, F1@{K_FOR_PRF_EVAL}: {f1_tfidf:.4f}")
        print(f"    Average Precision (AP): {ap_tfidf:.4f}")
        idx_counter += 1

    print(Fore.YELLOW + "\n--- Genel Ortalama Metrikler (Tüm Değerlendirilen Sorgular Üzerinden) ---" + Style.RESET_ALL)
    if all_relevant_sets_for_map_eval:
        if len(all_boolean_and_retrieved_for_map) == len(all_relevant_sets_for_map_eval):
            map_boolean_and = mean_average_precision(all_boolean_and_retrieved_for_map, all_relevant_sets_for_map_eval)
            print(Fore.GREEN + f"  Genel MAP (Boolean AND): {map_boolean_and:.4f}" + Style.RESET_ALL)
        else: print(Fore.RED + "  MAP (Boolean AND) hesaplanamadı (liste boyutları uyuşmuyor)." + Style.RESET_ALL)
    
        if len(all_boolean_or_retrieved_for_map) == len(all_relevant_sets_for_map_eval):
            map_boolean_or = mean_average_precision(all_boolean_or_retrieved_for_map, all_relevant_sets_for_map_eval)
            print(Fore.GREEN + f"  Genel MAP (Boolean OR): {map_boolean_or:.4f}" + Style.RESET_ALL)
        else: print(Fore.RED + "  MAP (Boolean OR) hesaplanamadı (liste boyutları uyuşmuyor)." + Style.RESET_ALL)

        if len(all_tfidf_retrieved_ranked_for_map) == len(all_relevant_sets_for_map_eval):
            map_tfidf = mean_average_precision(all_tfidf_retrieved_ranked_for_map, all_relevant_sets_for_map_eval)
            print(Fore.GREEN + f"  Genel MAP (TF-IDF): {map_tfidf:.4f}" + Style.RESET_ALL)
        else: print(Fore.RED + "  MAP (TF-IDF) hesaplanamadı (liste boyutları uyuşmuyor)." + Style.RESET_ALL)
    else:
        print(Fore.RED + "  Değerlendirilecek geçerli sorgu veya ground truth bulunamadığı için MAP hesaplanamadı." + Style.RESET_ALL)
    
    print(Fore.CYAN + Style.BRIGHT + "\nBölüm V ve Proje Raporlama Adımları Tamamlandı." + Style.RESET_ALL)
    print(Fore.CYAN + Style.BRIGHT + "="*80 + Style.RESET_ALL)

    # --- KULLANICIDAN DİNAMİK SORGU ALMA (İnteraktif Kısım) ---
    print(Fore.MAGENTA + Style.BRIGHT + "\n\n======================================================")
    print(Fore.MAGENTA + Style.BRIGHT + "      DİNAMİK SORGULAMA MODU BAŞLATILIYOR")
    print(Fore.MAGENTA + Style.BRIGHT + "======================================================" + Style.RESET_ALL)

    K_FOR_DYNAMIC_PRF = 10 # Dinamik sorgulama için P@K, R@K, F1@K için K değeri

    while True:
        print(Fore.YELLOW + "\nSorgu Türü Seçin:")
        print(Fore.GREEN + "1: " + Style.RESET_ALL + "Boolean Arama")
        print(Fore.GREEN + "2: " + Style.RESET_ALL + "TF-IDF Sıralı Arama")
        print(Fore.RED + "exit: " + Style.RESET_ALL + "Bu Moddan Çık (Program Sonlanır)")
        
        choice = input(Fore.WHITE + "Seçiminiz (1, 2, exit): " + Style.RESET_ALL).strip().lower()

        if choice == 'exit':
            print(Fore.MAGENTA + "Dinamik sorgulama modundan çıkılıyor... Program sonlandırıldı.")
            break
        
        if choice not in ['1', '2']:
            print(Fore.RED + "Geçersiz seçim. Lütfen 1, 2 veya exit girin.")
            continue

        user_query = input(Fore.WHITE + "Lütfen arama sorgunuzu girin: " + Style.RESET_ALL).strip()
        if not user_query:
            print(Fore.RED + "Sorgu boş olamaz.")
            continue

        print(Fore.BLUE + "-" * 50 + Style.RESET_ALL)

        # Dinamik sorgu için pseudo ground truth oluştur
        processed_user_query_terms = list(set(preprocess_text(user_query)))
        pseudo_relevant_docs_dynamic = set()
        if processed_user_query_terms:
            first_term_postings = inv_index.get_postings_list(processed_user_query_terms[0])
            if first_term_postings:
                current_relevant_set = set(first_term_postings)
                for i in range(1, len(processed_user_query_terms)):
                    term_postings = inv_index.get_postings_list(processed_user_query_terms[i])
                    if not term_postings:
                        current_relevant_set = set()
                        break
                    current_relevant_set.intersection_update(term_postings)
                pseudo_relevant_docs_dynamic = current_relevant_set
        
        print(Fore.BLUE + Style.DIM + f"  (Dinamik sorgu için 'pseudo' ilgili doküman sayısı: {len(pseudo_relevant_docs_dynamic)})" + Style.RESET_ALL)

        if choice == '1': # Boolean Arama
            print(Fore.MAGENTA + Style.BRIGHT + f"Boolean Arama için Sorgu: '{user_query}'" + Style.RESET_ALL)
            bool_operator_choice = input(Fore.WHITE + "Operatör seçin (AND / OR) [Varsayılan: AND]: " + Style.RESET_ALL).strip().upper()
            bool_operator = 'AND' if bool_operator_choice not in ['AND', 'OR'] else bool_operator_choice
            if bool_operator != bool_operator_choice: print(Fore.YELLOW + "Varsayılan olarak AND kullanıldı.")
            
            boolean_results = search_engine.boolean_search(user_query, operator=bool_operator)
            print(Fore.GREEN + Style.BRIGHT + f"\n>>> Boolean ({bool_operator}) Sonuçları ({len(boolean_results)} doküman bulundu):" + Style.RESET_ALL)
            if not boolean_results: print(Fore.YELLOW + "   Bu sorgu için sonuç bulunamadı.")
            else:
                for i, doc_id in enumerate(boolean_results[:K_FOR_DYNAMIC_PRF]):
                    print(Fore.CYAN + f"  {i+1}. {doc_id} - İçerik (ilk 70 krk): {raw_documents_all.get(doc_id, '')[:70]}...")
                if len(boolean_results) > K_FOR_DYNAMIC_PRF: print(Fore.YELLOW + f"   ... ({len(boolean_results) - K_FOR_DYNAMIC_PRF} daha fazla)")

            if pseudo_relevant_docs_dynamic or not processed_user_query_terms:
                p_b_dyn, r_b_dyn, f1_b_dyn = precision_recall_f1(boolean_results[:K_FOR_DYNAMIC_PRF], pseudo_relevant_docs_dynamic)
                print(Fore.YELLOW + f"  P@{K_FOR_DYNAMIC_PRF}: {p_b_dyn:.4f}, R@{K_FOR_DYNAMIC_PRF}: {r_b_dyn:.4f}, F1@{K_FOR_DYNAMIC_PRF}: {f1_b_dyn:.4f}" + Style.RESET_ALL)
            else: print(Fore.RED + "  P, R, F1 hesaplanamadı (pseudo GT yok)." + Style.RESET_ALL)
        
        elif choice == '2': # TF-IDF Sıralı Arama
            print(Fore.MAGENTA + Style.BRIGHT + f"TF-IDF Sıralı Arama için Sorgu: '{user_query}'" + Style.RESET_ALL)
            tfidf_results_scores = search_engine.tfidf_rank(user_query, top_n=K_FOR_DYNAMIC_PRF)
            tfidf_ids = [doc_id for doc_id, score in tfidf_results_scores]
            
            print(Fore.GREEN + Style.BRIGHT + f"\n>>> TF-IDF Sıralı Sonuçlar (ilk {len(tfidf_results_scores)}):" + Style.RESET_ALL)
            if not tfidf_results_scores: print(Fore.YELLOW + "   Bu sorgu için sonuç bulunamadı.")
            else:
                for i, (doc_id, score) in enumerate(tfidf_results_scores):
                    print(Fore.CYAN + f"  {i+1}. {doc_id} (Skor: {score:.4f}) - İçerik (ilk 70 krk): {raw_documents_all.get(doc_id, '')[:70]}...")

            if pseudo_relevant_docs_dynamic or not processed_user_query_terms:
                p_tf_dyn, r_tf_dyn, f1_tf_dyn = precision_recall_f1(tfidf_ids, pseudo_relevant_docs_dynamic)
                ap_tf_dyn = average_precision(tfidf_ids, pseudo_relevant_docs_dynamic)
                print(Fore.YELLOW + f"  P@{K_FOR_DYNAMIC_PRF}: {p_tf_dyn:.4f}, R@{K_FOR_DYNAMIC_PRF}: {r_tf_dyn:.4f}, F1@{K_FOR_DYNAMIC_PRF}: {f1_tf_dyn:.4f}" + Style.RESET_ALL)
                print(Fore.YELLOW + f"  Average Precision (AP@{K_FOR_DYNAMIC_PRF}): {ap_tf_dyn:.4f}" + Style.RESET_ALL)
            else: print(Fore.RED + "  P, R, F1, AP hesaplanamadı (pseudo GT yok)." + Style.RESET_ALL)
        
        print(Fore.BLUE + "=" * 50 + "\n" + Style.RESET_ALL)


if __name__ == '__main__':
    run_project_and_interactive_demo()