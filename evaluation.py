import numpy as np

def precision_recall_f1(retrieved_doc_ids, relevant_doc_ids):
    """
    Precision, Recall ve F1 skorlarını hesaplar.
    retrieved_doc_ids: Arama motorunun döndürdüğü doküman ID'leri (set veya list).
    relevant_doc_ids: Ground truth'daki ilgili doküman ID'leri (set veya list).
    """
    retrieved_set = set(retrieved_doc_ids)
    relevant_set = set(relevant_doc_ids)

    if not relevant_set: 
        return 0.0, 0.0, 0.0

    true_positives = len(retrieved_set.intersection(relevant_set))
    
    precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
    recall = true_positives / len(relevant_set) 
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def average_precision(retrieved_ranked_doc_ids, relevant_doc_ids):
    """
    Bir sorgu için Average Precision (AP) hesaplar.
    retrieved_ranked_doc_ids: Arama motorunun sıralı olarak döndürdüğü doküman ID'leri (list).
    relevant_doc_ids: Ground truth'daki ilgili doküman ID'leri (set veya list).
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    num_relevant_found_so_far = 0

    for i, doc_id in enumerate(retrieved_ranked_doc_ids):
        if doc_id in relevant_set:
            hits += 1
            num_relevant_found_so_far +=1
            precision_at_k = num_relevant_found_so_far / (i + 1)
            sum_precisions += precision_at_k
  
    return sum_precisions / len(relevant_set) if len(relevant_set) > 0 else 0.0


def mean_average_precision(all_query_results_ranked_ids, all_relevant_docs_sets):
    """
    Birden fazla sorgu için Mean Average Precision (MAP) hesaplar.
    all_query_results_ranked_ids: Her sorgu için sıralı doküman ID listelerini içeren bir liste.
                                 [[q1_doc1, q1_doc2,...], [q2_doc1, q2_doc2,...]]
    all_relevant_docs_sets: Her sorgu için ilgili doküman ID setlerini içeren bir liste.
                           [{q1_rel1, q1_rel2,...}, {q2_rel1, q2_rel2,...}]
    """
    if len(all_query_results_ranked_ids) != len(all_relevant_docs_sets):
        raise ValueError("Sorgu sonuçları ve ilgili doküman listeleri aynı sayıda olmalı.")

    ap_scores = []
    for i in range(len(all_query_results_ranked_ids)):
        retrieved_ranked_doc_ids = all_query_results_ranked_ids[i]
        relevant_doc_ids_set = all_relevant_docs_sets[i]
        if not relevant_doc_ids_set: 
            ap_scores.append(0.0)
            continue
        ap = average_precision(retrieved_ranked_doc_ids, relevant_doc_ids_set)
        ap_scores.append(ap)
        
    return np.mean(ap_scores) if ap_scores else 0.0


if __name__ == '__main__':
    retrieved1_ranked = ['doc1', 'doc_missing', 'doc3', 'doc4', 'doc5', 'doc6']
    relevant1_set = {'doc1', 'doc3', 'doc6', 'doc7'} # doc7 bulunamadı
    
    p, r, f1 = precision_recall_f1(retrieved1_ranked, relevant1_set)
    print(f"Query 1 (All retrieved) - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

    # P, R, F1 için genellikle ilk K sonuç değerlendirilir (örn P@10)
    p_at_5, r_at_5, f1_at_5 = precision_recall_f1(retrieved1_ranked[:5], relevant1_set)
    print(f"Query 1 (Top 5) - P@5: {p_at_5:.4f}, R (basierend auf Top 5): {r_at_5:.4f}, F1: {f1_at_5:.4f}")


    ap1 = average_precision(retrieved1_ranked, relevant1_set)
    print(f"Query 1 - Average Precision: {ap1:.4f}") # (1/1 + 2/3 + 3/6) / 4 = (1 + 0.666 + 0.5) / 4 = 2.166 / 4 = 0.5415

    retrieved2_ranked = ['docA', 'docB', 'docC'] # docD (relevant) bulunamadı
    relevant2_set = {'docA', 'docD'}
    ap2 = average_precision(retrieved2_ranked, relevant2_set) # (1/1 + 0) / 2 = 0.5
    print(f"\nQuery 2 - Average Precision: {ap2:.4f}")

    map_score = mean_average_precision([retrieved1_ranked, retrieved2_ranked], [relevant1_set, relevant2_set])
    print(f"\nMean Average Precision (MAP): {map_score:.4f}") # (0.5415 + 0.5) / 2 = 0.52075