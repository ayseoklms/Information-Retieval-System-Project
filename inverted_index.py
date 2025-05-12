from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(lambda: {'postings': defaultdict(int), 'df': 0, 'total_corpus_freq': 0})
        self.documents_tokens = {} 
        self.doc_lengths = defaultdict(int) 
        self.total_docs = 0
        self.avg_doc_length = 0

    def build_index(self, processed_documents_dict):
        """
        processed_documents_dict: {doc_id: [token1, token2, ...]}
        """
       
        self.total_docs = len(processed_documents_dict)
        total_length_sum = 0

        for doc_id, tokens in processed_documents_dict.items():
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_length_sum += doc_len
            
            term_counts_in_doc = defaultdict(int)
            for token in tokens:
                term_counts_in_doc[token] += 1
            
            for term, tf_in_doc in term_counts_in_doc.items():
                self.index[term]['postings'][doc_id] = tf_in_doc
                self.index[term]['df'] += 1 
                self.index[term]['total_corpus_freq'] += tf_in_doc 
        
        if self.total_docs > 0:
            self.avg_doc_length = total_length_sum / self.total_docs
        
        print(f"Ters indeks {len(self.index)} terim ve {self.total_docs} doküman ile oluşturuldu.")
        print(f"Ortalama doküman uzunluğu: {self.avg_doc_length:.2f} terim.")

    def get_postings_list(self, term):
        """Bir terimin postings listesini (sadece doc_id'ler) döndürür: set(doc_id)."""
        return set(self.index.get(term, {}).get('postings', {}).keys())

    def get_postings_with_tf(self, term):
        """Bir terimin postings listesini (doc_id: tf) döndürür: dict."""
        return self.index.get(term, {}).get('postings', {})

    def get_df(self, term):
        """Bir terimin doküman frekansını (df) döndürür."""
        return self.index.get(term, {}).get('df', 0)

    def get_total_corpus_freq(self, term):
        """Bir terimin tüm korpustaki toplam frekansını döndürür."""
        return self.index.get(term, {}).get('total_corpus_freq', 0)

    def get_doc_length(self, doc_id):
        return self.doc_lengths.get(doc_id, 0)

    def get_vocabulary(self):
        return list(self.index.keys())
    
    def get_term_data_for_report(self, term):
        """Sözlükteki bir terim için rapor formatında veri döndürür."""
        term_info = self.index.get(term)
        if not term_info:
            return f"'{term}' terimi indekste bulunamadı."
        
        report_str = f"Terim: '{term}'\n"
        report_str += f"  Doküman Frekansı (df): {term_info['df']}\n"
        report_str += f"  Toplam Korpus Frekansı (Koleksiyondaki Toplam Geçiş Sayısı): {term_info['total_corpus_freq']}\n"
        report_str += "  İlanlar (Postings - Örnek ilk 5):\n"
        count = 0
        for doc_id, tf in term_info['postings'].items():
            if count < 5:
                report_str += f"    - Doküman ID: {doc_id}, Terim Sıklığı (tf): {tf}\n"
                count += 1
            else:
                report_str += "    ...\n"
                break
        return report_str


if __name__ == '__main__':
    from utils import load_documents_imdb, preprocess_text
    import os

    IMDB_DATA_PATH = r'C:\Users\ayseo\OneDrive\Masaüstü\bilgi-erisim-sistemleri-proje\data'
    
   
    raw_documents_full = load_documents_imdb(IMDB_DATA_PATH)
    if raw_documents_full:
        sample_doc_ids = list(raw_documents_full.keys())[:100] # İlk 100 doküman
        raw_documents_sample = {doc_id: raw_documents_full[doc_id] for doc_id in sample_doc_ids}

        processed_docs_sample = {
            doc_id: preprocess_text(text) for doc_id, text in raw_documents_sample.items()
        }

        inv_idx = InvertedIndex()
        inv_idx.build_index(processed_docs_sample)

        sample_vocab = inv_idx.get_vocabulary()
        if sample_vocab:
            test_term = sample_vocab[len(sample_vocab)//2] # Ortadan bir terim al
            print(f"\nTest Terimi: '{test_term}'")
            print(f"  DF: {inv_idx.get_df(test_term)}")
            print(f"  Total Corpus Freq: {inv_idx.get_total_corpus_freq(test_term)}")
            print(f"  Postings (with TF): {dict(list(inv_idx.get_postings_with_tf(test_term).items())[:3])}...")
            print(f"\n{inv_idx.get_term_data_for_report(test_term)}")
        else:
            print("Test için sözlük boş.")
    else:
        print("Doküman yüklenemedi.")