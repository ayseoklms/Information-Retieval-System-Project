# app.py

from utils import preprocess_text
from search import SearchEngine
from main import initialize_search_engine
from colorama import Fore, Back, Style, init as colorama_init

# colorama'yı başlat (Windows'ta ANSI desteği için önemli)
colorama_init(autoreset=True) # autoreset=True her print'ten sonra stili sıfırlar

def run_app(search_engine, raw_documents):
    """
    Kullanıcı arayüzünü çalıştıran ana fonksiyon.
    """
    if not search_engine or not raw_documents:
        print(Fore.RED + "HATA: Arama motoru veya dokümanlar yüklenemedi. Arayüz başlatılamıyor.")
        return

    print(Fore.CYAN + Style.BRIGHT + "\n\n======================================================")
    print(Fore.CYAN + Style.BRIGHT + "      BASİT BİLGİ ERİŞİM SİSTEMİ ARAYÜZÜ")
    print(Fore.CYAN + Style.BRIGHT + "======================================================" + Style.RESET_ALL)

    K_FOR_DISPLAY = 10

    while True:
        print(Fore.YELLOW + "\nYapmak istediğiniz işlem nedir?")
        print(Fore.GREEN + "1: " + Style.RESET_ALL + "Boolean Arama")
        print(Fore.GREEN + "2: " + Style.RESET_ALL + "TF-IDF Sıralı Arama")
        print(Fore.RED + "exit: " + Style.RESET_ALL + "Çıkış")
        
        choice = input(Fore.WHITE + "Seçiminiz (1, 2, exit): " + Style.RESET_ALL).strip().lower()

        if choice == 'exit':
            print(Fore.MAGENTA + "Programdan çıkılıyor...")
            break
        
        if choice not in ['1', '2']:
            print(Fore.RED + "Geçersiz seçim. Lütfen 1, 2 veya exit girin.")
            continue

        user_query = input(Fore.WHITE + "Lütfen arama sorgunuzu girin: " + Style.RESET_ALL).strip()
        if not user_query:
            print(Fore.RED + "Sorgu boş olamaz.")
            continue

        print(Fore.BLUE + "-" * 50)

        if choice == '1': # Boolean Arama
            print(Fore.MAGENTA + Style.BRIGHT + f"Boolean Arama için Sorgu: '{user_query}'" + Style.RESET_ALL)
            bool_operator_choice = input(Fore.WHITE + "Operatör seçin (AND / OR) [Varsayılan: AND]: " + Style.RESET_ALL).strip().upper()
            if bool_operator_choice not in ['AND', 'OR']:
                print(Fore.YELLOW + "Geçersiz operatör. Varsayılan olarak AND kullanılacak.")
                bool_operator = 'AND'
            else:
                bool_operator = bool_operator_choice
            
            boolean_results = search_engine.boolean_search(user_query, operator=bool_operator)
            print(Fore.GREEN + Style.BRIGHT + f"\n>>> Boolean ({bool_operator}) Sonuçları ({len(boolean_results)} doküman bulundu):" + Style.RESET_ALL)
            if not boolean_results:
                print(Fore.YELLOW + "   Bu sorgu için sonuç bulunamadı.")
            else:
                for i, doc_id in enumerate(boolean_results[:K_FOR_DISPLAY]):
                    doc_content_snippet = raw_documents.get(doc_id, "İçerik bulunamadı.")[:100]
                    print(Fore.CYAN + f"  {i+1}. Doküman ID: {doc_id}")
                    print(Style.DIM + f"     İçerik (Önizleme): {doc_content_snippet}..." + Style.RESET_ALL) # DIM ile içeriği soluk yapabiliriz
                    print(Fore.BLUE + "-" * 30)
                if len(boolean_results) > K_FOR_DISPLAY:
                    print(Fore.YELLOW + f"   ... (ve {len(boolean_results) - K_FOR_DISPLAY} daha fazla sonuç)")
        
        elif choice == '2': # TF-IDF Sıralı Arama
            print(Fore.MAGENTA + Style.BRIGHT + f"TF-IDF Sıralı Arama için Sorgu: '{user_query}'" + Style.RESET_ALL)
            tfidf_results_with_scores = search_engine.tfidf_rank(user_query, top_n=K_FOR_DISPLAY)
            
            print(Fore.GREEN + Style.BRIGHT + f"\n>>> TF-IDF Sıralı Sonuçlar (En iyi {len(tfidf_results_with_scores)} gösteriliyor):" + Style.RESET_ALL)
            if not tfidf_results_with_scores:
                print(Fore.YELLOW + "   Bu sorgu için sonuç bulunamadı.")
            else:
                for i, (doc_id, score) in enumerate(tfidf_results_with_scores):
                    doc_content_snippet = raw_documents.get(doc_id, "İçerik bulunamadı.")[:100]
                    print(Fore.CYAN + f"  {i+1}. Doküman ID: {doc_id}" + Fore.YELLOW + f" (Skor: {score:.4f})")
                    print(Style.DIM + f"     İçerik (Önizleme): {doc_content_snippet}..." + Style.RESET_ALL)
                    print(Fore.BLUE + "-" * 30)
        
        print(Fore.BLUE + "=" * 50 + "\n" + Style.RESET_ALL)

if __name__ == '__main__':
    print(Fore.BLUE + "Bilgi Erişim Sistemi başlatılıyor... Lütfen bekleyin." + Style.RESET_ALL)
    search_engine_instance, raw_docs, inv_idx_instance = initialize_search_engine() 
    
    if search_engine_instance and raw_docs:
        run_app(search_engine_instance, raw_docs)
    else:
        print(Fore.RED + Style.BRIGHT + "Başlatma sırasında bir hata oluştu. Arayüz çalıştırılamıyor." + Style.RESET_ALL)