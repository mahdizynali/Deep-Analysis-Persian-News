from .tokenizer import MaZe_tokenizer
from .config import *

class DataPrep:
    '''کلاس آماده سازی اولیه داده ها برای آموزش'''
    def __init__(self, directory_path) -> None:

        self.categories = Persian_categories
        self.data_path = directory_path
        self.toke = MaZe_tokenizer()

    def data_process(self):
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        dataframes = []

        for file in csv_files:
            full_path = os.path.join(self.data_path, file)
            try:
                df = pd.read_csv(full_path, encoding='utf-8')
                    
                # فقط تایتل خبر و موضوع آن ذخیره شود
                if 'title' in df.columns and 'category' in df.columns:
                    df = df[['title', 'category']].copy()
                    
                    # حذف عنوانین اضافه و داده‌های نامعتبر
                    df = df[df['category'] != 'عکس']
                    df = df.dropna(subset=['title', 'category'])
                    
                    # پاکسازی و نرمال‌سازی دسته‌ها
                    df['category'] = df['category'].str.strip()
                    df['category'] = df['category'].map(self.categories)
                    
                    # حذف مواردی که در دسته‌بندی معتبر نیستند
                    df = df.dropna(subset=['category'])
                    df['category'] = df['category'].astype(int)
                    
                    dataframes.append(df)
                    
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
        
        if not dataframes:
            raise ValueError("No valid data found in the directory")
        
        # ترکیب و تصادفی‌سازی داده‌ها
        merged_df = pd.concat(dataframes, ignore_index=True)
        shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

        data = shuffled_df['title'].values
        labels = shuffled_df['category'].values

        # تقسیم داده‌ها با نسبت ۸۰ به ۲۰
        return train_test_split(data, labels, test_size=0.2, random_state=42)

    def freqs(self, text, y):
        """ساخت یک مپینگ از(خبر, موضوع)"""
        label = np.squeeze(y).tolist()
        freqs = {}

        for y, txt in zip(label, text):
            for word in self.toke.do_tokenize(txt):
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1
        # with open("freqs.txt", "+a") as tk:
        #     tk.write(str(freqs) + "\n")
        #     tk.close()
        return freqs
    
    def word_vocab(self, input):
        """ساخت یک دیکشنری از لغات و اختصاص یک شماره خاص به هر یک از آن ها"""
        
        Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 
        for word in input:
            for tk_word in self.toke.do_tokenize(word) :
                if tk_word not in Vocab: 
                    Vocab[tk_word] = len(Vocab)
        return Vocab