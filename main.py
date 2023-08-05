from triple_extraction import TripleExtractor
from zhon.hanzi import punctuation
import string
import pandas as pd
from tqdm import tqdm


def  event_filter(x_list):
    english_punctuation = string.punctuation
    chinese_punctuation =  punctuation
    y_list = []
    
    for x in x_list:
        if x[0] in english_punctuation or x[1] in english_punctuation or x[2] in english_punctuation:
            continue
        if x[0] in chinese_punctuation or x[1] in chinese_punctuation or x[2] in chinese_punctuation:
            continue
        if x[0] == '' or x[1] == '' or x[2] == '':
            continue
        e = "#".join(x)
        y_list.append(e)
    
    return y_list


def start_extracting(i, extractor):
    print(f"正在處理第_{i}_個文件ing")
    fn = f"dataset/news_train_{i}_entity_keyword.csv"
    df = pd.read_csv(fn)
    df['event'] = None
    df['signal'] = None
    save = f"dataset/news_train_{i}_entity_keyword_event.csv"
    for i in tqdm(range(len(df))):
        svos = extractor.triples_main(df['content'][i])
        svos = list(set(svos))
        svos = event_filter(svos)
        if len(svos) < 10:
            df['signal'][i] == 0
            continue
        e_list = "@".join(svos)
        df['event'][i] = e_list
        df['signal'] == 1
    df.to_csv(save, index=False)

if __name__ == '__main__':
    extractor = TripleExtractor()
    index_list = [1, 2, 3, 4, 5, 6, 7]
    for item in index_list:
        start_extracting(i=item, extractor=extractor)