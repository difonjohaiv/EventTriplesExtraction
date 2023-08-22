from triple_extraction import TripleExtractor
from zhon.hanzi import punctuation
import string
import pandas as pd
from tqdm import tqdm
import torch


def  event_filter(x_list):
    with torch.no_grad():
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


def start_extracting(x, extractor):
    with torch.no_grad():
        print(f"正在處理第_{x}_個文件ing")
        fn = f"dataset/02/news_train_2_{x}.csv"
        df = pd.read_csv(fn)
        df['event'] = None
        df['signal'] = None
        save = f"dataset/02/news_train_2_{x}_entity_keyword_event.csv"
        for item in tqdm(range(len(df))):
            svos = extractor.triples_main(df['content'][item])
            svos = event_filter(svos)
            # 處理玩之後呢，事件已經由三元組變成sentence，sentence組成的列表
            if len(svos) < 10:  # 事件數量不夠10，就丟掉
                df['signal'][item] == 0
                continue
            e_list = "@".join(svos)  # 這裏是把所有的事件變成str存儲
            df['event'][item] = e_list
            df['signal'] == 1
        df.to_csv(save, index=False)


if __name__ == '__main__':
    with torch.no_grad():
        extractor = TripleExtractor()
        index_list = [1]
        for item in index_list:
            start_extracting(x=item, extractor=extractor)
        # start_extracting(extractor=extractor)