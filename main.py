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
    content5 = ''' 以色列国防军20日对加沙地带实施轰炸，造成3名巴勒斯坦武装人员死亡。此外，巴勒斯坦人与以色列士兵当天在加沙地带与以交界地区发生冲突，一名巴勒斯坦人被打死。当天的冲突还造成210名巴勒斯坦人受伤。
    当天，数千名巴勒斯坦人在加沙地带边境地区继续“回归大游行”抗议活动。部分示威者燃烧轮胎，并向以军投掷石块、燃烧瓶等，驻守边境的以军士兵向示威人群发射催泪瓦斯并开枪射击。'''
    svos = extractor.triples_main(content=content5)
    print(svos)