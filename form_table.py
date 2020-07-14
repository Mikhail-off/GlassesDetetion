import pandas as pd
import numpy as np

LINKS_FILE = 'file.csv'
RESULT_FILE = 'result.csv'
DF_COLS=['link', 'class']


def main():
    df_links = pd.read_csv(LINKS_FILE)
    df_res = pd.read_csv(RESULT_FILE)
    res_map = dict()
    for name, class_id in df_res.values:
        res_map[name] = class_id

    df_all = pd.DataFrame(columns=DF_COLS)
    for name, link in df_links.values:
        df = pd.DataFrame(data=[[link, res_map[name]]], columns=DF_COLS)
        df_all = df_all.append(df)
    df_all.to_csv('ALL_RES.csv', index=False)

if __name__ == '__main__':
    main()