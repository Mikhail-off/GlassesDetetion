#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

argparser = argparse.ArgumentParser()
argparser.add_argument("--true", type=str,
                       help="Путь до истинной разметки")
argparser.add_argument("--pred", type=str,
                       help="Путь до полученной разметки")

args = argparser.parse_args()

def intersect_datasets(names_true, names_pred, y_true, y_pred):
    ind_true = 0
    ind_pred = 0

    y_true_intersected = []
    y_pred_intersected = []

    extra_names_true = []
    extra_names_pred = []

    while ind_true < len(names_true) and ind_pred < len(names_pred):
        name_true = names_true[ind_true]
        name_pred = names_pred[ind_pred]
        if name_true == name_pred:
            y_true_intersected.append(y_true[ind_true])
            y_pred_intersected.append(y_pred[ind_pred])
            ind_pred += 1
            ind_true += 1
        elif name_true < name_pred:
            extra_names_true.append(name_true)
            ind_true += 1
        else:
            extra_names_pred.append(name_pred)
            ind_pred += 1
    return y_true_intersected, y_pred_intersected, extra_names_true, extra_names_pred


def main():
    df_true = pd.read_csv(args.true)
    df_pred = pd.read_csv(args.pred)

    names_true = df_true.values[:, 0]
    names_pred = df_pred.values[:, 0]
    true_order = np.argsort(names_true)
    pred_order = np.argsort(names_pred)

    names_pred = names_pred[pred_order]
    names_true = names_true[true_order]

    y_true = df_true.values[true_order, 1].astype(int)
    y_pred = df_pred.values[pred_order, 1].astype(int)

    y_true, y_pred, extra_names_true, extra_names_pred = intersect_datasets(names_true, names_pred, y_true, y_pred)
    print(classification_report(y_true, y_pred))

    if len(extra_names_pred) != 0 or len(extra_names_true) != 0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nРазметка не совпадает по именам')
    else:
        print('Успешно')
    if len(extra_names_true) != 0:
        print('\nСписок имен, которые есть в исходной, но нет в полученной:')
        print(*extra_names_true, sep='\n')
    if len(extra_names_pred) != 0:
        print('\nСписок имен, которые есть в полученной, но нет в исходной:')
        print(*extra_names_pred, sep='\n')

if __name__ == '__main__':
    main()
