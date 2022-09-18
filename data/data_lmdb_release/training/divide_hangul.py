from jamo import h2j, j2hcj
import pandas as pd
import numpy as np
train_data = pd.read_csv('./train.csv', index_col=False)

for i, label in enumerate(train_data['text']):
    divided_label = j2hcj(h2j(label))
    train_data['text'][i] = divided_label
fractions = np.array([0.8,0.2])
train_data=train_data.sample(frac=1)
train, val = np.array_split(train_data, (fractions[:-1].cumsum()*len(train_data)).astype(int))
print(train_data)
train.to_csv("./train_divided.txt", index=False, sep="\t", header=False, encoding="utf-8-sig")
val.to_csv("./val_divided.txt", index=False, sep="\t", header=False, encoding="utf-8-sig")