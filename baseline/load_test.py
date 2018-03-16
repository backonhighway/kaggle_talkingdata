import pandas as pd

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}
train = pd.read_csv('../input/train_first_100k.csv', nrows=1000, dtype = dtypes)

print(train.head())