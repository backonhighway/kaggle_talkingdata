import pandas as pd

test = pd.read_csv('../input/test_first_1000k.csv')
test_old = pd.read_csv('../input/test_old_first_1000k.csv')

print(test.describe(include="all"))
print(test_old.describe(include="all"))

print("-"*40)

print(test.head())
print(test_old.head())
print("-"*40)

print(test.tail())
print(test_old.tail())
