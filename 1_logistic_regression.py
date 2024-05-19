import pandas as pd
import numpy as np


df = pd.read_csv('diabetes.csv')
print(df.head(3).to_string())  # wszystkie kolumny, 3 pierwsze wiersze
print(df.shape)   # kształt danych
print('Describe')
print(df.describe().T.to_string())   # T-zamiana wierszy z kolumnami

# imie = 'Ola'
# print('rozny zpais printa')
# print('Imie to ',imie,', a drugie imie to też ',imie)
# print(f'Imie to {imie}, a drugie imie to też {imie}')

print(f'Klasy: {df["outcome"].value_counts()}')   #2 razy więcej zdrowych
print(f'braki: \n{df.isna().sum()}')   # wypisz braki

for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    df[col] = df[col].replace(0, np.NaN)       # zamień zera na NIC
    mean_ = df[col].mean()           # policz średnią i zapisz
    df[col] = df[col].replace(np.NaN, mean_)   # zamień NaN na średnią

print(f'Po obrobce:\n{df.isna().sum()}')
