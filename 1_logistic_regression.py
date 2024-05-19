import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

df.to_csv('cukrzyca.csv', sep=';',index=False)

X = df.iloc[:,  :-1  ]   #wszystkie wiersze, kolumny bez ostatniej
y = df.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   # 20% to dane testowe

model = LogisticRegression()
model.fit(X_train, y_train)  # uczysz się na 80% danych
print(model.score(X_test, y_test))   #sprawdź dokładność
print(confusion_matrix(y_test, model.predict(X_test)))