import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('D:/lab6/.xlsx')
dd = pd.read_excel('D:/lab6/valid.xlsx')
df_origin=df.copy()
row_start=len(df.index)
miss_counts = df.isnull().sum()
print("Колличественные характеристики набора данных")
print(df.describe())
print("Количество пустых значений")
print(miss_counts)
print("BASE DF:")
print(df)
print('Количество записей в наборе до начала обработки:',row_start)

# Преобразование всех текстовых значений к верхнему регистру
df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)

# Преобразование всех форматов даты к единому
df['STARTMARKETINGDATE'] = pd.to_datetime(df['STARTMARKETINGDATE'], format='%Y%m%d')
df['STARTMARKETINGDATE'] = df['STARTMARKETINGDATE'].dt.strftime('%d.%m.%Y')
df['LISTING_RECORD_CERTIFIED_THROUGH'] = pd.to_datetime(df['LISTING_RECORD_CERTIFIED_THROUGH'], format='%Y%m%d')
df['LISTING_RECORD_CERTIFIED_THROUGH'] = df['LISTING_RECORD_CERTIFIED_THROUGH'].dt.strftime('%d.%m.%Y')


# Построение диаграммы
plt.figure()
plt.title('Исходное количество записей')
df.count().plot(kind='bar', linestyle='solid', color='blue')
plt.xlabel('Столбцы')
plt.ylabel('Количество записей')
plt.show()

# Удаление дубликатов записей в столбцах ACTIVE_NUMERATOR_STRENGTH,ACTIVE_INGRED_UNIT,SUBSTANCENAME,PRODUCTTYPENAME,PRODUCTTYPENAME,NONPROPRIETARYNAME,DOSAGEFORMNAME
df.drop_duplicates(subset=['ACTIVE_NUMERATOR_STRENGTH','ACTIVE_INGRED_UNIT','SUBSTANCENAME','PRODUCTTYPENAME',
                              'PRODUCTTYPENAME','NONPROPRIETARYNAME','DOSAGEFORMNAME'],inplace=True)
print("DF AFTER DELETE DUPLICATES:")
print(df)

# Построение диаграммы
plt.figure()
plt.title('Количество записей после удаления дубликатов записей в столбцах')
df.count().plot(kind='bar', linestyle='solid', color='blue')
plt.xlabel('Столбцы')
plt.ylabel('Количество записей')
plt.show()

# Удаление аномальных значений
df['ACTIVE_NUMERATOR_STRENGTH'] = pd.to_numeric(df['ACTIVE_NUMERATOR_STRENGTH'], errors='coerce').fillna(0)

df['ACTIVE_NUMERATOR_STRENGTH'] = df['ACTIVE_NUMERATOR_STRENGTH'].astype(int)

df = df[df['ACTIVE_NUMERATOR_STRENGTH'] <= 1000]
print("DataFrame после удаления аномально больших значений:")
print(df)

# Построение диаграммы
plt.figure()
plt.title('Количество записей после удаления аномальных значений')
df.count().plot(kind='bar', linestyle='solid', color='blue')
plt.xlabel('Столбцы')
plt.ylabel('Количество записей')
plt.show()

# Удаление записей с дубликатами в столбце PRODUCTNDC
df.drop_duplicates(subset='PRODUCTNDC',inplace=True)
print("AFTER DELETE DUPLICATES PRODUCTNDC")
print(df)

# Построение диаграммы
plt.figure()
plt.title('Количество записей после удаления дубликатов в столбце PRODUCTNDC')
df.count().plot(kind='bar', linestyle='solid', color='blue')
plt.xlabel('Столбцы')
plt.ylabel('Количество записей')
plt.show()

# Подсчет количества использования каждой единицы измерения
unit_counts = df['ACTIVE_INGRED_UNIT'].value_counts()
print(unit_counts)
#
# Получение списка уникальных единиц измерения, которые используются меньше 5 раз
units_to_remove = unit_counts[unit_counts < 3].index.tolist()
#
# Удаление строк, содержащих указанные единицы измерения
df = df[~df['ACTIVE_INGRED_UNIT'].isin(units_to_remove)]

df_end=df.copy()
# Вывод обновленного DataFrame
print(df)
# Построение диаграммы
plt.figure()
plt.title('Сравнение исходных данных и данных после манипуляций')
labels = ['Исходные данные', 'Данные после манипуляций']
counts = [len(df_origin), len(df_end)]
plt.bar(labels, counts)
plt.xlabel('Тип данных')
plt.ylabel('Количество записей')
plt.show()

# Сравнение получившегося набора данных с набором и уже одобренных
zz=pd.merge(df, dd, on=['PRODUCTNDC'], how='inner')
print(zz)

# Построение сравнительной диаграммы
plt.figure()
plt.title('Сравнение исходных данных и данных пригодные к использованию')
labels = ['Исходные данные', 'Данные пригодные к использованию']
counts = [len(df_origin), len(zz)]
plt.bar(labels, counts)
plt.xlabel('Тип данных')
plt.ylabel('Количество записей')
plt.show()

# сохранение итогового файла
zz.to_excel('after_all_manipulation.xlsx')




