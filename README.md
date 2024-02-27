## TASK NO : 2
# CREDIT CARD FRAUD DETECTION

![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/2e746365-c164-4d67-b411-3c67f308e1ef)

## CODE
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
```
```
train_data = pd.read_csv('fraudTrain.csv')
test_data = pd.read_csv('fraudTest.csv')

print("Training Dataset")
train_data.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/7859ffcd-b445-401e-9a4f-51645e661b2a)

```
print("Testing Dataset")
test_data
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/3a273692-a748-4792-ac84-490953c6d9f2)

```
train_data.info()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/a5659f01-ecea-4df0-af12-305af2868b45)

```
# Calculate the value counts of the target variable
tr_data = train_data['is_fraud'].value_counts()

# Create the pie chart
plt.figure(figsize=(7, 7))  # Adjust the figure size as needed
plt.pie(tr_data, labels=tr_data.index, autopct='%1.1f%%', startangle=180, explode=(0.1, 0))
plt.title("Distribution of the Target Variable (\"is_fraud\")")
plt.axis('equal')
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/1bc1d2e3-36a2-41f0-b140-527163cc714b)

```
train_data.describe()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/9ca4e031-d24c-4091-9c5b-e7b0c20ff025)

```
#Graph shows that the majority of non-fraudulent transactions are clustered around small amounts

fraud_data = train_data[train_data.is_fraud == 1]
data = fraud_data['amt']
plt.figure(figsize=(10, 6))
plt.hist(data, bins = 100)
plt.title('Frequency of Transaction Amounts Across Non-Fraudulent Transactions')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/d0f1e5f2-2f32-4d3f-9776-18c75968b255)

```
#Gender Distribution chart
data = train_data['gender'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(data, labels=data.index, autopct='%1.1f%%')
plt.title("Distribution Of The Gender")
plt.axis('equal')
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/85c482b0-90d2-45a4-9d9b-ac23bd5e6bb8)

```
fraud_datas = train_data[train_data['is_fraud'] == 1]

data = fraud_datas['category']
plt.figure(figsize=(10, 7))
plt.hist(data, bins = 100)
plt.title('Frequency of Fraudulent Transactions of categorical Types')
plt.xlabel('Category Types')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/6c6b3230-5067-4cce-8665-b931e706246c)

```
#Data preprocessing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_data["trans_date_trans_time"] = le.fit_transform(train_data["trans_date_trans_time"])
train_data["job"] = le.fit_transform(train_data["job"])
train_data["merchant"] = le.fit_transform(train_data["merchant"])
train_data["category"] = le.fit_transform(train_data["category"])
train_data["gender"] = le.fit_transform(train_data["gender"])

train_data.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/875c8fa3-750f-4d2a-8329-45b4fd6893b7)

```
import seaborn as sns

train_data_test = train_data.corr() 
plt.figure(figsize=(14, 8))
sns.heatmap(train_data_test, annot=True)
plt.title("Fraudulent Transactions' Heatmap")
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/4c154ffa-7621-453f-89ea-9b42c0e80546)

```
fraud_dataSet = train_data[train_data.is_fraud == 1]
data = fraud_dataSet['age']
plt.figure(figsize=(10, 6))
plt.hist(data, bins = 100)
plt.title('Frequency of Fraudulent Transactions Across Age Groups')
plt.xlabel('Age of credit card holder')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/083f490f-62be-4a34-a510-2184fec89a4d)

### After all the analysis of the dataset, the data is split into training and testing data for training the model in different algorithms. Thus the prediction and accuracy for each algorithm differs accordingly.

## Logistic Regression
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/0a429d8b-2201-41ea-895f-4964762b4bac)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/cc1582fc-e351-46ff-9a38-6bc5778a4688)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/1f112a1b-0a4c-446a-8b9e-5118f0572d91)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/5b65a059-e248-4a3b-9e3f-bff6eab44089)

## Decision tree
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/270caf3a-5166-427c-b17f-7e7d62bde049)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/d93ce675-d362-4398-aec5-bbfb1413bf63)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/6e79fca7-b1f9-4f1f-a140-fdaa4498bff6)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/c3da15bd-61e8-4c28-b43a-919a75d48fad)

## Random Forest
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/af39cd4d-e88c-48d3-8f16-ec896557f342)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/91f2b5c7-2aae-49b4-a71a-03e7605a6b9f)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/c94cc3c8-e34d-4d56-8527-07dd472e026f)
![image](https://github.com/AnnBlessy/codsoft_taskno.2/assets/119477835/1f16a5bc-1d48-4323-b920-913db079d34f)
