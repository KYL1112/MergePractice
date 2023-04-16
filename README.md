# 期中報告
![](https://i.imgur.com/jKB2Ctt.jpg)
1.
* 簡介
因為智慧型手機的無處不在，使人們能夠隨時宣布他們正在觀察的緊急情況。正因如此，很多人想以編程方式監控 Twitter。
某位Twitter用戶以隱喻方式使用了“ABLAZE”這個詞。這對人類來說是顯而易見的，尤其是在視覺輔助下。但是機器就不太清楚了。
* 參賽原因
作為一位Twitter的資深用戶，在長時間的使用下，我發現Twitter是一個極具影響力和重要性的社交媒體平台，它在全球範圍內有著廣泛的用戶群體，並且在政治、娛樂等各個領域都扮演著重要的角色。Twitter讓我們可以在第一時間了解到最新的新聞、和話題，並隨時隨地連接世界各地的人，讓我們分享自己的想法。這也使得Twitter成為了能夠傳播新聞時事與增強社交影響力的重要平台及工具。因為我深刻地了解Twitter對我們有多重要，因此想選擇這個題目。
* 資料集與目標介紹
在這個題目中，參賽者需要建立一個機器學習模型來預測哪些推文是關於真實災難的，哪些不是。資料庫包含 10,000 條人工分類的推文的數據集，有使用者id、location、text（推文內容）、keyword、target(1代表是，0代表否）。

2.
* 載入套件與匯入資料
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

```
* 查看資料資訊
```
train_df.head()
train_df.info()
```
* 查看有無缺失值並進行資料清理
可以看到location有很多缺失值，因此選擇直接用drop將此欄刪除。
有嘗試刪除keyword，但沒有太大的效果。
```
train_df.isnull().sum()
test_df.isnull().sum()

train = train_df.drop(columns=['location'])
test = test_df.drop(columns=['location'])
train = train_df.drop(columns=['keyword'])
test = test_df.drop(columns=['keyword'])
```
![](https://i.imgur.com/Iq94Cm3.png)

* 使用CountVectorizer
利用CountVectorizer對text中的字轉為字頻矩陣並存到X，Y則是存train中target的值。
將現有資料切分57.15%作為訓練資料集42.85%為測試資料集
```
vectorizer = CountVectorizer(stop_words='english',ngram_range=(1, 3), max_df=1.0, min_df=1)
vectors = vectorizer.fit_transform(train['text'])
X = vectors.toarray()
Y = train['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4285, random_state=67)
```
* 使用Logistic Regression模型
得到精準率0.7857799570946982、召回率0.6283249460819554、準確率0.8276515151515151
```
clf = LogisticRegression()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
clf.score(X_test,y_test)

from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y_test,pred)
recall_score(y_test,pred)
precision_score(y_test,pred)
```
![](https://i.imgur.com/v5aNtN1.png)
* 把得到的結果輸出成.csv檔
```
forSubmissionDF=pd.DataFrame(columns=['id','target'])
forSubmissionDF
print(pred.shape)
forSubmissionDF['id'] = test.id
forSubmissionDF['target'] = pred
forSubmissionDF.to_csv('for_submission_20230411.csv', index=False)
```
* 與上課內容的關聯性
step1:取得資料
用上課教的pd.read_csv()取得資料集裡的資料
step2:資料清理
用上課教的drop()將不需要用到的欄位（keyword與location）刪除
step3:資料切割
用上課教的train_test_split()將資料切分57.15%作為訓練資料集、42.85%為測試資料集
step4:模型選擇與使用
使用sklearn中的Logistic Regression模型
step5:結果分析與驗證
用上課教的accuracy_score, recall_score, precision_score來看預測的精準率、召回率、準確率

* 延伸學習
學習如何使用CountVectorizer對text中的字轉為字頻矩陣，並透過這些字頻矩陣來達到預測哪些推文是關於真實災難的目標
參考資料：https://blog.csdn.net/weixin_38278334/article/details/82320307

3.
* 可能的改善方式
1. 利用train_df.duplicated(["text", "target"]).sum()檢查text與target是否有相同的資料，有的話可以將其刪除
2. 進一步觀察刪除掉的keyword以及location，看有沒有特定的keyword與location，其target較容易為1

* 不同的嘗試與結果
1. 改用Random Forest Classifier模型，得到精準率0.7551333129022372、召回率0.501078360891445、準確率0.8690773067331671
```
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=150,bootstrap=True)
model2=rf.fit(X_train,y_train)
model2.score(X_train,y_train)
model2.score(X_test,y_test)
y_pred=model2.predict(X_test)
accuracy_score(y_test,y_pred)
recall_score(y_test,y_pred)
precision_score(y_test,y_pred)
```
![](https://i.imgur.com/G0edsmt.png)

2. 改用Decision Tree Classifier模型，得到精準率0.8276515151515151、召回率0.7534148094895758、準確率0.9980952380952381，比上課用的Logistic Regression模型表現得更好
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=98)
model1=dt.fit(X,Y)
model1.score(X_train,y_train)
model1.score(X_test,y_test)
y_pred=model1.predict(X_test)
accuracy_score(y_test,y_pred)
recall_score(y_test,y_pred)
precision_score(y_test,y_pred)
```
![](https://i.imgur.com/HRrWQhA.png)

* 比賽結果說明
1. 使用Logistic Regression模型，結果為0.5311
![](https://i.imgur.com/wuvyLCs.png)
2. 使用Decision Tree Classifier模型，結果為0.52896
![](https://i.imgur.com/EeluvEC.png)
雖然Decision Tree Classifier模型精準率較高，但Logistic Regression模型的比賽分數較高
