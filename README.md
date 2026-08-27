## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
from google.colab import drive drive.mount('/content/drive')

ls drive/MyDrive/'Colab Notebooks'/
```
### ENDODING
```
import pandas as pd import numpy as np

df=pd.read_csv('drive/MyDrive/Data Science/Encoding Data.csv') df
```
<img width="425" height="503" alt="image" src="https://github.com/user-attachments/assets/6cd4a547-1618-4e05-a3ba-a123b33931d0" />

### ORDINAL ENCODER
```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pm= ['Hot','Warm','Cold']

en1 = OrdinalEncoder(categories = [pm])

en1.fit_transform(df[["ord_2"]])
```
<img width="282" height="270" alt="image" src="https://github.com/user-attachments/assets/1e45301e-7ca4-4784-b128-99a74519b7a8" />

```
df['bo2']=en1.fit_transform(df[["ord_2"]]) df
```
### LABLE ENCODER
```
le=LabelEncoder()

dfc=df.copy()

dfc['ord_2'] = dfc['ord_2'].astype(str)

dfc
```
<img width="567" height="515" alt="image" src="https://github.com/user-attachments/assets/6e96c299-bf31-4206-ae48-29082667bd5b" />

### ONE HOT ENCODER
```
from sklearn.preprocessing import OneHotEncoder

One=OneHotEncoder(sparse_output=False) df2=df.copy()

enc=pd.DataFrame(One.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1) df2
```
<img width="662" height="502" alt="image" src="https://github.com/user-attachments/assets/ac6263ab-2b7c-40cf-8cbc-33eda9e6ce4b" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="950" height="522" alt="image" src="https://github.com/user-attachments/assets/fe562936-33fa-4b8e-bff5-37dc02f5fc5e" />

### BINARY ENCODER
```
pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("drive/MyDrive/Data Science/data.csv") df
```
<img width="673" height="498" alt="image" src="https://github.com/user-attachments/assets/e75f91ed-0c00-44e4-b3fd-ccc17778533f" />
```
be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2']) dfb=pd.concat([df,nd],axis=1) dfb1=df.copy()

dfb1
```
<img width="662" height="507" alt="image" src="https://github.com/user-attachments/assets/e389186e-0dde-428c-b625-278f1c16f222" />

### TARGET ENCODER

```
from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"]) cc=pd.concat([cc,new],axis=1) cc
```
<img width="761" height="507" alt="image" src="https://github.com/user-attachments/assets/b5757ad2-5ae6-466a-bf86-6467f5bd526f" />

### TRANSFORMATION
```
import numpy as np import pandas as pd import matplotlib.pyplot as plt import statsmodels.api as sm import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv('drive/MyDrive/Data Science/Data_to_Transform.csv') df

```
<img width="1072" height="628" alt="image" src="https://github.com/user-attachments/assets/99a29ff6-b25f-42ff-b882-a96c02db88bd" />
```
df.skew()
```
<img width="470" height="312" alt="image" src="https://github.com/user-attachments/assets/2ce543dc-9184-4182-bf2b-7ef9604d188b" />
```
np.log(df["Highly Positive Skew"])
```
<img width="432" height="602" alt="image" src="https://github.com/user-attachments/assets/59933ceb-4760-4ac3-98c4-d1360856d1c6" />
```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="481" height="608" alt="image" src="https://github.com/user-attachments/assets/19185423-c202-42a8-bed4-9299485378be" />
```
np.sqrt(df["Highly Positive Skew"])

```
<img width="431" height="602" alt="image" src="https://github.com/user-attachments/assets/935e3474-2043-428e-a967-1aa30ece7136" />
```
np.square(df["Highly Positive Skew"])

```
<img width="408" height="607" alt="image" src="https://github.com/user-attachments/assets/9e40cc54-ef73-4c03-aadb-ebcb878e23e5" />
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"]) df

```
<img width="1078" height="628" alt="image" src="https://github.com/user-attachments/assets/e76cb9bc-2eb1-48b3-8ec8-254e54b7ab3f" />
```
df["Moderate Negative Skew_yeojohnson"], lmbda = stats.yeojohnson(df["Moderate Negative Skew"])

df.skew()

```
<img width="570" height="401" alt="image" src="https://github.com/user-attachments/assets/9645bc60-9941-4138-9945-980128189f93" />
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()
```
<img width="590" height="446" alt="image" src="https://github.com/user-attachments/assets/ebf4d810-6be5-45bd-aab5-79154efc25fe" />
```
from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) df
```
<img width="1560" height="428" alt="image" src="https://github.com/user-attachments/assets/70aef453-6cf1-4b17-b56e-74bfa734bcb3" />
```
import seaborn as sns import statsmodels.api as sm import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45') plt.show()

```
<img width="883" height="637" alt="image" src="https://github.com/user-attachments/assets/d57a3893-cb3b-4e10-aa18-a6d4f83e1a14" />
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') plt.show()


```
<img width="895" height="640" alt="image" src="https://github.com/user-attachments/assets/98df9192-23ed-4fa0-ae3a-d87a0253bf33" />
```
from sklearn.preprocessing import QuantileTransformer qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45') plt.show()

```
<img width="826" height="627" alt="image" src="https://github.com/user-attachments/assets/89c5558c-e5b8-42cb-bb2d-9258c54447da" />
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]]) sm.qqplot(df['Highly Negative Skew'],line='45') plt.show()

```
<img width="846" height="622" alt="image" src="https://github.com/user-attachments/assets/2a7ddf34-36fb-406a-b35f-014a24bb1671" />
```
sm.qqplot(df['Highly Negative Skew_1'],line='45') plt.show()

```
<img width="812" height="626" alt="image" src="https://github.com/user-attachments/assets/5a215511-0c6f-4bcb-a661-522b17449350" />
```
dt=pd.read_csv("drive/MyDrive/Data Science/titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]]) sm.qqplot(dt['Age'],line='45') plt.show()
```
<img width="886" height="632" alt="image" src="https://github.com/user-attachments/assets/83d481d6-140d-4db3-b155-c8c9d4f6b781" />
```
sm.qqplot(dt['Age_1'],line='45') plt.show()

```
<img width="872" height="623" alt="image" src="https://github.com/user-attachments/assets/a8fee1ca-2c30-49ca-8e58-57d35d310b71" />


# RESULT:
   successfully performed Feature Encoding and Transformation process
       
