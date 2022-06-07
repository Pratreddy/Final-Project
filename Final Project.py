#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Display all the rows of the data.
pd.set_option("display.max_rows",None)


# In[3]:


#Display all the rows of the data.
pd.set_option("display.max_columns",None)


# # Train data Loading

# In[4]:


tr_d=pd.read_csv(r"C:\Users\HP'\Desktop\Data Sets\Property_Price_Train.csv")
#tr_d


# # Info

# In[5]:


#tr_d.info()


# In[6]:


tr_d.head()


# In[7]:


#tr_d.tail()


# In[8]:


#Printing shape of the train data with rows and columns
tr_d.shape


# In[9]:


#tr_d.skew()


# # EDA

# In[10]:


#Checking the missing values in train data
missing_train=tr_d.isnull().sum().sort_values(ascending=False)
missing_train


# In[11]:


#Plotting the top 20 missing values in the data
missing_x_axis=missing_train[:20]
missing_y_axis=missing_train[:20].index
width = 10
height = 8
plt.figure(figsize=(width,height))
sns.barplot(missing_x_axis,missing_y_axis)
plt.title('Missing values in train data')


# #Let us first check the how many numerical variables and categorical variables are present in the data.

# In[12]:


#List of Numerical variables:
numerical_features = [feature for feature in tr_d.columns if tr_d[feature].dtype != 'O']
print('Number of numerical features:',len(numerical_features))

#Visualising the numerical variables
tr_d[numerical_features].head()


# In[13]:


#List of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Year' in feature]
year_feature


# In[14]:


#Let's explore the content of these year variables
for feature in year_feature:
    print(feature, tr_d[feature].unique())


# #From the data we have 4 temporal variables(Datatime variables). To analyze these variables we need to check whether there is relation between these variables and Sales price

# In[15]:


tr_d.groupby('Year_Sold')['Sale_Price'].median().plot()
plt.xlabel('Year_Sold')
plt.ylabel('Sale_Price')
plt.title('Sale price vs Year sold')


# In[16]:


year_feature


# In[17]:


#Here we will compare the difference between all year features with Sale price
for feature in year_feature:
    if feature!='Year_Sold':
        data=tr_d.copy()
#Also we will find the difference between year variables and year of the house was sold for
        data[feature]=data['Year_Sold'] - data[feature]

        plt.scatter(data[feature],data['Sale_Price'])
        plt.xlabel(feature)
        plt.ylabel('Sale_Price')
        plt.show()


# In[18]:


##Numerical variables are usually of 2 types
#1. Continous variable and Discrete variable

discrete_feature=[feature for feature in numerical_features if len(tr_d[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete variables count: {}",format(len(discrete_feature)))


# In[19]:


discrete_feature


# In[20]:


# Finding the relationship between discrete features and sales price

for feature in discrete_feature:
    data=tr_d.copy()
    data.groupby(feature)['Sale_Price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale_Price')
    plt.title(feature)
    plt.show()


# In[21]:


#Continous Variable
continous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continous feature count: {}",format(len(continous_feature)))


# In[22]:


continous_feature


# In[23]:


# Finding the relationship between continous features and sales price

for feature in continous_feature:
    data=tr_d.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()


# In[24]:


# Logarithmic Transformation:

data=tr_d.copy()
for feature in continous_feature:
    data=tr_d.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['Sale_Price']=np.log(data['Sale_Price'])
        plt.scatter(data[feature],data['Sale_Price'])
        plt.xlabel(feature)
        plt.ylabel('Sale_Price')
        plt.title(feature)
        plt.show()

#Outliers

for feature in continous_feature:
    data=tr_d.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()    


# In[25]:


#List of Categorical variables:

categorical_features=[feature for feature in tr_d.columns if data[feature].dtypes=='O']
categorical_features


# In[26]:


tr_d[categorical_features].head()


# In[27]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(tr_d[feature].unique())))


# In[28]:


# Finding the relationship between categorical variables and Dependent variables(Sale price)

for feature in categorical_features:
    data=tr_d.copy()
    data.groupby(feature)['Sale_Price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale_Price')
    plt.title(feature)
    plt.show()


# # Describe

# In[29]:


#tr_d.describe()


# # Checking Null values

# In[30]:


#Checking the percentage of nan values present in each feature

#1.First step is to make the list of features which has null values
features_with_na=[features for features in tr_d.columns if tr_d[features].isnull().sum()>1]

#2.Second step is to print the feature name and the percentage of null values
for feature in features_with_na:
    print(feature,np.round(tr_d[feature].isnull().mean(),4),'% null values')


# #Since there are many null values,we need to find the relationship between null values and Sales price.

# In[31]:


for feature in features_with_na:
    data = tr_d.copy()
    
#Here let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)

# Let's calculate the mean Sales price where the information is missing or present
    data.groupby(feature)['Sale_Price'].median().plot.bar()
    plt.title(feature)
    plt.show()


# In[32]:


tr_d.isnull().sum()


# In[33]:


#Ploting the heatmap to the missing values
sns.heatmap(tr_d.isnull(),yticklabels=False,cbar=False)


# # Filling Null values

# In[53]:


tr_d.Lot_Extent.value_counts()


# In[54]:


tr_d['Lot_Extent']=tr_d['Lot_Extent'].fillna(tr_d['Lot_Extent'].mean())


# In[55]:


tr_d.Basement_Height.value_counts()


# In[56]:


tr_d.Basement_Height=tr_d.Basement_Height.fillna("TA")


# In[57]:


#The variable "Lane_Type" containes more than 50% null values so we need to drop that column
tr_d.drop(['Lane_Type'],axis=1,inplace=True)


# In[58]:


tr_d.Basement_Condition.value_counts()


# In[59]:


tr_d.Basement_Condition=tr_d.Basement_Height.fillna("TA")


# In[60]:


tr_d.Brick_Veneer_Type.value_counts()


# In[61]:


tr_d.Brick_Veneer_Type=tr_d.Brick_Veneer_Type.fillna("None")


# In[62]:


tr_d.Brick_Veneer_Area.value_counts()


# In[63]:


tr_d['Brick_Veneer_Area']=tr_d['Brick_Veneer_Area'].fillna(tr_d['Brick_Veneer_Area'].mean())


# In[64]:


tr_d.Exposure_Level.value_counts()


# In[65]:


tr_d.Exposure_Level=tr_d.Exposure_Level.fillna("No")


# In[66]:


tr_d['BsmtFinType1']=tr_d['BsmtFinType1'].fillna(tr_d['BsmtFinType1'].mode()[0])
tr_d['BsmtFinType2']=tr_d['BsmtFinType2'].fillna(tr_d['BsmtFinType2'].mode()[0])


# In[67]:


tr_d.Electrical_System.value_counts()


# In[68]:


tr_d.Electrical_System=tr_d.Electrical_System.fillna("SBrkr")


# In[69]:


#As these variables are categorical then we can replace the most frequent value by using mode
tr_d['Fireplace_Quality']=tr_d['Fireplace_Quality'].fillna(tr_d['Fireplace_Quality'].mode()[0])
tr_d['Garage']=tr_d['Garage'].fillna(tr_d['Garage'].mode()[0])
tr_d['Garage_Finish_Year']=tr_d['Garage_Finish_Year'].fillna(tr_d['Garage_Finish_Year'].mode()[0])
tr_d['Garage_Quality']=tr_d['Garage_Quality'].fillna(tr_d['Garage_Quality'].mode()[0])
tr_d['Garage_Condition']=tr_d['Garage_Condition'].fillna(tr_d['Garage_Condition'].mode()[0])


# In[70]:


tr_d.drop(['Garage_Built_Year'],axis=1,inplace=True)


# In[71]:


#The variables "Pool_Quality","Fence_Quality","Miscellaneous_Feature" containes more than 50% null values so we need to drop that columns
tr_d.drop(['Pool_Quality','Fence_Quality','Miscellaneous_Feature'],axis=1,inplace=True)


# In[72]:


sns.heatmap(tr_d.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[73]:


tr_d.shape


# # Checking Data type

# In[74]:


tr_d.dtypes.sort_values(ascending=False)


# # Data type Conversion

# In[75]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[76]:


tr_d.Heating_Type = le.fit_transform(tr_d.Heating_Type)
tr_d.Exterior_Condition  = le.fit_transform(tr_d.Exterior_Condition )
tr_d.Exposure_Level  = le.fit_transform(tr_d.Exposure_Level )
tr_d.Garage_Quality = le.fit_transform(tr_d.Garage_Quality)
tr_d.Brick_Veneer_Type = le.fit_transform(tr_d.Brick_Veneer_Type)
tr_d.Exterior_Material = le.fit_transform(tr_d.Exterior_Material)
tr_d.Functional_Rate = le.fit_transform(tr_d.Functional_Rate)
tr_d.Foundation_Type  = le.fit_transform(tr_d.Foundation_Type)
tr_d.Basement_Height = le.fit_transform(tr_d.Basement_Height)


# In[77]:


tr_d.Roof_Design = le.fit_transform(tr_d.Roof_Design)
tr_d.Kitchen_Quality = le.fit_transform(tr_d.Kitchen_Quality)
tr_d.Air_Conditioning = le.fit_transform(tr_d.Air_Conditioning )
tr_d.Garage_Condition = le.fit_transform(tr_d.Garage_Condition)
tr_d.BsmtFinType2 = le.fit_transform(tr_d.BsmtFinType2)
tr_d.Pavedd_Drive = le.fit_transform(tr_d.Pavedd_Drive)
tr_d.BsmtFinType1 = le.fit_transform(tr_d.BsmtFinType1)
tr_d.Basement_Condition = le.fit_transform(tr_d.Basement_Condition)
tr_d.Electrical_System = le.fit_transform(tr_d.Electrical_System)


# In[78]:


tr_d.Exterior2nd = le.fit_transform(tr_d.Exterior2nd)
tr_d.Condition2 = le.fit_transform(tr_d.Condition2)
tr_d.Garage = le.fit_transform(tr_d.Garage)
tr_d.Garage_Finish_Year = le.fit_transform(tr_d.Garage_Finish_Year)
tr_d.Zoning_Class = le.fit_transform(tr_d.Zoning_Class)
tr_d.Heating_Quality = le.fit_transform(tr_d.Heating_Quality)
tr_d.Exterior1st = le.fit_transform(tr_d.Exterior1st)
tr_d.Roof_Quality = le.fit_transform(tr_d.Roof_Quality)
tr_d.Sale_Type = le.fit_transform(tr_d.Sale_Type)


# In[79]:


tr_d.House_Design = le.fit_transform(tr_d.House_Design)
tr_d.Utility_Type = le.fit_transform(tr_d.Utility_Type)
tr_d.Condition1 = le.fit_transform(tr_d.Condition1)
tr_d.Fireplace_Quality = le.fit_transform(tr_d.Fireplace_Quality)
tr_d.House_Type = le.fit_transform(tr_d.House_Type)
tr_d.Neighborhood = le.fit_transform(tr_d.Neighborhood)
tr_d.Lot_Configuration = le.fit_transform(tr_d.Lot_Configuration)
tr_d.Property_Slope = le.fit_transform(tr_d.Property_Slope)
tr_d.Land_Outline = le.fit_transform(tr_d.Land_Outline)
tr_d.Sale_Condition = le.fit_transform(tr_d.Sale_Condition)
tr_d.Road_Type = le.fit_transform(tr_d.Road_Type)
tr_d.Property_Shape = le.fit_transform(tr_d.Property_Shape)


# In[80]:


tr_d.dtypes


# In[81]:


tr_d.head()


# In[82]:


tr_d.columns


# # Feature Engineering

# ##Always we need to remember that there is way always be a chance of data leakage so we need to split the data first and then feature engineering

# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(tr_d,tr_d['Sale_Price'],test_size=0.1,random_state=0)


# In[35]:


x_train.shape,x_test.shape


# In[36]:


#Let us capture all the nan values
#First lets handle the categorical features which are missing
features_nan=[feature for feature in tr_d.columns if tr_d[feature].isnull().sum()>1 and tr_d[feature].dtypes=='O']

for feature in features_nan:
     print("{}: {}% missing values".format(feature,np.round(tr_d[feature].isnull().mean(),4)))


# In[37]:


#Replace missing value with a new label:
def replace_cat_feature(dataset,features_nan):
    data=tr_d.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

tr_d=replace_cat_feature(tr_d,features_nan)

tr_d[features_nan].isnull().sum()


# In[38]:


tr_d.head()


# In[39]:


#Now let's check for numerical variables that contains missing values
numerical_with_nan=[feature for feature in tr_d.columns if tr_d[feature].isnull().sum()>1 and tr_d[feature].dtypes!='O']

#Printing the numerical nan variables and percentage of nmissing values

for feature in numerical_with_nan:
    print("{}: {}% missing values".format(feature,np.around(tr_d[feature].isnull().sum(),4)))


# In[40]:


#Replacing the numerical Missing values

for feature in numerical_with_nan:
    median_value=tr_d[feature].median() #Repalcing by using median since there are outliers
    tr_d[feature+'nan']=np.where(tr_d[feature].isnull(), 1,0) #Creating the new feature to find nan values
    tr_d[feature].fillna(median_value,inplace=True)
    
tr_d[numerical_with_nan].isnull().sum()


# In[41]:


tr_d.head(50)


# In[42]:


# Temporal variables (Date time variables)
for feature in ['Construction_Year','Remodel_Year','Garage_Built_Year']:
    tr_d[feature]=tr_d['Year_Sold']-tr_d[feature]


# In[43]:


tr_d.head()


# In[44]:


tr_d[['Construction_Year','Remodel_Year','Garage_Built_Year']].head()


# In[45]:


#Numerical variables
#since the numerical variables are skewed we need to perform log normal distribution
tr_d.head()


# In[46]:


num_features=['Lot_Extent','Lot_Size','First_Floor_Area','Grade_Living_Area','Sale_Price']

for feature in num_features:
    tr_d[feature]=np.log(tr_d[feature])


# In[47]:


tr_d.head()


# In[48]:


#Handeling rare categorical features:
#We will remove categorical features that are present less than 1% of the observations

categorical_features=[feature for feature in tr_d.columns if tr_d[feature].dtype=='O']
categorical_features


# In[49]:


for feature in categorical_features:
    temp=tr_d.groupby(feature)['Sale_Price'].count()/len(tr_d)
    temp_df=temp[temp>0.01].index
    tr_d[feature]=np.where(tr_d[feature].isin(temp_df),tr_d[feature],'RV')


# In[50]:


tr_d.head(100)


# In[51]:


tr_d.groupby(['Zoning_Class'])['Sale_Price'].mean()


# In[52]:


tr_d.head()


# # Feature scaling

# In[83]:


feature_scale=[feature for feature in tr_d.columns if feature not in ['Id','Sale_Price']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(tr_d[feature_scale])


# In[84]:


scaler.transform(tr_d[feature_scale])


# # Feature Selection

# In[85]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
pd.set_option("display.max_columns",None)


# In[86]:


tr_d.head()


# In[87]:


#Capturing the dependent feature
y_train=tr_d[['Sale_Price']]


# In[88]:


#Dropping the dependent feature
x_train=tr_d.drop(['Id','Sale_Price'],axis=1)


# In[89]:


#Applying feature selection
#First,specifing the Lasso Regression model, and 
#selecting a suitable alpha (equivalent to penalty).
#The bigger the alpha the less features that will be selected.

#Then by using SelectFromModel object from sklearn,which will select the features which coefficients are non-zero


feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(x_train,y_train)


# In[90]:


feature_sel_model.get_support()


# In[91]:


#Let's print the number of total and selected features

#This is how we can make a list of the selected features
selected_feat = x_train.columns[(feature_sel_model.get_support())]

#Let's print some stats
print('total features: {}'.format((x_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))


# In[92]:


selected_feat


# In[93]:


x_train=x_train[selected_feat]


# In[94]:


x_train.head()


# In[95]:


tr_d.drop(['Id','Lot_Extent','Land_Outline','Utility_Type','Lot_Configuration','Property_Slope','Condition1','Condition2','House_Type','House_Design','Roof_Design','Roof_Quality','Exterior2nd','Brick_Veneer_Type','Exterior_Material','Exterior_Condition','Foundation_Type','Basement_Condition','BsmtFinType2','BsmtUnfSF','Heating_Type','Air_Conditioning','Electrical_System','Grade_Living_Area','Underground_Half_Bathroom','Full_Bathroom_Above_Grade','Half_Bathroom_Above_Grade','Bedroom_Above_Grade','Kitchen_Above_Grade','Functional_Rate','Garage_Finish_Year','Garage_Quality','W_Deck_Area','Miscellaneous_Value','Year_Sold','Sale_Type'],axis=1,inplace=True)


# In[96]:


train_df=x_train.shape
train_df


# In[97]:


tr_d.shape


# # Splitting the data into training and test sets

# In[98]:


x=tr_d.iloc[:,1:]
y=tr_d.iloc[:,0]


# In[99]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=101)


# In[100]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # 1.Multiple Linear Regression

# In[101]:


from sklearn.metrics import explained_variance_score


# In[102]:


from sklearn.linear_model import LinearRegression


# In[103]:


mlr=LinearRegression()
mlr.fit(x_train,y_train)
mlr_score = mlr.score(x_test,y_test)
pred_mlr = mlr.predict(x_test)
expl_mlr = explained_variance_score(pred_mlr,y_test)


# # 2.Random Forest

# In[104]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()


# In[105]:


rf_regressor = RandomForestRegressor(n_estimators=28,random_state=0)
rf_regressor.fit(x_train,y_train)
rf_regressor.score(x_test,y_test)
rf_pred = rf_regressor.predict(x_test)
rf_score = rf_regressor.score(x_test,y_test)
expl_rf = explained_variance_score(rf_pred,y_test)


# # 3.Decision Tree 

# In[106]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()


# In[110]:


tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(x_train,y_train)
tr_regressor.score(x_test,y_test)
tr_pred = tr_regressor.predict(x_test)
decision_score = tr_regressor.score(x_test,y_test)
expl_tr = explained_variance_score(tr_pred,y_test)


# In[111]:


print("Multiple Linear Regressioin on Model Score is",round(mlr.score(x_test,y_test)*100))
print("Decision tree regression on Model Score is",round(tr_regressor.score(x_test,y_test)*100))
print("Random Forest Regressioin on Model Score is",round(rf_regressor.score(x_test,y_test)*100))


# In[112]:


models_score = pd.DataFrame({'Model':['Multiple Linear Regression','Decision Tree','Random Forest Regression'],'score':[mlr_score,decision_score,rf_score],'Explained Variance Score':[expl_mlr,expl_tr,expl_rf]})
models_score


# In[ ]:





# # Test data Loading

# In[77]:


te_d=pd.read_csv(r"C:\Users\HP'\Desktop\Data Sets\Property_Price_Test.csv")
#te_d


# # Info

# In[78]:


#te_d.info


# In[79]:


#te_d.head


# In[80]:


#te_d.tail


# In[81]:


#Printing the shape of the test data with columns and rows
te_d.shape


# In[82]:


te_d.columns


# In[83]:


#te_d.skew


# # Describe

# In[84]:


te_d.describe()


# # Checking Null Values

# In[85]:


te_d.isnull().sum()


# In[86]:


te_d.dtypes


# In[87]:


#Ploting the heatmap to see the missing values
sns.heatmap(te_d.isnull(),yticklabels=False,cbar=False)


# # Feature Engineering

# In[88]:


#Let us capture all the nan values
#First lets handle the categorical features which are missing
features_nan=[feature for feature in te_d.columns if te_d[feature].isnull().sum()>1 and te_d[feature].dtypes=='O']

for feature in features_nan:
     print("{}: {}% missing values".format(feature,np.round(te_d[feature].isnull().mean(),4)))


# In[89]:


#Replace missing value with a new label:
def replace_cat_feature(dataset,features_nan):
    data=te_d.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

te_d=replace_cat_feature(te_d,features_nan)

te_d[features_nan].isnull().sum()


# In[90]:


te_d.head()


# In[91]:


#Now let's check for numerical variables that contains missing values
numerical_with_nan=[feature for feature in te_d.columns if te_d[feature].isnull().sum()>1 and te_d[feature].dtypes!='O']

#Printing the numerical nan variables and percentage of nmissing values

for feature in numerical_with_nan:
    print("{}: {}% missing values".format(feature,np.around(te_d[feature].isnull().sum(),4)))


# In[92]:


#Replacing the numerical Missing values

for feature in numerical_with_nan:
    median_value=te_d[feature].median() #Repalcing by using median since there are outliers
    te_d[feature+'nan']=np.where(te_d[feature].isnull(), 1,0) #Creating the new feature to find nan values
    te_d[feature].fillna(median_value,inplace=True)
    
te_d[numerical_with_nan].isnull().sum()


# In[93]:


te_d.head(50)


# In[94]:


# Temporal variables (Date time variables)
for feature in ['Construction_Year','Remodel_Year','Garage_Built_Year']:
    te_d[feature]=te_d['Year_Sold']-te_d[feature]


# In[95]:


te_d.head()


# In[96]:


te_d[['Construction_Year','Remodel_Year','Garage_Built_Year']].head()


# In[97]:


#Numerical variables
#since the numerical variables are skewed we need to perform log normal distribution
te_d.head()


# In[98]:


num_features=['Lot_Extent','Lot_Size','First_Floor_Area','Grade_Living_Area']

for feature in num_features:
    te_d[feature]=np.log(te_d[feature])


# In[99]:


te_d.head()


# In[100]:


#Handeling rare categorical features:
#We will remove categorical features that are present less than 1% of the observations

categorical_features=[feature for feature in te_d.columns if te_d[feature].dtype=='O']
categorical_features


# In[101]:


te_d.head(100)


# In[102]:


te_d.groupby(['Zoning_Class'])['Sale_Price'].mean()


# In[ ]:


te_d.head()


# # Filling Null Values

# In[103]:


te_d['Zoning_Class']=te_d['Zoning_Class'].fillna(te_d['Zoning_Class'].mode()[0])
te_d['Lot_Extent']=te_d['Lot_Extent'].fillna(te_d['Lot_Extent'].mode()[0])


# In[104]:


#The variable "Lane_Type" containes more than 50% null values so we need to drop that column
te_d.drop(['Lane_Type'],axis=1,inplace=True)


# In[105]:


te_d.Utility_Type.value_counts()


# In[106]:


te_d['Utility_Type']=te_d['Utility_Type'].fillna(te_d['Utility_Type'].mode()[0])


# In[107]:


te_d['Exterior1st']=te_d['Exterior1st'].fillna(te_d['Exterior1st'].mode()[0])
te_d['Exterior2nd']=te_d['Exterior2nd'].fillna(te_d['Exterior2nd'].mode()[0])


# In[108]:


te_d['Brick_Veneer_Type']=te_d['Brick_Veneer_Type'].fillna(te_d['Brick_Veneer_Type'].mode()[0])


# In[109]:


te_d['Brick_Veneer_Area']=te_d['Brick_Veneer_Area'].fillna(te_d['Brick_Veneer_Area'].mean())
te_d['Lot_Size']=te_d['Lot_Size'].fillna(te_d['Lot_Size'].mean())


# In[110]:


te_d['Basement_Height']=te_d['Basement_Height'].fillna(te_d['Basement_Height'].mode()[0])
te_d['Basement_Condition']=te_d['Basement_Condition'].fillna(te_d['Basement_Condition'].mode()[0])
te_d['Exposure_Level']=te_d['Exposure_Level'].fillna(te_d['Exposure_Level'].mode()[0])
te_d['BsmtFinType1']=te_d['BsmtFinType1'].fillna(te_d['BsmtFinType1'].mode()[0])
te_d['BsmtFinType2']=te_d['BsmtFinType2'].fillna(te_d['BsmtFinType2'].mode()[0])


# In[111]:


te_d['BsmtFinSF1']=te_d['BsmtFinSF1'].fillna(te_d['BsmtFinSF1'].mean())
te_d['BsmtFinSF2']=te_d['BsmtFinSF2'].fillna(te_d['BsmtFinSF2'].mean())
te_d['BsmtUnfSF']=te_d['BsmtUnfSF'].fillna(te_d['BsmtUnfSF'].mean())
te_d['Total_Basement_Area']=te_d['Total_Basement_Area'].fillna(te_d['Total_Basement_Area'].mean())


# In[112]:


te_d['Underground_Full_Bathroom']=te_d['Underground_Full_Bathroom'].fillna(te_d['Underground_Full_Bathroom'].mean())
te_d['Underground_Half_Bathroom']=te_d['Underground_Half_Bathroom'].fillna(te_d['Underground_Half_Bathroom'].mean())


# In[113]:


te_d['Kitchen_Quality']=te_d['Kitchen_Quality'].fillna(te_d['Kitchen_Quality'].mode()[0])
te_d['Functional_Rate']=te_d['Functional_Rate'].fillna(te_d['Functional_Rate'].mode()[0])


# In[114]:


te_d['Fireplace_Quality']=te_d['Fireplace_Quality'].fillna(te_d['Fireplace_Quality'].mode()[0])
te_d['Garage']=te_d['Garage'].fillna(te_d['Garage'].mode()[0])
te_d['Garage_Finish_Year']=te_d['Garage_Finish_Year'].fillna(te_d['Garage_Finish_Year'].mode()[0])
te_d['Garage_Quality']=te_d['Garage_Quality'].fillna(te_d['Garage_Quality'].mode()[0])
te_d['Garage_Condition']=te_d['Garage_Condition'].fillna(te_d['Garage_Condition'].mode()[0])


# In[115]:


te_d['Garage_Size']=te_d['Garage_Size'].fillna(te_d['Garage_Size'].mean())
te_d['Garage_Area']=te_d['Garage_Area'].fillna(te_d['Garage_Area'].mean())


# In[116]:


te_d['Sale_Type']=te_d['Sale_Type'].fillna(te_d['Sale_Type'].mode()[0])


# In[117]:


#The variables "Pool_Quality","Fence_Quality","Miscellaneous_Feature" containes more than 50% null values so we need to drop that columns
te_d.drop(['Pool_Quality','Fence_Quality','Miscellaneous_Feature'],axis=1,inplace=True)


# In[118]:


te_d.isnull().sum()


# In[119]:


sns.heatmap(te_d.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[120]:


te_d.shape


# # Checking data type

# In[121]:


te_d.dtypes.sort_values(ascending=False)


# # Data type Conversion

# In[122]:


te_d.Heating_Type = le.fit_transform(te_d.Heating_Type)
te_d.Exterior_Condition = le.fit_transform(te_d.Exterior_Condition )
te_d.Air_Conditioning = le.fit_transform(te_d.Air_Conditioning)
te_d.Foundation_Type = le.fit_transform(te_d.Foundation_Type)
te_d.Basement_Height = le.fit_transform(te_d.Basement_Height)
te_d.Basement_Condition = le.fit_transform(te_d.Basement_Condition)
te_d.Exposure_Level = le.fit_transform(te_d.Exposure_Level)
te_d.BsmtFinType1 = le.fit_transform(te_d.BsmtFinType1)
te_d.BsmtFinType2 = le.fit_transform(te_d.BsmtFinType2)


# In[123]:


te_d.Sale_Condition = le.fit_transform(te_d.Sale_Condition)
te_d.Heating_Quality = le.fit_transform(te_d.Heating_Quality)
te_d.Exterior1st = le.fit_transform(te_d.Exterior1st)
te_d.Exterior2nd = le.fit_transform(te_d.Exterior2nd)
te_d.Brick_Veneer_Type = le.fit_transform(te_d.Brick_Veneer_Type)
te_d.Exterior_Material = le.fit_transform(te_d.Exterior_Material)
te_d.Pavedd_Drive = le.fit_transform(te_d.Pavedd_Drive)
te_d.Zoning_Class = le.fit_transform(te_d.Zoning_Class)
te_d.Garage_Condition = le.fit_transform(te_d.Garage_Condition)


# In[124]:


te_d.Garage_Finish_Year = le.fit_transform(te_d.Garage_Finish_Year)
te_d.Condition2 = le.fit_transform(te_d.Condition2)
te_d.Garage_Quality = le.fit_transform(te_d.Garage_Quality)
te_d.Kitchen_Quality = le.fit_transform(te_d.Kitchen_Quality)
te_d.Sale_Type = le.fit_transform(te_d.Sale_Type)
te_d.Roof_Design = le.fit_transform(te_d.Roof_Design)
te_d.Functional_Rate = le.fit_transform(te_d.Functional_Rate)
te_d.Fireplace_Quality = le.fit_transform(te_d.Fireplace_Quality)
te_d.Garage = le.fit_transform(te_d.Garage)


# In[125]:


te_d.Lot_Configuration = le.fit_transform(te_d.Lot_Configuration)
te_d.House_Type = le.fit_transform(te_d.House_Type)
te_d.Electrical_System = le.fit_transform(te_d.Electrical_System)
te_d.Road_Type = le.fit_transform(te_d.Road_Type)
te_d.Property_Shape = le.fit_transform(te_d.Property_Shape)
te_d.Land_Outline = le.fit_transform(te_d.Land_Outline)
te_d.Utility_Type = le.fit_transform(te_d.Utility_Type)
te_d.House_Design = le.fit_transform(te_d.House_Design)
te_d.Property_Slope = le.fit_transform(te_d.Property_Slope)
te_d.Neighborhood = le.fit_transform(te_d.Neighborhood)
te_d.Condition1 = le.fit_transform(te_d.Condition1)
te_d.Roof_Quality = le.fit_transform(te_d.Roof_Quality)


# In[128]:


te_d.columns


# In[126]:


te_d.dtypes


# In[127]:


te_d.drop(['Lot_Extent','Road_Type','Condition1','Condition2','House_Type','House_Design','Roof_Design','Roof_Quality','Exterior2nd','Brick_Veneer_Type'],axis=1,inplace=True)
te_d.drop(['Exterior_Material','Exterior_Condition','Foundation_Type','Basement_Condition','BsmtFinType2','Heating_Type','Air_Conditioning','Electrical_System','Grade_Living_Area','Underground_Half_Bathroom'],axis=1,inplace=True)
te_d.drop(['Full_Bathroom_Above_Grade','Half_Bathroom_Above_Grade','Bedroom_Above_Grade','Functional_Rate','Garage_Built_Year','Garage_Finish_Year','Garage_Quality','W_Deck_Area','Miscellaneous_Value','Year_Sold'],axis=1,inplace=True)
te_d.drop(['Sale_Type','Sale_Type','Id','Road_Type','Land_Outline',''],axis=1,inplace=True)


# In[ ]:




