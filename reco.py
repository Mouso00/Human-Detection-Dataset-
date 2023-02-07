# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:04:00 2023

@author: Mous00
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df = pd.read_csv('CarPrices.csv')
# View the dataset
df.head()

# =============================================================================
# DATA CLEANING
# =============================================================================

# View data info and shape
df.info()
df.shape

#Remove duplicates

#check if there are any 'True' boolean statements in the duplicate function
df.duplicated().value_counts()


#Removing unnesecary attributes 

#Select required columns
output = df[['region', 'price', 'year', 'manufacturer', 'model', 
             'fuel', 'odometer', 'title_status', 'transmission', 
             'condition', 'cylinders', 'drive', 'size', 'type', 
             'paint_color', 'state','image_url', 'lat', 'long', 'posting_date']]
output.to_csv('vehicles2.csv',index=False)

df = pd.read_csv('vehicles2.csv')
df.head()
df.duplicated().value_counts()


#There seem to be 35 duplicate rows to be removed.

df.drop_duplicates(inplace=True)
df.shape

#Examine the posting dates against the geographical positions to filter hotspots.

#view posting date format
df.posting_date

# change posting date to datetime format and get the day and month
df=df.dropna(subset=['posting_date'],axis=0)
df['postmd'] = pd.to_datetime(df['posting_date'],utc=True).apply(lambda x: x.strftime('%B-%d')
                                                            if not pd.isnull(x) else '')
df['postmd']

df.sort_values(by='postmd')
df.shape


#Examine and remove missing data

#visualize the missing data
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,5))
#first plot, bar plot of missing values
ax1.bar(df.columns,df.isna().sum())
ax1.set_ylabel('Missing values')
ax1.set_ylim(0,df.shape[0])
ax1.tick_params('x',labelrotation=90)
#second plot, heatmap of missing values
sns.heatmap(df.isna(),yticklabels=False,cbar=False, cmap='Blues',ax=ax2)
plt.show()

#A few columns contain a lot of missing data, including size. 
#As the size column has more than 50% missing values it will be dropped as it is not informative.


#Remove size column and replot
df.drop(['size'],axis=1,inplace=True)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,5))
#first plot, bar plot of missing values
ax1.bar(df.columns,df.isna().sum())
ax1.set_ylabel('Missing values')
ax1.set_ylim(0,df.shape[0])
ax1.tick_params('x',labelrotation=90)
#second plot, heatmap of missing values
sns.heatmap(df.isna(),yticklabels=False,cbar=False, cmap='Blues',ax=ax2)
plt.show()


# =============================================================================
# Remove rows with missing values in key columns 
# (year, model, fuel, odometer, transmission). 
# Then, remove rows with missing values in remaining columns 
# (manufacturer, condition, cylinders, title_status, drive, type, paint_color, lat, long). 
# Retain rows with at least 7 non-NA values.
# 
# =============================================================================

#drop missing rows in year, model, fuel, odometer, transmission
df.dropna(subset=['year','model','fuel','odometer','transmission'],axis=0,inplace=True)
df.shape

#It seems there are no rows with concurrent missing values in those columns.

#keep rows with at least 7 non-NA values
df.dropna(subset=['manufacturer', 'condition', 'cylinders', 'title_status', 
                    'drive', 'type', 'paint_color', 'lat', 'long'], axis=0, thresh=7, inplace=True)
df.shape

# =============================================================================
# Remove outliers
# =============================================================================


#Search for outliers in the columns with numerical values, namely: price and odometer

#check the price column for outliers
fig,ax=plt.subplots(figsize=(8,4))
df.price.hist()
plt.title('Price distribution')
plt.grid(False)

#use boxplots to find outliers
fig,ax=plt.subplots(figsize=(8,4))
df.price.plot(kind='box')
plt.title('Price boxplot')
plt.grid(False)

fig,ax=plt.subplots(1,1,figsize=(16,5))
df.boxplot('price','manufacturer',ax=ax)
plt.grid(True)
plt.xticks(rotation=90)
plt.show()

#high price strangely related to ford and toyota. 
#Might be erorr in data enrty or scraping because these manufacturers are 
#not known for high end luxury or sporting cars

fig,ax=plt.subplots(1,1,figsize=(16,5))
df.boxplot('price','cylinders',ax=ax)
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
#high price related to 6 and 8 cylinder cars which indicate large pickups or sporting cars

fig,ax=plt.subplots(1,1,figsize=(16,5))
df.boxplot('price','type',ax=ax)
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
#high price strangely related to pickup trucks


fig,ax=plt.subplots(1,1,figsize=(16,5))
df.boxplot('price','condition',ax=ax)
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
# high price not related to new cars


#view the high priced car outliers
#for prices upwards of 100,000,000, ford, toyota, chevrolet, nissan and buick are present.
outlier=df.loc[((df.price > 500_000) & (df.price < 100_000_000))]
outlier



#Errors in price values, including random and extreme values. 
#Possible data entry or scraping errors. Unreasonable odometer readings.
#Recommend using a reasonable range for odometer analysis.


#more tests on the price column
test=df.loc[((df.manufacturer == "tesla"))]
test.head()


#No used Tesla cars under 30k in good condition.
#set a reasonable price range of 1,000-200,000 USD.

#filter data for price between 1k and 150k
df2 = df.loc[(df.price >= 1_000) & (df.price <= 200_000)]
sns.set()
fig,(ax1)=plt.subplots(figsize=(8,4))
df2.price.plot(kind='box')
ax2.set_title('1,000 - 200,000 price range')
plt.grid(False)


#check the odometer column for outliers
fig,ax=plt.subplots(figsize=(8,6))
df2.odometer.hist()
plt.title('Odometer distribution')
plt.grid(False)


#use boxplots to find outliers
fig,ax=plt.subplots(figsize=(8,6))
df2.odometer.plot(kind='box')
plt.title('Odometer boxplot')
plt.grid(False)

#Filter odometer readings using US average mileage per year (14,000 miles)
#and minimum car age (2 years) to keep range of 45,000 km to 500,000 km 
#(28,000 to 300,000 miles) driven for analysis.


#filter data for odometer between 0 miles and 500,000 km
df3 = df2.loc[(df2.odometer >= 45_000) & (df2.odometer <= 500_000)]
fig,(ax)=plt.subplots(figsize=(8,6))
df3.odometer.plot(kind='box')
ax.set_title('0 - 500,000 km odometer range')
plt.grid(None)

df3.shape

#save new dataset and load
df3.to_csv('vehicles2_cleaned.csv',index=False)
import pandas as pd
data = pd.read_csv('vehicles2_cleaned.csv')
data.head()


data.isna().sum()

# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

# Check number of listings for each day of data entry
fig,ax=plt.subplots(figsize=(12,6))
sns.set()
data.postmd.value_counts().sort_index().plot(kind='bar')
plt.ylabel('Number of postings')
plt.title('Number of postings by date')
plt.grid(None)

# Check number of postings per state
fig,ax=plt.subplots(figsize=(16,6))
data.state.str.upper().value_counts().sort_values(ascending=False).plot(kind='bar')
plt.ylabel('Number of postings')
plt.title('Number of postings by state')
y=[5000,5000]
x=[-1,100]
_=plt.plot(x,y,color='k',label='5,000 postings cut-off',linewidth=3)
_=plt.legend()
plt.grid(None)

#Focus on states with at least 5,000 used car postings in April-May to increase sales.
#Target high used car availability states.

# get the states with <5000 used car postings
data['state'].apply(lambda x: x.upper()).value_counts()>=5000
#data.state.value_counts().index

# Filter out the states with <5000 used car postings
datav2 = data.loc[~(data.state.isin(['tn', 'nj', 'ia', 'id', 'va', 'il', 'az', 
                                     'ma', 'mn', 'ks', 'mt', 'ga',
       'in', 'ct', 'ok', 'ky', 'al', 'sc', 'md', 'nm', 'ak', 'mo', 'nh', 'nv',
       'vt', 'me', 'dc', 'ri', 'ar', 'la', 'hi', 'sd', 'ut', 'ne', 'ms', 'de',
       'wv', 'nd', 'wy']))]
datav2.shape

#check the distribution of manufacturers 
fig,ax=plt.subplots(figsize=(16,6))
sns.set_context('paper',font_scale=2)
datav2.manufacturer.value_counts().sort_values(ascending=False).plot(kind='bar')
plt.ylabel('Count of manufacturers')
plt.title('Distribution of manufacturers')
plt.grid(False)
#y=[2500,2500]
#x=[-1,100]
#_=plt.plot(x,y,color='k',label='5,000 postings cut-off',linewidth=3)
#_=plt.legend()

#The top 3 most popular used car manufacturers are Ford, Chevrolet, and Toyota. 
#The least popular are Land Rover, Aston Martin, and Datsun.

#check the condition of the listed used cars 
fig,ax=plt.subplots(figsize=(10,6))
datav2.condition.value_counts().sort_values(ascending=False).plot(kind='bar')
plt.ylabel('Count')
plt.title('Distribution of condition of listed cars')
plt.grid(None)

#Most listed cars are in good or excellent condition, making them more marketable to buyers.

#check the number of cylinders of the listed used cars 
fig,ax=plt.subplots(figsize=(10,6))
datav2.cylinders.value_counts().sort_values(ascending=False).plot(kind='bar')
plt.ylabel('Count')
plt.title('Number of cylinders of cars')
plt.grid(None)

#Most popular listings on craigslist are 4, 6, and 8 cylinder cars, 
#as they are common in most cars.

#check the fuel type used for listed used cars 
fig,ax=plt.subplots(figsize=(10,6))
datav2.fuel.value_counts().sort_values(ascending=False).plot(kind='bar')
plt.ylabel('Count')
plt.title('Type of fuel')
plt.grid(None)

#check the odometer range for the listed cars 
fig,ax=plt.subplots(figsize=(10,6))
datav2.odometer.hist(bins=50)
plt.ylabel('Count')
plt.xticks(rotation=90)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50000))

plt.title('Odometer reading distribution')
plt.grid(None)

#Used cars mostly have odometer readings between 45,000 - 175,000 km.

#check the title stutus column
sns.countplot(x='title_status',data=datav2)
plt.title('Title status')
plt.xticks(rotation=90)
plt.grid([])


#check the transmission column
sns.countplot(x='transmission',data=datav2)
plt.title('Transmission')
plt.grid([])

#check the drive column
sns.countplot(x='drive',data=datav2)
plt.title('Type of drive mechanics')
plt.tick_params(labelsize=12)
plt.grid([])



#FWD and 4WD vehicles outnumber RWD, FWD are standard in many cars,
#crossovers and SUVs while 4WD are common in trucks and large SUVs, 
#RWD also found in trucks and SUVs.

#check the car type distribution
plt.figure(figsize=(10,6))
datav2.type.value_counts().sort_values(ascending=False).plot(kind='bar')
plt.title('Car type distribution')
plt.tick_params(labelsize=12)
plt.grid([])


#Sedans and SUVs are most popular, followed by pickups and trucks.

#check the car color distribution
plt.figure(figsize=(10,6))
datav2.paint_color.value_counts().sort_values(ascending=False).plot(kind='bar')
plt.title('Car colors distribution')
plt.tick_params(labelsize=12)
plt.grid([])

#White, black are the most popular car colors, followed by silver, blue, red and grey.


#save data to new file
datav2.to_csv('vehicles2_EDA.csv', index=False)
data = pd.read_csv('vehicles2_EDA.csv')

#next lets look at the distribution of the year of the listed used cars.

#convert the year column from 'float64' to year type as 'int64'
data['year']=pd.to_datetime(data['year'],errors='ignore',format='%y')


#check the car year distribution
plt.figure(figsize=(16,6))
sns.countplot(x='year',data=data,palette='Blues',dodge=False)
plt.title('Car year distribution')
plt.tick_params(labelsize=12)
plt.xticks(fontsize=9,rotation=90)
plt.tick_params(axis='y', labelleft=False, labelright=True)
plt.show()

#Most used cars model years are between 2011-2020, 
#cars tend to be sold within 10 years of ownership.
#Create two new columns: age of car by year, and average mileage per year
#to distinguish older from newer cars.

#create Age column
data['Age'] = 2021 - (data['year']-1)
#plot Age distribution
plt.figure(figsize=(16,6))
sns.set()
sns.countplot(x='Age',data=data,palette='Blues_r')
plt.xticks(fontsize=9,rotation=90)
plt.show()

#Sentiment on when a car is considered old varies, prime age 
#for used cars is 2-3 years old, for this analysis cars older 
#than 15 years will be considered old. Most listed cars are 4-5 years old.

#take out the 0-1 year old cars
data = data.loc[~(data.Age < 2)]

#create average mileage per year column
data['avg_mil'] = data['odometer'] / data['Age']
data['avg_mil']

#create a plot of the average mileages
plt.figure(figsize=(8,4))
data.avg_mil.hist(bins=25)
y=[0,40000]
x=[21500,21500]
_=plt.plot(x,y,color='k',label='Average mileage',linewidth=3)
_=plt.legend()
plt.grid(None)

#Create a column to classify cars based on average mileage per year,
#above or below 13,500 miles (21,500 km) in US.

#create milage rating column
data['mil_rating'] = "below average"
data.loc[data['avg_mil'] > 21500, 'mil_rating'] = "above average"

#plot the chart of mileage rating
ax = sns.countplot(x='mil_rating',data=data)
for p in ax.patches:
   ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+0.01))
plt.grid()
plt.show()


#Most cars have mileage below 21,500 km/year. 
#Many missing values in dataset, rename categorical variable 
#and check variables before removing missing rows for model creation.

#save data to new file
data.to_csv('vehicles2_cleaned_v2.csv',index=False)
data = pd.read_csv('vehicles2_cleaned_v2.csv')

#Review missing data to clean and use in price prediction and recommendation system models.

data.isna().sum()

#Remove rows with missing manufacturers to avoid cumbersome task of extracting it from model
#and random manufacturers not sufficing.

#drop missing values in manufacturer column
data.dropna(subset=['manufacturer'],axis=0,inplace=True)

#Classify manufacturers by their countries
#print unique manufacturer entries 
data.manufacturer.unique()

#further remove outliers in odometer for model purposes
def find_outliers_limit(df,col):
    print(col)
    print('-'*50)
    #removing outliers
    q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    print('Lower:',lower,' Upper:',upper)
    return lower,upper
def remove_outlier(df,col,upper,lower):
    # identify outliers
    outliers = [x for x in df[col] if x > upper or x < lower]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in df[col] if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    final= np.where(df[col]>upper,upper,np.where(df[col]<lower,lower,df[col]))
    return final
outlier_cols=['odometer']
for col in outlier_cols:
    lower,upper=find_outliers_limit(df,col)
    df[col]=remove_outlier(df,col,upper,lower)
    
plt.figure(figsize=(6,4))
df[outlier_cols].boxplot()

#Convert categorical columns to numerical using scikit learn's OrdinalEncoder.

# Loading Libraries
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from prettytable import PrettyTable
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
#Prepare data for model by handling categorical features. Prepare categorical features for correlation matrix.

#select numerical and categorical data
num_df=df.select_dtypes(include=np.number)
cat_df=df.select_dtypes(include=object)
#assign encoder
encoding=OrdinalEncoder()
#place categorical columns to list and encode
cat_cols=cat_df.columns.tolist()
encoding.fit(cat_df[cat_cols])
#transform categorical encoding and place in dataframe
cat_oe=encoding.transform(cat_df[cat_cols])
cat_oe=pd.DataFrame(cat_oe,columns=cat_cols)
cat_df.reset_index(inplace=True,drop=True)

cat_oe.head()

num_df.reset_index(inplace=True,drop=True)

cat_oe.reset_index(inplace=True,drop=True)

final_all_df=pd.concat([num_df,cat_oe],axis=1)

#plot the correlation matrix
plt.figure(figsize=(16,8))
sns.heatmap(data=final_all_df.corr(),annot=True)


#Price correlation with year, odometer, cylinders, and age (similar to year) is medium.
#Log transformation of price improves correlation and will be used to train the mode

df.columns


data_reg = df.loc[:, ['price', 'year', 'Age', 'odometer','cylinders','manufacturer', 'model', 'fuel', 
                      'title_status', 'transmission', 'condition',  'drive',
                      'type', 'paint_color', 'state']]
label_reg = data_reg
data_reg_original = data_reg
data_reg.head()


#Using LabelEncoder to convert categorical values for the model.
label_reg = data_reg

le_manufacturer = LabelEncoder()
le_model = LabelEncoder()
le_fuel = LabelEncoder()
le_title_status = LabelEncoder()
le_transmission = LabelEncoder()
le_condition = LabelEncoder()
le_drive = LabelEncoder()
le_type = LabelEncoder()
le_color = LabelEncoder()
le_state = LabelEncoder()

label_reg["type"] = le_type.fit_transform(label_reg['type'])
label_reg["manufacturer"] = le_manufacturer.fit_transform(label_reg['manufacturer'])
label_reg['paint_color'] = le_color.fit_transform(label_reg['paint_color'])
label_reg['drive'] = le_drive.fit_transform(label_reg['drive'])
label_reg["fuel"] = le_fuel.fit_transform(label_reg['fuel'])
label_reg["title_status"] = le_title_status.fit_transform(label_reg['title_status'])
label_reg['transmission'] = le_transmission.fit_transform(label_reg['transmission'])
label_reg['condition'] = le_condition.fit_transform(label_reg['condition'])
label_reg['state'] = le_state.fit_transform(label_reg['state'])
label_reg['model'] = le_model.fit_transform(label_reg['model'])

label_reg.head()

#Split data into 80-20 train and test to build prediction model.

X_regla = label_reg.drop('price', axis = 1)
y_regla = label_reg['price']

X_train, X_test, y_train, y_test = train_test_split(X_regla, y_regla, test_size = 0.2, random_state = 25)

#scaler=StandardScaler()
#X_train_scaled=scaler.fit_transform(X_train)
#X_test_scaled=scaler.transform(X_test)

#Three models built, best performer chosen for price prediction.
#Linear Regression model
#XGBoost Regressor model
#Random Forest Regressor model


# Model Building
def train_ml_model(x,y,model_type):
    if model_type=='lr':
        model=LinearRegression()
    elif model_type=='xgb':
        model=XGBRegressor()
    elif model_type=='rf':
        model=RandomForestRegressor()
    model.fit(x,y)
    
    return model

model_lr=train_ml_model(X_train,y_train,'lr')

model_xgb=train_ml_model(X_train,y_train,'xgb')

model_rf=train_ml_model(X_train,y_train,'rf')


#evaluate models
def model_evaluate(model,x,y):
    predictions=model.predict(x)
   # predictions=np.exp(predictions)
    mse=mean_squared_error(y,predictions)
    mae=mean_absolute_error(y,predictions)
    mape=mean_absolute_percentage_error(y,predictions)
    #msle=mean_squared_log_error(y,predictions)
    
    mse=round(mse,2)
    mae=round(mae,2)
    mape=round(mape,2)
    #msle=round(msle,2)
    
    return [mse,mae,mape]

summary=PrettyTable(['Model','MSE','MAE','MAPE'])
summary.add_row(['LR']+model_evaluate(model_lr,X_test,y_test))
summary.add_row(['XGB']+model_evaluate(model_xgb,X_test,y_test))
summary.add_row(['RF']+model_evaluate(model_rf,X_test,y_test))

print(summary)

#Above is the summary of evaluation metrics, MSE, MAE, and MAPE, for machine learning models.
#Random Forest Regressor model performed best with lowest evaluation parameters.

rf = X_train.join(y_train)
price_pred_df=rf.join(pd.Series(model_rf.predict(X_train), name='price_pred'))
price_pred_df.to_csv('vehicles_price_pred.csv')

#plot the actual vs the predicted price
y_pred=(model_rf.predict(X_test))

number_of_observations=50

x_ax = range(len(y_test[:number_of_observations]))

plt.figure(figsize=(16,8))
plt.plot(x_ax, y_test[:number_of_observations], label="Actual")
plt.plot(x_ax, y_pred[:number_of_observations], label="Predicted")
plt.title("Car Price - Actual vs Predicted data")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.xticks(np.arange(number_of_observations))
plt.legend()
plt.grid()
plt.show()


import pickle
rf_uc_model = 'rf_uc_model.sav'
pickle.dump(model_rf,open(rf_uc_model, 'wb'))

#Instructions on how users can use a specific function, "predict," to make predictions.

def predict(year,age,odom,cylin,manuf,model,fueltyp,title,transm,cond,drivetype,type,color,state):
    pred = (year,age,odom,cylin,manuf,model,fueltyp,title,transm,cond,drivetype,type,color,state)
    x = np.array([pred])
    #convert input using LabelEncoder
    le_reg = [np.nan,np.nan,np.nan,np.nan,le_manufacturer, le_model,le_fuel,le_title_status,le_transmission,
                  le_condition,le_drive,le_type,le_color,le_state]
    
    for i in range(4,14):
        x[:,i] = le_reg[i].transform(x[:,i])
        x
       
    loaded_model = pickle.load(open(rf_uc_model,'rb'))
    price_est = loaded_model.predict(x)
    return "Price Estimate =  $" + str(price_est)

#Use defined predict function to make price predictions 
#format - (year,age,odom,cylin,manuf,model,fueltyp,title,transm,cond,drivetype,type,color,state)
predict(2013,8,100000.0,4,'nissan','maxima','gas','clean','automatic'
                 ,'good','rwd','sedan','blue','ca')

#make prediction manual
#pred_format = (year,age,odom,cylin,manuf,model,fueltyp,title,transm,cond,drivetype,type,color,state)

pred = (2020,8,100000.0,4,'nissan','maxima','gas','clean','automatic'
                 ,'good','rwd','sedan','blue','ca') # user input
#system converts input to array
x=np.array([pred])
x

#input array converted using LabelEncoder
x[:, 4] = le_manufacturer.transform(x[:,4])
x[:, 5] = le_model.transform(x[:,5])
x[:, 6] = le_fuel.transform(x[:,6])
x[:, 7] = le_title_status.transform(x[:,7])
x[:, 8] = le_transmission.transform(x[:,8])
x[:, 9] = le_condition.transform(x[:,9])
x[:, 10] = le_drive.transform(x[:,10])
x[:, 11] = le_type.transform(x[:,11])
x[:, 12] = le_color.transform(x[:,12])
x[:, 13] = le_state.transform(x[:,13])

x = x.astype(float)
#x = pd.to_numeric(x)
x

#transformed input is fed into the model
price_pred = model_rf.predict(x)
print("Price Estimate = " + "$"+str(price_pred))

#load model for use using
loaded_model = pickle.load(open(rf_uc_model,'rb'))
result=loaded_model.score(X_test,y_test)
print(result)
            
