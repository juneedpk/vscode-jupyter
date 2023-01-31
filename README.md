# Exploratory Data Analysis - EDA

1. EDA is applied to investigate the data and summarize the key insights.
2. It will give you the basic understanding of your data, it’s distribution, null values and much more.
3. You can either explore data using graphs or through some python functions.
4. There will be two type of analysis. Univariate and Bivariate. In the univariate, you will be analyzing a single attribute. But in the bivariate, you will be analyzing an attribute with the target attribute.
5. In the non-graphical approach, you will be using functions such as shape, summary, describe, isnull, info, datatypes and more.
   1. In the graphical approach, you will be using plots such as scatter, box, bar, density and correlation plots.

# Basic information about data - EDA

1. The `df.info()` & `df.describe()` function will give us the basic information about the dataset. For any data, it is good to start by knowing its information. 
   
   # Duplicate values
   1. You can use the `df.duplicate.sum()` function to the sum of duplicate value present if any. It will show the number of duplicate values if they are present in the data.
   
   # Unique values in the data
   1. You can find the number of unique values in the particular column using unique() function in python.The unique() function has returned the unique values which are present in the data and it is pretty much cool! e.g `df['Pclass'].unique()`

`df['Survived'].unique()` `df['Sex'].unique()` .

# Visualize the Unique counts

1. You can visualize the unique values present in the data. For this, we will be using the seaborn library. You have to call the sns.countlot() function and specify the variable to plot the count plot. e.g ``sns.countplot(df['Pclass']).unique()`

# Find the Null values

1. Finding the null values is the most important step in the EDA. As I told many a time, ensuring the quality of data is paramount. e.g. `df.isnull().sum()`
   
   ##  Replace the Null values

   1. we got a replace() function to replace all the null values with a specific data. It is too good!
     `df.replace(np.nan,'0',inplace = True)` , to check changes we use code : `df.isnull().sum()`

# Know the datatypes
1. Knowing the datatypes which you are exploring is very important and an easy process too by using command `df.dtypes` .
   

# Filter the Data

1. Yes, you can filter the data based on some logic, by using command :  e.g. `df[df['Pclass']==1].head()`

# A quick box plot

1. You can create a box plot for any numerical column using a single line of code command : `df[['Fare']].boxplot()`


# Correlation Plot - EDA

1. Finally, to find the correlation among the variables, we can make use of the correlation function. This will give you a fair idea of the correlation strength between different variables. `df.corr()`
This is the correlation matrix with the range from +1 to -1 where +1 is highly and positively correlated and -1 will be highly negatively correlated.

You can even visualize the correlation matrix using seaborn library command : `sns.heatmap(df.corr())`




