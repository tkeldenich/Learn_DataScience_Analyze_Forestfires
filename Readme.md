# **Data Science And Environment – How to correctly use EDA for Forestfire**

- [English Article](https://inside-machinelearning.com/en/data-science-and-environment/)
- [French Article](https://inside-machinelearning.com/data-science-et-environnement/)

In this article I propose you to use Data Science to solve Environment issues: we’re going to use EDA to better prevent forest fires.

**Each year, more than 2.8 million hectares of forest are burned by fires in the U.S.**

These fires are caused in specific locations. One can imagine that certain environmental characteristics influence the propagation of fires: temperature, dryness, wind speed, etc.

**But is this really the case?**

In this tutorial we’ll use Exploratory Data Analysis (EDA) to understand what causes the spread of forest fires.

## **Le Dataset forestfires.csv**

### **Forestfires.csv dataset**

We’ll use the [forestfires.csv](https://github.com/tkeldenich/Learn_DataScience_Analyze_Forestfires) dataset that you can find [on this link.](https://github.com/tkeldenich/Learn_DataScience_Analyze_Forestfires)

This dataset gathers the meteorological characteristics of fire starts in Portugal.

Objective of this dataset is to predict the number of hectares burned per fire given the characteristics of the environment.

Once you have [downloaded the dataset](https://github.com/tkeldenich/Learn_DataScience_Analyze_Forestfires), put it in your working environment (Notebook or local).

You can now load the dataset into a Pandas Dataframe and display its dimensions:


```python
import pandas as pd

df = pd.read_csv("forestfires.csv")
df.shape
```




    (517, 13)



517 rows for 13 columns.

**We notice first of all that the dataset is small. Only 517 rows. This can impact our analysis and it is an important information to be aware of.**

Indeed, the larger a dataset is, the more we can generalize the conclusions of this dataset. Conversely for a small dataset.

Now, let’s try to understand the columns of the dataset by displaying their type (int, float, string, …):


```python
df.dtypes
```




    X          int64
    Y          int64
    month     object
    day       object
    FFMC     float64
    DMC      float64
    DC       float64
    ISI      float64
    temp     float64
    RH         int64
    wind     float64
    rain     float64
    area     float64
    dtype: object



Here, I display the type and description of each column:

- X - X coordinates
- Y – Y coordinates
- month – month of the year
- day – day of the week
- FFMC – Fine Fuel Moisture Code
- DMC – Duff Moisture Code – Humus Moisture Index (top soil layer)
- DC – Drought Code – Soil Drought Indicator
- ISI – Initial Spread Index – A numerical estimate of the rate of fire spread
- temp – Temperature in C
- RH – Relative Humidity
- wind – Wind speed in km/h
- rain – Rain in mm/m2
- area – Burned area of the forest (in ha)

Then we can display the first lines of this dataset:


```python
df.head()
```





  <div id="df-9b318e25-8c34-4084-bb0d-aec5445d69ba">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>month</th>
      <th>day</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>temp</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>5</td>
      <td>mar</td>
      <td>fri</td>
      <td>86.2</td>
      <td>26.2</td>
      <td>94.3</td>
      <td>5.1</td>
      <td>8.2</td>
      <td>51</td>
      <td>6.7</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>oct</td>
      <td>tue</td>
      <td>90.6</td>
      <td>35.4</td>
      <td>669.1</td>
      <td>6.7</td>
      <td>18.0</td>
      <td>33</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>4</td>
      <td>oct</td>
      <td>sat</td>
      <td>90.6</td>
      <td>43.7</td>
      <td>686.9</td>
      <td>6.7</td>
      <td>14.6</td>
      <td>33</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>6</td>
      <td>mar</td>
      <td>fri</td>
      <td>91.7</td>
      <td>33.3</td>
      <td>77.5</td>
      <td>9.0</td>
      <td>8.3</td>
      <td>97</td>
      <td>4.0</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>6</td>
      <td>mar</td>
      <td>sun</td>
      <td>89.3</td>
      <td>51.3</td>
      <td>102.2</td>
      <td>9.6</td>
      <td>11.4</td>
      <td>99</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9b318e25-8c34-4084-bb0d-aec5445d69ba')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9b318e25-8c34-4084-bb0d-aec5445d69ba button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9b318e25-8c34-4084-bb0d-aec5445d69ba');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Now that we have a good overview of our dataset, we can start the EDA with the univariate analysis.

*Note: some people will use `df.describe()` to analyze the whole dataset in one shot. I find this method too heavy for a first analysis. I personally prefer to analyze each feature one by one.*

## **Univariate Analysis**

Univariate analysis is the fact of examining each feature separately.

This will allow us to get a deeper understanding of the dataset.

> Here, we are in the comprehension phase.

**The question associated with the Univariate Analysis is: What is the characteristics of the data that compose our dataset?**

### **Target: Area impacted by fire**

First, let’s analyze the target of this dataset: area impacted by fires.

We use the Seaborn library to display the distribution of our target:


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,5))
ax = sns.kdeplot(df['area'],shade=True,color='g')
plt.show()
```


![png](Readme_files/Readme_16_0.png)


Here we see that most of the forest fires burned less than 100 hectares of forest and that the majority burned 0 hectares.

Let’s push the analysis a little further by using a Tukey box. This will help us to understand the distribution of the target by displaying the median, quartiles, deciles and extreme values:


```python
plt.figure(figsize=(16,5))
ax = sns.boxplot(df['area'])
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning



![png](Readme_files/Readme_18_1.png)


Here it is really difficult to analyze the graph.

Indeed the extreme values are so far from the median that it is impossible to see clearly the Tukey box

We would still like to analyze this box. So, in addition to displaying it, we will zoom in:


```python
plt.figure(figsize=(16,5))
ax = sns.boxplot(df['area'])
ax.set_xlim(-1, 40)
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning



![png](Readme_files/Readme_20_1.png)


This is much more readable!

We see here that the mean is between 0 and 1. The third quartile is at 6. And the ninth decile is at 16.

All values above 16 are extreme values. That is to say that very few values are on this scale.

The most extreme values are called “outliers”. We can display [Skewness and Kurtosis](https://inside-machinelearning.com/en/skewness-and-kurtosis/) to evaluate the disparity of these extreme values:

Don’t know what [Skewness and Kurtosis are?](https://inside-machinelearning.com/en/skewness-and-kurtosis/) Feel free to read [our short article on the topic](https://inside-machinelearning.com/en/skewness-and-kurtosis/) to learn more 😉


```python
print("Skew: {}".format(df['area'].skew()))
print("Kurtosis: {}".format(df['area'].kurtosis()))
```

    Skew: 12.846933533934868
    Kurtosis: 194.1407210942299


The values of these two metrics are huge! This explains why we had to zoom in our graph.

**Skewness tells us that the majority of the data are on the left and the outliers are on the right.**

**Kurtosis tells us that the data tend to move away from the average.**

This is what we saw on our graph.

> With these two metrics we can understand how much the outliers affect our target.

Let’s now move on to the analysis of these outliers.

#### **Z-Score & Outliers**

As you’ve seen in the Tukey box, there are many extreme values.

But be careful, all these extreme values are not necessarily outliers.

It is indeed normal in any distribution to have extreme values.

**On the other hand, it is uncommon, even anomalous, to have outliers. This may indicate an error in the dataset.**

Therefore, let’s display these outliers to determine if it is an error.

To determine outliers, we use the z-score.

> Z-score calculates the distance of a point from the mean.

**If the z-score is less than -3 or greater than 3, it is considered an outlier.**

Let’s see this by displaying all points below -3 and above 3:


```python
from scipy.stats import zscore

y_outliers = df[abs(zscore(df['area'])) >= 3 ]
y_outliers
```





  <div id="df-e968df53-fb3e-4156-bf12-446b1d2fec6a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>month</th>
      <th>day</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>temp</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>237</th>
      <td>1</td>
      <td>2</td>
      <td>sep</td>
      <td>tue</td>
      <td>91.0</td>
      <td>129.5</td>
      <td>692.6</td>
      <td>7.0</td>
      <td>18.8</td>
      <td>40</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>212.88</td>
    </tr>
    <tr>
      <th>238</th>
      <td>6</td>
      <td>5</td>
      <td>sep</td>
      <td>sat</td>
      <td>92.5</td>
      <td>121.1</td>
      <td>674.4</td>
      <td>8.6</td>
      <td>25.1</td>
      <td>27</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1090.84</td>
    </tr>
    <tr>
      <th>415</th>
      <td>8</td>
      <td>6</td>
      <td>aug</td>
      <td>thu</td>
      <td>94.8</td>
      <td>222.4</td>
      <td>698.6</td>
      <td>13.9</td>
      <td>27.5</td>
      <td>27</td>
      <td>4.9</td>
      <td>0.0</td>
      <td>746.28</td>
    </tr>
    <tr>
      <th>479</th>
      <td>7</td>
      <td>4</td>
      <td>jul</td>
      <td>mon</td>
      <td>89.2</td>
      <td>103.9</td>
      <td>431.6</td>
      <td>6.4</td>
      <td>22.6</td>
      <td>57</td>
      <td>4.9</td>
      <td>0.0</td>
      <td>278.53</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e968df53-fb3e-4156-bf12-446b1d2fec6a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e968df53-fb3e-4156-bf12-446b1d2fec6a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e968df53-fb3e-4156-bf12-446b1d2fec6a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We have 4 outliers and good news for us, it seems to be correct data.

**No error in the dataset!**

It means we have a rather peculiar distribution of our target, that must be kept in mind for the rest of the project.

Let’s now move on to the analysis of our features.

### **Categorical data**

In a dataset, several types of data must be differentiated:

- categorical data
- numerical data

The analysis of these two types of data will be different. Therefore, I propose to extract each of them in two sub-Dataframe :

- categorical columns
- numerical columns

With Pandas it’s easy ! You just have to use the select_dtypes function and indicate include='object' for categorical data and exclude='object' for numerical data:


```python
dfa = df.drop(columns='area')
cat_columns = dfa.select_dtypes(include='object').columns.tolist()
num_columns = dfa.select_dtypes(exclude='object').columns.tolist()

cat_columns,num_columns
```




    (['month', 'day'],
     ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'])



We have two categorical data: months and days.

The rest is numerical data.

We can analyze the categorical data directly!

Let’s display their distribution:


```python
plt.figure(figsize=(16,10))
for i,col in enumerate(cat_columns,1):
    plt.subplot(2,2,i)
    sns.countplot(data=dfa,y=col)
    plt.subplot(2,2,i+2)
    df[col].value_counts(normalize=True).plot.bar()
    plt.ylabel(col)
    plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show()  
```


![png](Readme_files/Readme_32_0.png)


*Reminder: each row of our dataset represents a forest fire.*

Here, we see that most forest fires occur in August and September. It seems consistent since these are the hottest months of the year.

The distribution according to the days of the week, on the contrary, is more balanced. But we see that Sunday, Saturday, Friday stand out more often than the other days.

The weekend days are therefore the days when there are most often forest fires. This would corroborate the information of the [U.S. Department of Agriculture](https://www.nps.gov/articles/wildfire-causes-and-evaluation.htm) which indicates that 85% of forest fires are caused by human beings. Do we have a similar statistic in Portugal?

### **Numerical data**

Let’s continue our analysis with the numerical data.

Here, the analysis will be the same as with the target :

- understand the distribution
- evaluate the outliers

When we do this kind of analysis, we ask ourselves the question: is there a recurrent pattern in our graph?

Let’s display rain data:


```python
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.kdeplot(df['rain'],color='g',shade=True)
plt.subplot(1,2,2)
df['rain'].plot.box()
plt.show()
```


![png](Readme_files/Readme_36_0.png)


We see that on fire days, most of the time, there is no rain. This makes sense, as the moisture caused by rain is not favorable to fire.

**In July 2021, the forest fires in Turkey took place when it was 40°C outside**.

Are forest fires related to temperature?

Let’s see now:


```python
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.kdeplot(df['temp'],color='g',shade=True)
plt.subplot(1,2,2)
df['temp'].plot.box()
plt.show()
```


![png](Readme_files/Readme_38_0.png)


Surprisingly, we see that forest fires in Portugal do not seem to be correlated with the outside temperature. This invalidates our hypothesis.

**I invite you now to try at home to display graphs of the other numerical data.**

You can do it in one shot with this piece of code :


```python
plt.figure(figsize=(18,40))
for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(df[col],color='g',shade=True)
    plt.subplot(8,4,i+10)
    df[col].plot.box()
plt.tight_layout() 
plt.show()
num_data = df[num_columns]
```


![png](Readme_files/Readme_40_0.png)


The result is not displayed here for lack of space. But the analysis on the FFMC, DMC and DC are particularly interesting!

> There also lies the job of Data Scientists. We are experts in data but we need to understand the indicators that concern our data.

**Go Google to understand what these indicators mean and if the graphical results seem consistent. Give your answer in comments** 😉

Finally we can use Skewness and Kurtosis again to detect our outliers:


```python
pd.DataFrame(data=[df[num_columns].skew(),df[num_columns].kurtosis()],index=['skewness','kurtosis'])
```





  <div id="df-1e2d7c69-195a-41f4-9c87-316db6b86eb0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>temp</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>skewness</th>
      <td>0.036246</td>
      <td>0.417296</td>
      <td>-6.575606</td>
      <td>0.547498</td>
      <td>-1.100445</td>
      <td>2.536325</td>
      <td>-0.331172</td>
      <td>0.862904</td>
      <td>0.571001</td>
      <td>19.816344</td>
    </tr>
    <tr>
      <th>kurtosis</th>
      <td>-1.172331</td>
      <td>1.420553</td>
      <td>67.066041</td>
      <td>0.204822</td>
      <td>-0.245244</td>
      <td>21.458037</td>
      <td>0.136166</td>
      <td>0.438183</td>
      <td>0.054324</td>
      <td>421.295964</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1e2d7c69-195a-41f4-9c87-316db6b86eb0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1e2d7c69-195a-41f4-9c87-316db6b86eb0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1e2d7c69-195a-41f4-9c87-316db6b86eb0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




FFMC, ISI and rain are the 3 columns with an extreme [Skewness and Kurtosis](https://inside-machinelearning.com/en/skewness-and-kurtosis/) value and therefore with outliers.

## **Bivariate Analysis**

Now that we have understood our data thanks to the Univariate Analysis we can continue the project by trying to find links between our features and the area impacted by the forest fires.

**Bivariate Analysis is the examination of each of the features in relation to our target.**

This will allow us to make hypotheses about the dataset.

> Here we are in the theorization phase.

**The question associated with Bivariate Analysis is: Is there a link between our features and the target?**

If there is a linear relationship, for example between FFMC and our target, then we can easily predict the impacted area.

Indeed, if the more the FFMC increases, the larger the impacted area is, it will be easy to draw a linear relationship between these two data. And therefore to predict the target!

But will it be that simple?

Let’s see that now!

### **Categorical features**

First of all I propose to modify our target. In addition to having numerical data specifying the impacted area, it would be nice to have categorical data to perform other analyses.

If the impacted area is 0 ha, we indicate “No damage”, if it is less than 25 “Moderate damage”, less than 100, “High”, more than 100 “Very high”:




```python
def area_cat(area):
    if area == 0.0:
        return "No damage"
    elif area <= 25:
        return "moderate"
    elif area <= 100:
        return "high"
    else:
        return "very high"

df['damage_category'] = df['area'].apply(area_cat)
df.sample(frac=1).head()
```





  <div id="df-5bfb3b6c-e523-4e73-837b-fb4eace9dfb9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>month</th>
      <th>day</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>temp</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
      <th>area</th>
      <th>damage_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>9</td>
      <td>9</td>
      <td>jul</td>
      <td>tue</td>
      <td>85.8</td>
      <td>48.3</td>
      <td>313.4</td>
      <td>3.9</td>
      <td>18.0</td>
      <td>42</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0.36</td>
      <td>moderate</td>
    </tr>
    <tr>
      <th>410</th>
      <td>6</td>
      <td>3</td>
      <td>feb</td>
      <td>fri</td>
      <td>84.1</td>
      <td>7.3</td>
      <td>52.8</td>
      <td>2.7</td>
      <td>14.7</td>
      <td>42</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>No damage</td>
    </tr>
    <tr>
      <th>514</th>
      <td>7</td>
      <td>4</td>
      <td>aug</td>
      <td>sun</td>
      <td>81.6</td>
      <td>56.7</td>
      <td>665.6</td>
      <td>1.9</td>
      <td>21.2</td>
      <td>70</td>
      <td>6.7</td>
      <td>0.0</td>
      <td>11.16</td>
      <td>moderate</td>
    </tr>
    <tr>
      <th>315</th>
      <td>3</td>
      <td>4</td>
      <td>sep</td>
      <td>wed</td>
      <td>91.2</td>
      <td>134.7</td>
      <td>817.5</td>
      <td>7.2</td>
      <td>18.5</td>
      <td>30</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>No damage</td>
    </tr>
    <tr>
      <th>312</th>
      <td>2</td>
      <td>4</td>
      <td>sep</td>
      <td>sun</td>
      <td>50.4</td>
      <td>46.2</td>
      <td>706.6</td>
      <td>0.4</td>
      <td>12.2</td>
      <td>78</td>
      <td>6.3</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>No damage</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5bfb3b6c-e523-4e73-837b-fb4eace9dfb9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5bfb3b6c-e523-4e73-837b-fb4eace9dfb9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5bfb3b6c-e523-4e73-837b-fb4eace9dfb9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




As you can see here, we have not removed the “area” column. We have simply added a column that will allow us to perform different types of analysis.

We can use this new column by comparing it to our “month” and “day” categorical data:


```python
import numpy as np

plt.figure(figsize=(15,30))
sns.set(rc={'axes.facecolor':'#DEDEDE', 'figure.facecolor':'white'})
for col in cat_columns:
    cross = pd.crosstab(index=df['damage_category'],columns=df[col],normalize='index')
    cross.plot.barh(stacked=True,rot=40,cmap='hot')
    plt.xlabel('% distribution per category')
    plt.xticks(np.arange(0,1.1,0.1))
    plt.title("Forestfire damage each {}".format(col))
plt.show()
```


    <Figure size 1080x2160 with 0 Axes>



![png](Readme_files/Readme_50_1.png)



![png](Readme_files/Readme_50_2.png)


On the first graph, we can see at what time of the year the most devastating fires occur. Most of them occur in September, August and a small part in July.

In the Univariate Analysis, we saw that **there is a correlation between month of the year and forest fires.** Here we deepen this analysis: **there seems to be a correlation between month of the year and surface impacted by forest fires.**

Concerning the days of the year, we can see that the area impacted does not seem to depend particularly on them.

In the Univariate Analysis, we saw that it is during the weekend days that forest fires occur most often. Nevertheless, this does not seem to impact the area burned by forest fires. The latter is surely more impacted by the weather and environmental conditions than by the days of the week.

### **Numerical features**

To display the previous graph, we have modified the basic configurations of Matplotlib with `sns.set(rc={'axes.facecolor':'#DEDEDE', 'figure.facecolor':'white'})`.

Let’s reset them :


```python
import matplotlib
matplotlib.rc_file_defaults()
```

We can continue our analysis.

Is there a link between the wind speed and the surface impacted by forest fires?

As wind can participate in fire propagation, this is a significant analysis to do:


```python
plt.figure(figsize=(8,4))
sns.scatterplot(data=df,x='area',y='wind')
plt.show()
```


![png](Readme_files/Readme_56_0.png)


Here we have taken the numerical target data. We put them in relation with the wind speed. Nevertheless, we do not see any relation between these two data.

**As for the Univariate Analysis, I invite you to repeat this analysis on all our numerical data.**

This piece of code will help you to do it in one shot:


```python
plt.figure(figsize=(10,30))
acc = 1
for i,col in enumerate(num_columns,1):
    if col not in ['X','Y']:
      plt.subplot(8,1,acc)
      sns.scatterplot(data=df,x='area',y=col)
      acc += 1
plt.show()
```


![png](Readme_files/Readme_58_0.png)


Display the result to try to find a correlation!

### **Pearson & Heatmap formula**

The result of the graphical analysis is inconclusive. We can’t see any correlation between our numerical features and the target.

**Is this really the case?**

> We can calculate mathematically a correlation between two data thanks to the [Pearson Formula.](https://inside-machinelearning.com/en/pearson-formula-in-python-linear-correlation-coefficient/)

If you are interested in the subject, we have explained in detail in [this short article](https://inside-machinelearning.com/en/pearson-formula-in-python-linear-correlation-coefficient/) what the [Pearson Formula is and how to interpret it.](https://inside-machinelearning.com/en/pearson-formula-in-python-linear-correlation-coefficient/)

With Pandas and Seaborn we can easily calculate this correlation with the function `sns.heatmap(df.corr())`:


```python
plt.figure(figsize=(15,2))
sns.heatmap(df.corr().iloc[[-1]],
            cmap='RdBu_r',
            annot=True,
            vmin=-1, vmax=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f95566bcb50>




![png](Readme_files/Readme_62_1.png)


We call this a heatmap and we could display it for the whole dataset. For example to calculate the correlation between FFMC and DMC. But here we are interested in knowing if there is a correlation between our features and our target.

**The graphical result seems to be verified by the [Pearson Formula](https://inside-machinelearning.com/en/pearson-formula-in-python-linear-correlation-coefficient/): there is no correlation between the numerical features and our target.**

Here is the piece of code that will display the heatmap for the whole dataset:


```python
sns.set(font_scale=1.15)
plt.figure(figsize=(15,15))

sns.heatmap(
    df.corr(),        
    cmap='RdBu_r', 
    annot=True, 
    vmin=-1, vmax=1);
```


![png](Readme_files/Readme_64_0.png)


## **Multivariate Analysis**

Bivariate analysis allowed us to understand our data slightly better.

In particular, we found a relationship between the months of the year, and the area impacted by fires.

Nevertheless, this did not allow us to make any conclusive hypothesis to predict the number of hectares burned by forest fires.

We then continue our project with the Multivariate Analysis.

**Multivariate analysis is the inspection of several features at the same time by relating them to our target.**

This will allow us to make hypotheses about the dataset.

> We are still in theorizing phase.

**The question associated with Multivariate Analysis is: Is there a link between several of our features and the target?**

If there is a link, for example between temperature, relative humidity (RH) and the target, we can predict the impacted surface.

Indeed, we can imagine that the more the temperature increases, the more the relative humidity (RH) decreases (relation determined thanks to the previous heatmap) and thus that the impacted surface also increases. Therefore we could use a Decision Tree to predict the target!



### **Numerical Features**

Let’s analyze these features now:


```python
sns.set_palette('ch:s=.25,rot=-.25')
sns.relplot(
    x='temp', 
    y='RH', 
    data=df, 
    #palette='bright',
    kind='scatter', 
    hue='damage_category');
```


![png](Readme_files/Readme_69_0.png)


First of all, we confirm the visual link between RH and temperature. Indeed, as the temperature increases, the humidity decreases. However, this does not influence the surface impacted by the fire.

**We see that the damage caused by fire is randomly distributed on the graph.**

Let’s display another relationship. For example between the FFMC and the wind :


```python
sns.relplot(
    x='FFMC', 
    y='wind', 
    data=df, 
    #palette='bright',
    kind='scatter', 
    hue='damage_category');
```


![png](Readme_files/Readme_71_0.png)


Here again no correlation. Neither between wind and FFMC. Nor between these two data and the surface impacted by the fire.

**Instead of displaying these graphs one by one, we can display them all at the same time.**

To do this, we select the numerical features:


```python
selected_features = df.drop(columns=['damage_category','day','month','area']).columns
selected_features
```




    Index(['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'], dtype='object')



Then we display the relationship between each of these features and our target:


```python
sns.pairplot(df,hue='damage_category',vars=selected_features)
plt.show()
```


![png](Readme_files/Readme_75_0.png)


**Do you find any correlations? What to do if so? What to do if not?**

## **Conclusion**

In this article, we’ve used Data Science to analyze the forestfires.csv dataset.

**To understand this dataset we used Exploratory Data Analysis (EDA), namely with the :**

- Univariate
- Bivariate
- Multivariate

You now know the usefulness of each of these analyses.

I invite you to perform this project on your own and share your findings.

**Are the data sufficient to predict the area impacted by fire?**

**Or is there a lack of relevance in our data?**

Both of these conclusions can occur in any data science project and a thorough analysis is needed to determine this.
