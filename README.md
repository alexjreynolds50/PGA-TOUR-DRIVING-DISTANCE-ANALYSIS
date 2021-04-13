# 2017 PGA Tour Data Analysis on Importance of Driving Distance

## Import Libraries and Dataset into Notebook


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
```

Import dataset into notebook


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive

```python
df = pd.read_csv('/content/gdrive/My Drive/Colab Datasets/PGATOUR_meta2.csv')
```

## Introduction into Dataset and Analysis

**Introduction to dataset:**
*   Dataset found on Kaggle.com
*   Data collected from the 2017 Professional Golfers Association (PGA) Tour season
*   Includes 195 players and 70 variables


---

**Purpose of analysis:**
*   Analyze importance of driving distance off the tee to success in golf.
*   Working to increase driving distance has been linked to shorter golfing careers on PGA tour due to injuries from increased torque forces on joints but especially on players backs.
*   Analysis will help make better decisions on whether chasing increased driving distance off the tee is worth the risk of increased injury and/or shorter career.
*   Analysis would be benefical to any golfers but especially the competitive golfer and any fitness professional who works with golfers on increasing their strength, flexibility, and power to increase their driving distance off the tee.


---


**Key infomation with Analysis**
*   Golf is one of the few sports where you are trying to shoot a low score. So the lower the score, the better.
*   Strokes gained is a statistic the measures how many strokes are gained on the field from various shots on each holes.  The more strokes gained the better.

## Hypotheses

**Hypothesis 1:**
*   HO = Players with higher average driving distance will not have more success on PGA Tour.
*   HA = Players with higher average driving distance will have more success on PGA Tour.

---

**Hypothesis 2:**
*   HO = Players with higher average driving distance will not have better strokes gained per round
*   HA = Players with higher average driving distance will have better strokes gained per round

---

**Hypothesis 3:**
*   HO = Players with higher average driving distance will not have lower scoring averages
*   HA = Players with higher average driving distance will have lower scoring averages


## Exploratory Dataset Analysis and Wrangle


Includes 195 players and 71 columns


```python
df.shape
```


    (195, 71)



Columns in dataset


```python
df.columns
```


    Index(['Player', 'EVENTS_PLAYED', 'POINTS', 'NUMBER_OF_WINS',
           'NUMBER_OF_TOP_Tens', 'POINTS_BEHIND_LEAD', 'ROUNDS_PLAYED',
           'SG_PUTTING_PER_ROUND', 'TOTAL_SG:PUTTING', 'MEASURED_ROUNDS',
           'AVG_Driving_DISTANCE', 'UP_AND_DOWN_%', 'PAR_OR_BETTER', 'MISSED_GIR',
           'FAIRWAY_HIT_%', 'FAIRWAYS_HIT', 'POSSIBLE_FAIRWAYS', 'GIR_RANK',
           'GOING_FOR_GREEN_IN_2%', 'ATTEMPTS_GFG', 'NON-ATTEMPTS_GFG',
           'RTP-GOING_FOR_THE_GREEN', 'RTP-NOT_GOING_FOR_THE_GRN', 'HOLE_OUTS',
           'SAND_SAVE%', 'NUMBER_OF_SAVES', 'NUMBER_OF_BUNKERS', 'TOTAL_O/U_PAR',
           'Three_PUTT%', 'TOTAL_3_PUTTS', 'SG_PER_ROUND', 'SG:OTT', 'SG:APR',
           'SG:ARG', 'DRIVES_320+%', 'TOTAL_DRIVES_FOR_320+', 'TOTAL_DRIVES',
           'ROUGH_TENDNECY%', 'TOTAL_ROUGH', 'FAIRWAY_BUNKER%',
           'TOTAL_FAIRWAY_BUNKERS', 'AVG_CLUB_HEAD_SPEED', 'FASTEST_CH_SPEED',
           'SLOWEST_CH_SPEED', 'AVG_BALL_SPEED', 'FASTEST_BALL_SPEED',
           'SLOWEST_BALL_SPEED', 'AVG_SMASH_FACTOR', 'HIGHEST_SF', 'LOWEST_SF',
           'AVG_LAUNCH_ANGLE', 'LOWEST_LAUNCH_ANGLE', 'STEEPEST_LAUNCH_ANGLE',
           'AVG_SPIN_RATE', 'HIGHEST_SPIN_RATE', 'LOWEST_SPIN_RATE',
           'AVG_HANG_TIME', 'LONGEST_ACT.HANG_TIME', 'SHORTEST_ACT.HANG_TIME',
           'AVG_CARRY_DISTANCE', 'LONGEST_CARRY_DISTANCE',
           'SHORTEST_CARRY_DISTANCE', 'AVG_SCORE', 'TOTAL_STROKES', 'TOTAL_ROUNDS',
           'MAKES_BOGEY%', 'BOGEYS_MADE', 'HOLES_PLAYED', 'AGE', 'MONEY',
           'COUNTRY'],
          dtype='object')

Look at dtypes of columns and quick look at null values


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 195 entries, 0 to 194
    Data columns (total 71 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   Player                     195 non-null    object 
     1   EVENTS_PLAYED              195 non-null    int64  
     2   POINTS                     195 non-null    int64  
     3   NUMBER_OF_WINS             195 non-null    int64  
     4   NUMBER_OF_TOP_Tens         195 non-null    int64  
     5   POINTS_BEHIND_LEAD         194 non-null    float64
     6   ROUNDS_PLAYED              195 non-null    int64  
     7   SG_PUTTING_PER_ROUND       195 non-null    float64
     8   TOTAL_SG:PUTTING           195 non-null    float64
     9   MEASURED_ROUNDS            195 non-null    int64  
     10  AVG_Driving_DISTANCE       195 non-null    float64
     11  UP_AND_DOWN_%              195 non-null    float64
     12  PAR_OR_BETTER              195 non-null    int64  
     13  MISSED_GIR                 195 non-null    int64  
     14  FAIRWAY_HIT_%              195 non-null    float64
     15  FAIRWAYS_HIT               195 non-null    object 
     16  POSSIBLE_FAIRWAYS          195 non-null    int64  
     17  GIR_RANK                   195 non-null    int64  
     18  GOING_FOR_GREEN_IN_2%      195 non-null    float64
     19  ATTEMPTS_GFG               195 non-null    int64  
     20  NON-ATTEMPTS_GFG           195 non-null    int64  
     21  RTP-GOING_FOR_THE_GREEN    195 non-null    int64  
     22  RTP-NOT_GOING_FOR_THE_GRN  195 non-null    int64  
     23  HOLE_OUTS                  195 non-null    int64  
     24  SAND_SAVE%                 195 non-null    float64
     25  NUMBER_OF_SAVES            195 non-null    int64  
     26  NUMBER_OF_BUNKERS          195 non-null    int64  
     27  TOTAL_O/U_PAR              195 non-null    int64  
     28  Three_PUTT%                195 non-null    float64
     29  TOTAL_3_PUTTS              195 non-null    int64  
     30  SG_PER_ROUND               195 non-null    float64
     31  SG:OTT                     195 non-null    float64
     32  SG:APR                     195 non-null    float64
     33  SG:ARG                     195 non-null    float64
     34  DRIVES_320+%               195 non-null    float64
     35  TOTAL_DRIVES_FOR_320+      195 non-null    int64  
     36  TOTAL_DRIVES               195 non-null    object 
     37  ROUGH_TENDNECY%            195 non-null    float64
     38  TOTAL_ROUGH                195 non-null    int64  
     39  FAIRWAY_BUNKER%            195 non-null    float64
     40  TOTAL_FAIRWAY_BUNKERS      195 non-null    int64  
     41  AVG_CLUB_HEAD_SPEED        195 non-null    float64
     42  FASTEST_CH_SPEED           195 non-null    float64
     43  SLOWEST_CH_SPEED           195 non-null    float64
     44  AVG_BALL_SPEED             195 non-null    float64
     45  FASTEST_BALL_SPEED         195 non-null    float64
     46  SLOWEST_BALL_SPEED         195 non-null    float64
     47  AVG_SMASH_FACTOR           195 non-null    float64
     48  HIGHEST_SF                 195 non-null    float64
     49  LOWEST_SF                  195 non-null    float64
     50  AVG_LAUNCH_ANGLE           195 non-null    float64
     51  LOWEST_LAUNCH_ANGLE        195 non-null    float64
     52  STEEPEST_LAUNCH_ANGLE      195 non-null    float64
     53  AVG_SPIN_RATE              195 non-null    float64
     54  HIGHEST_SPIN_RATE          195 non-null    int64  
     55  LOWEST_SPIN_RATE           195 non-null    int64  
     56  AVG_HANG_TIME              195 non-null    float64
     57  LONGEST_ACT.HANG_TIME      195 non-null    float64
     58  SHORTEST_ACT.HANG_TIME     195 non-null    float64
     59  AVG_CARRY_DISTANCE         195 non-null    float64
     60  LONGEST_CARRY_DISTANCE     195 non-null    float64
     61  SHORTEST_CARRY_DISTANCE    195 non-null    float64
     62  AVG_SCORE                  195 non-null    float64
     63  TOTAL_STROKES              195 non-null    int64  
     64  TOTAL_ROUNDS               195 non-null    int64  
     65  MAKES_BOGEY%               195 non-null    float64
     66  BOGEYS_MADE                195 non-null    int64  
     67  HOLES_PLAYED               195 non-null    int64  
     68  AGE                        195 non-null    int64  
     69  MONEY                      195 non-null    int64  
     70  COUNTRY                    195 non-null    object 
    dtypes: float64(37), int64(30), object(4)
    memory usage: 108.3+ KB


All data is int64 or float64 except PLAYER, COUNTRY, FAIRWAYS_HIT, and TOTAL_DRIVES

Will need to convert FAIRWAYS_HIT and TOTAL_DRIVES to int64 for analysis


```python
df.FAIRWAYS_HIT.max()
```


    '1060'




```python
df.TOTAL_DRIVES.max()
```


    '1344'

Both FAIRWAY_HIT and TOTAL_DRIVES have values greater than 1000 and contain commas leading to object dtypes. 

Lets remove commas and change to int64 dtypes


```python
df.FAIRWAYS_HIT = df.FAIRWAYS_HIT.str.replace(',','')
df.FAIRWAYS_HIT = df.FAIRWAYS_HIT.astype('int64')
df.TOTAL_DRIVES = df.TOTAL_DRIVES.str.replace(',','')
df.TOTAL_DRIVES = df.TOTAL_DRIVES.astype('int64')
```


```python
df.FAIRWAYS_HIT.dtypes
df.TOTAL_DRIVES.dtypes
```


    dtype('int64')

All appropriate columns converted to int64 dtypes for analysis

5 row sample of dataset


```python
df.sample(5)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>EVENTS_PLAYED</th>
      <th>POINTS</th>
      <th>NUMBER_OF_WINS</th>
      <th>NUMBER_OF_TOP_Tens</th>
      <th>POINTS_BEHIND_LEAD</th>
      <th>ROUNDS_PLAYED</th>
      <th>SG_PUTTING_PER_ROUND</th>
      <th>TOTAL_SG:PUTTING</th>
      <th>MEASURED_ROUNDS</th>
      <th>AVG_Driving_DISTANCE</th>
      <th>UP_AND_DOWN_%</th>
      <th>PAR_OR_BETTER</th>
      <th>MISSED_GIR</th>
      <th>FAIRWAY_HIT_%</th>
      <th>FAIRWAYS_HIT</th>
      <th>POSSIBLE_FAIRWAYS</th>
      <th>GIR_RANK</th>
      <th>GOING_FOR_GREEN_IN_2%</th>
      <th>ATTEMPTS_GFG</th>
      <th>NON-ATTEMPTS_GFG</th>
      <th>RTP-GOING_FOR_THE_GREEN</th>
      <th>RTP-NOT_GOING_FOR_THE_GRN</th>
      <th>HOLE_OUTS</th>
      <th>SAND_SAVE%</th>
      <th>NUMBER_OF_SAVES</th>
      <th>NUMBER_OF_BUNKERS</th>
      <th>TOTAL_O/U_PAR</th>
      <th>Three_PUTT%</th>
      <th>TOTAL_3_PUTTS</th>
      <th>SG_PER_ROUND</th>
      <th>SG:OTT</th>
      <th>SG:APR</th>
      <th>SG:ARG</th>
      <th>DRIVES_320+%</th>
      <th>TOTAL_DRIVES_FOR_320+</th>
      <th>TOTAL_DRIVES</th>
      <th>ROUGH_TENDNECY%</th>
      <th>TOTAL_ROUGH</th>
      <th>FAIRWAY_BUNKER%</th>
      <th>TOTAL_FAIRWAY_BUNKERS</th>
      <th>AVG_CLUB_HEAD_SPEED</th>
      <th>FASTEST_CH_SPEED</th>
      <th>SLOWEST_CH_SPEED</th>
      <th>AVG_BALL_SPEED</th>
      <th>FASTEST_BALL_SPEED</th>
      <th>SLOWEST_BALL_SPEED</th>
      <th>AVG_SMASH_FACTOR</th>
      <th>HIGHEST_SF</th>
      <th>LOWEST_SF</th>
      <th>AVG_LAUNCH_ANGLE</th>
      <th>LOWEST_LAUNCH_ANGLE</th>
      <th>STEEPEST_LAUNCH_ANGLE</th>
      <th>AVG_SPIN_RATE</th>
      <th>HIGHEST_SPIN_RATE</th>
      <th>LOWEST_SPIN_RATE</th>
      <th>AVG_HANG_TIME</th>
      <th>LONGEST_ACT.HANG_TIME</th>
      <th>SHORTEST_ACT.HANG_TIME</th>
      <th>AVG_CARRY_DISTANCE</th>
      <th>LONGEST_CARRY_DISTANCE</th>
      <th>SHORTEST_CARRY_DISTANCE</th>
      <th>AVG_SCORE</th>
      <th>TOTAL_STROKES</th>
      <th>TOTAL_ROUNDS</th>
      <th>MAKES_BOGEY%</th>
      <th>BOGEYS_MADE</th>
      <th>HOLES_PLAYED</th>
      <th>AGE</th>
      <th>MONEY</th>
      <th>COUNTRY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>Charl Schwartzel</td>
      <td>20</td>
      <td>528</td>
      <td>0</td>
      <td>3</td>
      <td>5089.0</td>
      <td>65</td>
      <td>0.048</td>
      <td>1.932</td>
      <td>40</td>
      <td>299.5</td>
      <td>58.40</td>
      <td>226</td>
      <td>387</td>
      <td>54.86</td>
      <td>429</td>
      <td>782</td>
      <td>183</td>
      <td>62.50</td>
      <td>95</td>
      <td>57</td>
      <td>-46</td>
      <td>2</td>
      <td>4</td>
      <td>48.87</td>
      <td>65</td>
      <td>133</td>
      <td>42</td>
      <td>2.97</td>
      <td>31</td>
      <td>-0.205</td>
      <td>-0.075</td>
      <td>-0.092</td>
      <td>-0.038</td>
      <td>15.83</td>
      <td>88</td>
      <td>556</td>
      <td>32.36</td>
      <td>177</td>
      <td>7.5</td>
      <td>41</td>
      <td>118.79</td>
      <td>122.29</td>
      <td>115.56</td>
      <td>175.55</td>
      <td>179.35</td>
      <td>168.95</td>
      <td>1.478</td>
      <td>1.512</td>
      <td>1.435</td>
      <td>12.18</td>
      <td>7.49</td>
      <td>16.77</td>
      <td>2849.8</td>
      <td>5006</td>
      <td>2064</td>
      <td>6.1</td>
      <td>8.0</td>
      <td>4.3</td>
      <td>287.4</td>
      <td>311.9</td>
      <td>267.7</td>
      <td>71.349</td>
      <td>4160</td>
      <td>58</td>
      <td>18.87</td>
      <td>197</td>
      <td>1044</td>
      <td>34</td>
      <td>1710179</td>
      <td>South Africa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Andrew Putnam</td>
      <td>27</td>
      <td>1063</td>
      <td>1</td>
      <td>5</td>
      <td>4554.0</td>
      <td>93</td>
      <td>0.023</td>
      <td>1.655</td>
      <td>73</td>
      <td>294.2</td>
      <td>60.92</td>
      <td>279</td>
      <td>458</td>
      <td>60.40</td>
      <td>717</td>
      <td>1187</td>
      <td>21</td>
      <td>53.36</td>
      <td>143</td>
      <td>125</td>
      <td>-65</td>
      <td>-7</td>
      <td>10</td>
      <td>49.18</td>
      <td>60</td>
      <td>122</td>
      <td>38</td>
      <td>3.00</td>
      <td>48</td>
      <td>0.413</td>
      <td>-0.104</td>
      <td>0.357</td>
      <td>0.161</td>
      <td>5.95</td>
      <td>64</td>
      <td>1076</td>
      <td>31.62</td>
      <td>338</td>
      <td>6.0</td>
      <td>64</td>
      <td>113.36</td>
      <td>118.91</td>
      <td>108.36</td>
      <td>167.31</td>
      <td>173.17</td>
      <td>162.47</td>
      <td>1.476</td>
      <td>1.512</td>
      <td>1.400</td>
      <td>13.28</td>
      <td>8.51</td>
      <td>17.35</td>
      <td>2790.5</td>
      <td>6625</td>
      <td>1461</td>
      <td>6.4</td>
      <td>7.4</td>
      <td>4.4</td>
      <td>272.6</td>
      <td>296.0</td>
      <td>248.0</td>
      <td>70.579</td>
      <td>6239</td>
      <td>89</td>
      <td>14.17</td>
      <td>227</td>
      <td>1602</td>
      <td>29</td>
      <td>2243382</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>159</th>
      <td>Satoshi Kodaira</td>
      <td>18</td>
      <td>600</td>
      <td>1</td>
      <td>1</td>
      <td>5017.0</td>
      <td>51</td>
      <td>-0.536</td>
      <td>-20.386</td>
      <td>38</td>
      <td>293.2</td>
      <td>52.76</td>
      <td>172</td>
      <td>326</td>
      <td>64.97</td>
      <td>421</td>
      <td>648</td>
      <td>186</td>
      <td>48.06</td>
      <td>62</td>
      <td>67</td>
      <td>-23</td>
      <td>6</td>
      <td>4</td>
      <td>40.24</td>
      <td>33</td>
      <td>82</td>
      <td>43</td>
      <td>5.09</td>
      <td>44</td>
      <td>-1.183</td>
      <td>0.026</td>
      <td>-0.285</td>
      <td>-0.924</td>
      <td>8.08</td>
      <td>43</td>
      <td>532</td>
      <td>26.74</td>
      <td>142</td>
      <td>5.1</td>
      <td>27</td>
      <td>110.87</td>
      <td>113.37</td>
      <td>107.70</td>
      <td>165.13</td>
      <td>169.19</td>
      <td>159.88</td>
      <td>1.489</td>
      <td>1.516</td>
      <td>1.461</td>
      <td>9.77</td>
      <td>6.25</td>
      <td>12.91</td>
      <td>2458.1</td>
      <td>4203</td>
      <td>1526</td>
      <td>6.3</td>
      <td>7.3</td>
      <td>4.4</td>
      <td>276.8</td>
      <td>304.3</td>
      <td>247.1</td>
      <td>72.182</td>
      <td>3467</td>
      <td>48</td>
      <td>23.84</td>
      <td>206</td>
      <td>864</td>
      <td>34</td>
      <td>1471462</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Brian Harman</td>
      <td>24</td>
      <td>1116</td>
      <td>0</td>
      <td>8</td>
      <td>4501.0</td>
      <td>90</td>
      <td>0.536</td>
      <td>34.332</td>
      <td>64</td>
      <td>292.6</td>
      <td>57.96</td>
      <td>273</td>
      <td>471</td>
      <td>66.20</td>
      <td>764</td>
      <td>1154</td>
      <td>60</td>
      <td>53.71</td>
      <td>123</td>
      <td>106</td>
      <td>-69</td>
      <td>-13</td>
      <td>9</td>
      <td>48.87</td>
      <td>65</td>
      <td>133</td>
      <td>37</td>
      <td>3.33</td>
      <td>54</td>
      <td>0.038</td>
      <td>0.170</td>
      <td>-0.029</td>
      <td>-0.103</td>
      <td>10.15</td>
      <td>97</td>
      <td>956</td>
      <td>24.58</td>
      <td>234</td>
      <td>6.5</td>
      <td>62</td>
      <td>109.93</td>
      <td>116.41</td>
      <td>107.31</td>
      <td>164.02</td>
      <td>167.14</td>
      <td>160.94</td>
      <td>1.492</td>
      <td>1.516</td>
      <td>1.424</td>
      <td>12.44</td>
      <td>7.43</td>
      <td>16.78</td>
      <td>2509.7</td>
      <td>6055</td>
      <td>1942</td>
      <td>6.3</td>
      <td>7.4</td>
      <td>3.3</td>
      <td>271.1</td>
      <td>294.0</td>
      <td>246.1</td>
      <td>70.536</td>
      <td>6341</td>
      <td>90</td>
      <td>16.85</td>
      <td>273</td>
      <td>1620</td>
      <td>31</td>
      <td>2715103</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Brian Gay</td>
      <td>29</td>
      <td>910</td>
      <td>0</td>
      <td>6</td>
      <td>4707.0</td>
      <td>99</td>
      <td>0.480</td>
      <td>40.290</td>
      <td>84</td>
      <td>282.8</td>
      <td>63.93</td>
      <td>381</td>
      <td>596</td>
      <td>70.36</td>
      <td>914</td>
      <td>1299</td>
      <td>147</td>
      <td>37.79</td>
      <td>113</td>
      <td>186</td>
      <td>-71</td>
      <td>-36</td>
      <td>15</td>
      <td>51.88</td>
      <td>69</td>
      <td>133</td>
      <td>50</td>
      <td>2.20</td>
      <td>40</td>
      <td>0.103</td>
      <td>-0.136</td>
      <td>-0.019</td>
      <td>0.257</td>
      <td>2.28</td>
      <td>28</td>
      <td>1230</td>
      <td>21.84</td>
      <td>268</td>
      <td>5.1</td>
      <td>63</td>
      <td>105.76</td>
      <td>109.25</td>
      <td>102.15</td>
      <td>157.17</td>
      <td>160.26</td>
      <td>153.51</td>
      <td>1.486</td>
      <td>1.512</td>
      <td>1.442</td>
      <td>12.47</td>
      <td>9.21</td>
      <td>15.50</td>
      <td>2339.9</td>
      <td>5462</td>
      <td>1606</td>
      <td>6.4</td>
      <td>7.5</td>
      <td>3.6</td>
      <td>261.5</td>
      <td>289.6</td>
      <td>236.0</td>
      <td>70.280</td>
      <td>7062</td>
      <td>101</td>
      <td>14.96</td>
      <td>272</td>
      <td>1818</td>
      <td>46</td>
      <td>2126761</td>
      <td>United States</td>
    </tr>
  </tbody>
</table>


Descriptive Statistics across all columns


```python
df.describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EVENTS_PLAYED</th>
      <th>POINTS</th>
      <th>NUMBER_OF_WINS</th>
      <th>NUMBER_OF_TOP_Tens</th>
      <th>POINTS_BEHIND_LEAD</th>
      <th>ROUNDS_PLAYED</th>
      <th>SG_PUTTING_PER_ROUND</th>
      <th>TOTAL_SG:PUTTING</th>
      <th>MEASURED_ROUNDS</th>
      <th>AVG_Driving_DISTANCE</th>
      <th>UP_AND_DOWN_%</th>
      <th>PAR_OR_BETTER</th>
      <th>MISSED_GIR</th>
      <th>FAIRWAY_HIT_%</th>
      <th>FAIRWAYS_HIT</th>
      <th>POSSIBLE_FAIRWAYS</th>
      <th>GIR_RANK</th>
      <th>GOING_FOR_GREEN_IN_2%</th>
      <th>ATTEMPTS_GFG</th>
      <th>NON-ATTEMPTS_GFG</th>
      <th>RTP-GOING_FOR_THE_GREEN</th>
      <th>RTP-NOT_GOING_FOR_THE_GRN</th>
      <th>HOLE_OUTS</th>
      <th>SAND_SAVE%</th>
      <th>NUMBER_OF_SAVES</th>
      <th>NUMBER_OF_BUNKERS</th>
      <th>TOTAL_O/U_PAR</th>
      <th>Three_PUTT%</th>
      <th>TOTAL_3_PUTTS</th>
      <th>SG_PER_ROUND</th>
      <th>SG:OTT</th>
      <th>SG:APR</th>
      <th>SG:ARG</th>
      <th>DRIVES_320+%</th>
      <th>TOTAL_DRIVES_FOR_320+</th>
      <th>TOTAL_DRIVES</th>
      <th>ROUGH_TENDNECY%</th>
      <th>TOTAL_ROUGH</th>
      <th>FAIRWAY_BUNKER%</th>
      <th>TOTAL_FAIRWAY_BUNKERS</th>
      <th>AVG_CLUB_HEAD_SPEED</th>
      <th>FASTEST_CH_SPEED</th>
      <th>SLOWEST_CH_SPEED</th>
      <th>AVG_BALL_SPEED</th>
      <th>FASTEST_BALL_SPEED</th>
      <th>SLOWEST_BALL_SPEED</th>
      <th>AVG_SMASH_FACTOR</th>
      <th>HIGHEST_SF</th>
      <th>LOWEST_SF</th>
      <th>AVG_LAUNCH_ANGLE</th>
      <th>LOWEST_LAUNCH_ANGLE</th>
      <th>STEEPEST_LAUNCH_ANGLE</th>
      <th>AVG_SPIN_RATE</th>
      <th>HIGHEST_SPIN_RATE</th>
      <th>LOWEST_SPIN_RATE</th>
      <th>AVG_HANG_TIME</th>
      <th>LONGEST_ACT.HANG_TIME</th>
      <th>SHORTEST_ACT.HANG_TIME</th>
      <th>AVG_CARRY_DISTANCE</th>
      <th>LONGEST_CARRY_DISTANCE</th>
      <th>SHORTEST_CARRY_DISTANCE</th>
      <th>AVG_SCORE</th>
      <th>TOTAL_STROKES</th>
      <th>TOTAL_ROUNDS</th>
      <th>MAKES_BOGEY%</th>
      <th>BOGEYS_MADE</th>
      <th>HOLES_PLAYED</th>
      <th>AGE</th>
      <th>MONEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>194.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.00000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>1.950000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.174359</td>
      <td>745.974359</td>
      <td>0.241026</td>
      <td>2.651282</td>
      <td>4896.262887</td>
      <td>78.358974</td>
      <td>0.051708</td>
      <td>3.111672</td>
      <td>59.928205</td>
      <td>296.756923</td>
      <td>58.597897</td>
      <td>256.615385</td>
      <td>437.887179</td>
      <td>61.053846</td>
      <td>620.907692</td>
      <td>1015.025641</td>
      <td>97.923077</td>
      <td>56.224821</td>
      <td>124.148718</td>
      <td>96.692308</td>
      <td>-67.748718</td>
      <td>-6.779487</td>
      <td>10.994872</td>
      <td>49.934462</td>
      <td>61.984615</td>
      <td>124.164103</td>
      <td>40.189744</td>
      <td>2.866923</td>
      <td>38.815385</td>
      <td>0.136246</td>
      <td>0.042308</td>
      <td>0.061779</td>
      <td>0.032108</td>
      <td>10.817179</td>
      <td>92.230769</td>
      <td>859.584615</td>
      <td>28.773949</td>
      <td>243.964103</td>
      <td>6.109744</td>
      <td>52.128205</td>
      <td>113.997846</td>
      <td>118.370051</td>
      <td>110.116872</td>
      <td>169.622667</td>
      <td>174.298513</td>
      <td>163.605231</td>
      <td>1.487728</td>
      <td>1.515636</td>
      <td>1.431810</td>
      <td>11.060974</td>
      <td>7.00041</td>
      <td>14.887333</td>
      <td>2633.989744</td>
      <td>5085.112821</td>
      <td>1736.723077</td>
      <td>6.347179</td>
      <td>7.638462</td>
      <td>3.209744</td>
      <td>278.155385</td>
      <td>305.030769</td>
      <td>244.854359</td>
      <td>70.893369</td>
      <td>5328.810256</td>
      <td>75.497436</td>
      <td>16.483897</td>
      <td>222.194872</td>
      <td>1359.000000</td>
      <td>32.887179</td>
      <td>1.653204e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.151805</td>
      <td>700.518059</td>
      <td>0.599277</td>
      <td>2.201554</td>
      <td>608.024609</td>
      <td>14.039134</td>
      <td>0.313897</td>
      <td>18.320074</td>
      <td>13.997606</td>
      <td>8.191270</td>
      <td>3.473569</td>
      <td>53.353816</td>
      <td>86.863725</td>
      <td>5.079533</td>
      <td>139.350784</td>
      <td>201.438703</td>
      <td>56.456709</td>
      <td>9.138289</td>
      <td>34.816950</td>
      <td>31.008108</td>
      <td>21.775647</td>
      <td>9.887907</td>
      <td>4.345275</td>
      <td>5.791218</td>
      <td>16.237121</td>
      <td>29.040742</td>
      <td>13.129407</td>
      <td>0.642765</td>
      <td>10.674991</td>
      <td>0.752243</td>
      <td>0.418211</td>
      <td>0.396112</td>
      <td>0.220894</td>
      <td>6.999289</td>
      <td>62.617256</td>
      <td>195.369058</td>
      <td>3.939820</td>
      <td>59.698329</td>
      <td>1.105403</td>
      <td>15.032575</td>
      <td>3.902298</td>
      <td>4.183791</td>
      <td>3.819907</td>
      <td>5.764802</td>
      <td>5.814774</td>
      <td>6.076666</td>
      <td>0.014070</td>
      <td>0.005842</td>
      <td>0.026439</td>
      <td>1.382925</td>
      <td>1.63012</td>
      <td>1.631474</td>
      <td>209.331951</td>
      <td>1364.666542</td>
      <td>219.316194</td>
      <td>0.252676</td>
      <td>0.331985</td>
      <td>1.155827</td>
      <td>10.060564</td>
      <td>12.552019</td>
      <td>13.149478</td>
      <td>0.820646</td>
      <td>1005.287126</td>
      <td>14.465520</td>
      <td>2.095008</td>
      <td>41.704446</td>
      <td>260.378571</td>
      <td>5.668469</td>
      <td>1.489543e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2328.000000</td>
      <td>46.000000</td>
      <td>-0.750000</td>
      <td>-42.673000</td>
      <td>30.000000</td>
      <td>278.400000</td>
      <td>44.010000</td>
      <td>136.000000</td>
      <td>227.000000</td>
      <td>40.850000</td>
      <td>268.000000</td>
      <td>531.000000</td>
      <td>1.000000</td>
      <td>28.360000</td>
      <td>38.000000</td>
      <td>35.000000</td>
      <td>-128.000000</td>
      <td>-51.000000</td>
      <td>1.000000</td>
      <td>33.800000</td>
      <td>23.000000</td>
      <td>57.000000</td>
      <td>15.000000</td>
      <td>1.480000</td>
      <td>12.000000</td>
      <td>-3.586000</td>
      <td>-1.585000</td>
      <td>-1.586000</td>
      <td>-0.924000</td>
      <td>1.240000</td>
      <td>8.000000</td>
      <td>420.000000</td>
      <td>16.670000</td>
      <td>93.000000</td>
      <td>3.100000</td>
      <td>18.000000</td>
      <td>105.270000</td>
      <td>109.160000</td>
      <td>102.150000</td>
      <td>157.170000</td>
      <td>160.260000</td>
      <td>146.500000</td>
      <td>1.423000</td>
      <td>1.473000</td>
      <td>1.337000</td>
      <td>7.140000</td>
      <td>1.25000</td>
      <td>10.700000</td>
      <td>2127.300000</td>
      <td>2819.000000</td>
      <td>1400.000000</td>
      <td>5.500000</td>
      <td>6.800000</td>
      <td>0.500000</td>
      <td>249.800000</td>
      <td>271.600000</td>
      <td>192.800000</td>
      <td>68.702000</td>
      <td>3261.000000</td>
      <td>45.000000</td>
      <td>12.200000</td>
      <td>123.000000</td>
      <td>810.000000</td>
      <td>21.000000</td>
      <td>2.487800e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.000000</td>
      <td>267.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4592.750000</td>
      <td>67.000000</td>
      <td>-0.146500</td>
      <td>-8.024000</td>
      <td>49.000000</td>
      <td>291.050000</td>
      <td>56.475000</td>
      <td>214.500000</td>
      <td>373.500000</td>
      <td>57.255000</td>
      <td>522.000000</td>
      <td>857.000000</td>
      <td>49.500000</td>
      <td>50.485000</td>
      <td>100.500000</td>
      <td>74.500000</td>
      <td>-80.000000</td>
      <td>-12.000000</td>
      <td>8.000000</td>
      <td>45.965000</td>
      <td>50.500000</td>
      <td>103.000000</td>
      <td>31.000000</td>
      <td>2.405000</td>
      <td>31.500000</td>
      <td>-0.217500</td>
      <td>-0.144500</td>
      <td>-0.150500</td>
      <td>-0.124000</td>
      <td>5.765000</td>
      <td>48.000000</td>
      <td>702.000000</td>
      <td>26.235000</td>
      <td>199.000000</td>
      <td>5.400000</td>
      <td>41.000000</td>
      <td>111.290000</td>
      <td>115.570000</td>
      <td>107.355000</td>
      <td>165.720000</td>
      <td>170.455000</td>
      <td>159.415000</td>
      <td>1.483000</td>
      <td>1.514000</td>
      <td>1.417000</td>
      <td>10.205000</td>
      <td>5.88000</td>
      <td>13.720000</td>
      <td>2508.300000</td>
      <td>4127.500000</td>
      <td>1541.500000</td>
      <td>6.200000</td>
      <td>7.500000</td>
      <td>2.400000</td>
      <td>270.900000</td>
      <td>295.600000</td>
      <td>237.050000</td>
      <td>70.440500</td>
      <td>4529.000000</td>
      <td>64.000000</td>
      <td>15.205000</td>
      <td>193.500000</td>
      <td>1152.000000</td>
      <td>29.000000</td>
      <td>5.752075e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.000000</td>
      <td>587.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>5037.000000</td>
      <td>80.000000</td>
      <td>0.059000</td>
      <td>3.967000</td>
      <td>60.000000</td>
      <td>296.200000</td>
      <td>58.480000</td>
      <td>259.000000</td>
      <td>445.000000</td>
      <td>61.090000</td>
      <td>613.000000</td>
      <td>1031.000000</td>
      <td>98.000000</td>
      <td>56.710000</td>
      <td>124.000000</td>
      <td>96.000000</td>
      <td>-67.000000</td>
      <td>-7.000000</td>
      <td>10.000000</td>
      <td>49.650000</td>
      <td>62.000000</td>
      <td>125.000000</td>
      <td>38.000000</td>
      <td>2.840000</td>
      <td>38.000000</td>
      <td>0.143000</td>
      <td>0.073000</td>
      <td>0.059000</td>
      <td>0.022000</td>
      <td>8.720000</td>
      <td>75.000000</td>
      <td>864.000000</td>
      <td>28.760000</td>
      <td>243.000000</td>
      <td>6.100000</td>
      <td>52.000000</td>
      <td>113.630000</td>
      <td>118.000000</td>
      <td>109.900000</td>
      <td>169.120000</td>
      <td>173.980000</td>
      <td>163.200000</td>
      <td>1.490000</td>
      <td>1.517000</td>
      <td>1.432000</td>
      <td>11.110000</td>
      <td>6.96000</td>
      <td>14.850000</td>
      <td>2625.800000</td>
      <td>4931.000000</td>
      <td>1714.000000</td>
      <td>6.400000</td>
      <td>7.600000</td>
      <td>3.400000</td>
      <td>278.200000</td>
      <td>304.500000</td>
      <td>245.900000</td>
      <td>70.850000</td>
      <td>5379.000000</td>
      <td>76.000000</td>
      <td>16.280000</td>
      <td>223.000000</td>
      <td>1368.000000</td>
      <td>33.000000</td>
      <td>1.287040e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>27.000000</td>
      <td>1038.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>5349.750000</td>
      <td>88.500000</td>
      <td>0.237000</td>
      <td>14.503000</td>
      <td>70.000000</td>
      <td>301.650000</td>
      <td>60.925000</td>
      <td>295.000000</td>
      <td>492.000000</td>
      <td>64.430000</td>
      <td>713.000000</td>
      <td>1164.000000</td>
      <td>146.500000</td>
      <td>62.450000</td>
      <td>150.000000</td>
      <td>114.000000</td>
      <td>-55.000000</td>
      <td>-1.000000</td>
      <td>14.000000</td>
      <td>53.375000</td>
      <td>73.000000</td>
      <td>143.500000</td>
      <td>49.000000</td>
      <td>3.195000</td>
      <td>45.000000</td>
      <td>0.562500</td>
      <td>0.294000</td>
      <td>0.324500</td>
      <td>0.197500</td>
      <td>13.990000</td>
      <td>121.000000</td>
      <td>999.000000</td>
      <td>31.220000</td>
      <td>281.500000</td>
      <td>6.800000</td>
      <td>62.000000</td>
      <td>116.770000</td>
      <td>120.880000</td>
      <td>112.565000</td>
      <td>173.435000</td>
      <td>178.170000</td>
      <td>167.510000</td>
      <td>1.498000</td>
      <td>1.518500</td>
      <td>1.450000</td>
      <td>12.095000</td>
      <td>8.09000</td>
      <td>16.025000</td>
      <td>2757.700000</td>
      <td>5869.000000</td>
      <td>1900.000000</td>
      <td>6.500000</td>
      <td>7.800000</td>
      <td>4.100000</td>
      <td>283.950000</td>
      <td>313.600000</td>
      <td>253.100000</td>
      <td>71.324000</td>
      <td>6106.500000</td>
      <td>87.000000</td>
      <td>17.420000</td>
      <td>251.000000</td>
      <td>1566.000000</td>
      <td>36.000000</td>
      <td>2.242648e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>32.000000</td>
      <td>5617.000000</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>5607.000000</td>
      <td>110.000000</td>
      <td>0.862000</td>
      <td>60.061000</td>
      <td>92.000000</td>
      <td>320.200000</td>
      <td>66.590000</td>
      <td>392.000000</td>
      <td>635.000000</td>
      <td>74.330000</td>
      <td>1060.000000</td>
      <td>1454.000000</td>
      <td>195.000000</td>
      <td>78.440000</td>
      <td>227.000000</td>
      <td>228.000000</td>
      <td>-22.000000</td>
      <td>27.000000</td>
      <td>25.000000</td>
      <td>66.000000</td>
      <td>108.000000</td>
      <td>193.000000</td>
      <td>74.000000</td>
      <td>5.090000</td>
      <td>73.000000</td>
      <td>1.987000</td>
      <td>1.006000</td>
      <td>0.990000</td>
      <td>0.632000</td>
      <td>33.920000</td>
      <td>307.000000</td>
      <td>1344.000000</td>
      <td>42.640000</td>
      <td>428.000000</td>
      <td>9.500000</td>
      <td>94.000000</td>
      <td>124.670000</td>
      <td>129.200000</td>
      <td>118.730000</td>
      <td>182.220000</td>
      <td>187.500000</td>
      <td>177.340000</td>
      <td>1.507000</td>
      <td>1.539000</td>
      <td>1.492000</td>
      <td>14.710000</td>
      <td>11.32000</td>
      <td>18.620000</td>
      <td>3346.100000</td>
      <td>9640.000000</td>
      <td>2314.000000</td>
      <td>6.900000</td>
      <td>8.700000</td>
      <td>5.100000</td>
      <td>302.600000</td>
      <td>337.700000</td>
      <td>275.700000</td>
      <td>74.891000</td>
      <td>7515.000000</td>
      <td>107.000000</td>
      <td>28.250000</td>
      <td>330.000000</td>
      <td>1926.000000</td>
      <td>49.000000</td>
      <td>8.225921e+06</td>
    </tr>
  </tbody>
</table>


Assess for null/empty values


```python
num_na = df.isna().sum()
num_null = df.isnull().sum()
sort_num_na = num_na.sort_values(ascending=False)
sort_num_null = num_null.sort_values(ascending=False)
print(sort_num_na)
print(sort_num_null)
```

    POINTS_BEHIND_LEAD    1
    COUNTRY               0
    NUMBER_OF_SAVES       0
    ATTEMPTS_GFG          0
    NON-ATTEMPTS_GFG      0
                         ..
    AVG_SMASH_FACTOR      0
    HIGHEST_SF            0
    LOWEST_SF             0
    AVG_LAUNCH_ANGLE      0
    Player                0
    Length: 71, dtype: int64
    POINTS_BEHIND_LEAD    1
    COUNTRY               0
    NUMBER_OF_SAVES       0
    ATTEMPTS_GFG          0
    NON-ATTEMPTS_GFG      0
                         ..
    AVG_SMASH_FACTOR      0
    HIGHEST_SF            0
    LOWEST_SF             0
    AVG_LAUNCH_ANGLE      0
    Player                0
    Length: 71, dtype: int64


Only empty value has no consequence to analysis

## EDA of driving distance statistics

Descriptive Statistics of average driving distance column


```python
df.AVG_Driving_DISTANCE.describe()
```


    count    195.000000
    mean     296.756923
    std        8.191270
    min      278.400000
    25%      291.050000
    50%      296.200000
    75%      301.650000
    max      320.200000
    Name: AVG_Driving_DISTANCE, dtype: float64

Separate data into groups based on average driving distance for future analysis

Group 1: Over 300 yard average driving distance 

Group 2: Under 300 yard average driving distance 

Group 3: Top 50% average driving distance

Group 4: Bottom 50% average driving distance 


```python
over_300_avg_driving_dist = df.AVG_Driving_DISTANCE >= 300
under_300_avg_driving_dist = df.AVG_Driving_DISTANCE < 300
top_50_driving_dist = df.AVG_Driving_DISTANCE >= 296.20
bottom_50_driving_dist = df.AVG_Driving_DISTANCE <= 296.20
```

Create new column for over 300 yard and top 50% drivers for future analysis


```python
df['Over_300_yards']=over_300_avg_driving_dist
df['Top_50']=top_50_driving_dist
```

Create new dataframes for new groups


```python
over_300_df = df.iloc[over_300_avg_driving_dist.values]
under_300_df = df.iloc[under_300_avg_driving_dist.values]
top_50_df = df.iloc[top_50_driving_dist.values]
bottom_50_df = df.iloc[bottom_50_driving_dist.values]
```


```python
over_300_df.count()
```


    Player                62
    EVENTS_PLAYED         62
    POINTS                62
    NUMBER_OF_WINS        62
    NUMBER_OF_TOP_Tens    62
                          ..
    AGE                   62
    MONEY                 62
    COUNTRY               62
    Over_300_yards        62
    Top_50                62
    Length: 73, dtype: int64




```python
percent_over300 = (62/195) * 100
print(round(percent_over300, 2), 'percent of the dataset averages over 300 yards in driving distance.')
```

    31.79 percent of the dataset averages over 300 yards in driving distance.


Confirm top 50 and bottom 50 dataframes are split evenly 


```python
top_50_df.count()
```


    Player                99
    EVENTS_PLAYED         99
    POINTS                99
    NUMBER_OF_WINS        99
    NUMBER_OF_TOP_Tens    99
                          ..
    AGE                   99
    MONEY                 99
    COUNTRY               99
    Over_300_yards        99
    Top_50                99
    Length: 73, dtype: int64




```python
bottom_50_df.count()
```


    Player                98
    EVENTS_PLAYED         98
    POINTS                98
    NUMBER_OF_WINS        98
    NUMBER_OF_TOP_Tens    98
                          ..
    AGE                   98
    MONEY                 98
    COUNTRY               98
    Over_300_yards        98
    Top_50                98
    Length: 73, dtype: int64



## Distribution of data

Obtain skewness and kurtosis values of data distribution


```python
skew = df.skew(axis=0)
kurt = df.kurt(axis=0)
```



---


Visualize data distribution for average driving distance


```python
sns.distplot(df['AVG_Driving_DISTANCE'], kde=True)

print(f'The skewness is {round(skew.AVG_Driving_DISTANCE, 2)} and kurtosis is {round(kurt.AVG_Driving_DISTANCE,2)}.')
```

    The skewness is 0.37 and kurtosis is 0.04.




![png](images/output_49_1.png)
    


We see a normal distribution of the AVG_Driving_DISTANCE data



---


Visualize data distribution for Money


```python
sns.distplot(df['MONEY'])

print(f'The skewness is {round(skew.MONEY, 2)} and kurtosis is {round(kurt.MONEY,2)}.')
```

    The skewness is 1.95 and kurtosis is 4.81.




![png](images/output_52_1.png)
    


There is a high positive skewness and kurtosis with the distribution of the MONEY data



---


Visualize data distribution for strokes gained per round


```python
sns.distplot(df['SG_PER_ROUND'])

print(f'The skewness is {round(skew.SG_PER_ROUND, 2)} and kurtosis is {round(kurt.SG_PER_ROUND,2)}.')
```

    The skewness is -0.94 and kurtosis is 3.57.




![png](images/output_55_1.png)
    


There is a slight negative skewness and moderate kurtosis with the distribution of the SG_PER_ROUND data



---


Visualize data distribution for average score


```python
sns.distplot(df['AVG_SCORE'])

print(f'The skewness is {round(skew.AVG_SCORE, 2)} and kurtosis is {round(kurt.AVG_SCORE,2)}.')
```

    The skewness is 0.78 and kurtosis is 3.15.




![png](images/output_58_1.png)
    


We see a normal distribution of the AVG_SCORE data with slightly postivie skewness and moderate kurtosis

## Hypothesis 1 Analysis


*   **HO = Players with higher average driving distance will not have more success on PGA Tour.**
*   **HA = Players with higher average driving distance will have more success on PGA Tour.** 




For purpose of this analysis, success on PGA Tour will include numbers of wins, top 10s, and money earned.



---


Create list of variables for H1 analysis


```python
h1_list = (['AVG_Driving_DISTANCE', 'NUMBER_OF_WINS', 'NUMBER_OF_TOP_Tens', 'MONEY'])
```



---


Descriptive Statistics for H1 in Bottom 50%, Top 50%, and Over 300 yard driving distance groupings


```python
bottom_50_df[h1_list].describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>NUMBER_OF_WINS</th>
      <th>NUMBER_OF_TOP_Tens</th>
      <th>MONEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>9.800000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>290.298980</td>
      <td>0.142857</td>
      <td>2.122449</td>
      <td>1.204137e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.222143</td>
      <td>0.351726</td>
      <td>1.812132</td>
      <td>9.446331e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>278.400000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.487800e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>287.950000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.411060e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>291.050000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>9.386740e+05</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>293.350000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.815034e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>296.200000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>4.691667e+06</td>
    </tr>
  </tbody>
</table>



```python
top_50_df[h1_list].describe()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>NUMBER_OF_WINS</th>
      <th>NUMBER_OF_TOP_Tens</th>
      <th>MONEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>9.900000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>303.138384</td>
      <td>0.333333</td>
      <td>3.181818</td>
      <td>2.107215e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.687055</td>
      <td>0.755929</td>
      <td>2.421686</td>
      <td>1.776043e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>296.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.910400e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>7.952340e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>301.600000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.595942e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>305.900000</td>
      <td>0.000000</td>
      <td>4.500000</td>
      <td>2.761220e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>320.200000</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>8.225921e+06</td>
    </tr>
  </tbody>
</table>



```python
over_300_df[h1_list].describe()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>NUMBER_OF_WINS</th>
      <th>NUMBER_OF_TOP_Tens</th>
      <th>MONEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>6.200000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>306.083871</td>
      <td>0.435484</td>
      <td>3.564516</td>
      <td>2.373406e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.240172</td>
      <td>0.880033</td>
      <td>2.551894</td>
      <td>2.008144e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>300.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.910400e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>302.425000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000609e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>303.800000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.611929e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>309.475000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>3.373650e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>320.200000</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>8.225921e+06</td>
    </tr>
  </tbody>
</table>


We see an increase in means and medians across wins, top 10s, and money won as the groupings increase in average driving distance. 

---


Next lets look at correlations between these variables


```python
h1_list_corr_df = df[h1_list].corr()
h1_list_corr_df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>NUMBER_OF_WINS</th>
      <th>NUMBER_OF_TOP_Tens</th>
      <th>MONEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AVG_Driving_DISTANCE</th>
      <td>1.000000</td>
      <td>0.310112</td>
      <td>0.390101</td>
      <td>0.447671</td>
    </tr>
    <tr>
      <th>NUMBER_OF_WINS</th>
      <td>0.310112</td>
      <td>1.000000</td>
      <td>0.525058</td>
      <td>0.767855</td>
    </tr>
    <tr>
      <th>NUMBER_OF_TOP_Tens</th>
      <td>0.390101</td>
      <td>0.525058</td>
      <td>1.000000</td>
      <td>0.874703</td>
    </tr>
    <tr>
      <th>MONEY</th>
      <td>0.447671</td>
      <td>0.767855</td>
      <td>0.874703</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>


We notice positive correlations between average driving distance, wins, top 10s, and money won.  With highest correlation between average driving distance and money won at 0.447.

Heatmap of correlations with the more red being closer to .30 postitive correlation and the more green close to 1.0 positive correlation


```python
sns.heatmap(h1_list_corr_df, annot = True, cmap = 'RdYlGn')
plt.show()
```


​    
![png](images/output_73_0.png)
​    




---

Box and whisker plot of Wins vs Driving Distance


```python
plt.figure(figsize = (12,6))
sns.boxplot(x = 'NUMBER_OF_WINS', y='AVG_Driving_DISTANCE', data = df)
plt.xlabel('Number of Wins')
plt.ylabel('Average Driving Distance')
plt.title('Wins vs Driving Distance')
plt.show()
```


​    
![png](images/output_75_0.png)
​    


Visually see the positive correlation between wins and driving driving distance.  As wins increase, so does mean driving distance.  Also worth noting the whiskers associated with the box plots indicating a smaller standard deviation as wins increase. Thus telling us that higher driving distance players are the player winning more than one tournament during the season.  This is also verified in the above descriptive statistics in the bottom 50% driving distance grouping where the maximum wins in this groups is only 1.


---



Box and Whisker plot of Top 10s vs Driving Distance


```python
plt.figure(figsize = (12,6))
sns.boxplot(x = 'NUMBER_OF_TOP_Tens', y='AVG_Driving_DISTANCE', data = df)
plt.xlabel('# of Top 10 Finishes')
plt.ylabel('Average Driving Distance')
plt.title('Top 10s vs Driving Distance')
plt.show()
```


​    
![png](images/output_78_0.png)
​    


Again we visually see the positive correlation between top 10 finishes and average driving distance. As the number of top 10 finishes increases so does the mean of driving distance, with the except of 8 wins which only had one player.

---



Regression plot of Money Won vs Driving Distance


```python
plt.figure(figsize = (12,6))
sns.regplot(x = 'MONEY', y='AVG_Driving_DISTANCE', data = df)
plt.xlabel('Money Won (in millions)')
plt.ylabel('Average Driving Distance')
plt.title('Money Won vs Driving Distance')
plt.show()
```


​    
![png](images/output_81_0.png)
​    


Visually see the positive relationship between money won and driving distance.  As a player increases their driving distance, they tend to win more money on tour.  Especially with the high money winners in 2017 as only players in the top 50% driving distance won over 5 million dollars as the max won in the bottom 50% driving distance was 4.69 million. 


---




T-test analysis between means of top 50% and bottom 50% groupings from data sample


```python
for x in h1_list:
  print(f'The t-test results for {x} column comparing the top and bottom 50% in average driving distance are:')
  print(stats.ttest_ind(top_50_df[x], bottom_50_df[x]))
  print('\n')
```

    The t-test results for AVG_Driving_DISTANCE column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=17.977004473009526, pvalue=2.9831462094179926e-43)
    
    The t-test results for NUMBER_OF_WINS column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=2.263607206946361, pvalue=0.024699518256500137)
    
    The t-test results for NUMBER_OF_TOP_Tens column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=3.4735514944529537, pvalue=0.0006327165997008301)
    
    The t-test results for MONEY column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=4.449060442456906, pvalue=1.4480132101464633e-05)
    

​      


Statistical significance seen between the two groups in all 3 variables with all p-values <= 0.05.


---



T-test analysis between means of over 300 yard average and under 300 yard average groupings from data sample


```python
for y in h1_list:
  print(f'The t-test results for {y} column comparing the over and under average driving distance of 300 yards are:')
  print(stats.ttest_ind(over_300_df[y], under_300_df[y]))
  print('\n')
```

    The t-test results for AVG_Driving_DISTANCE column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=17.283678491926892, pvalue=4.681984277618193e-41)
    
    The t-test results for NUMBER_OF_WINS column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=3.164830495977456, pvalue=0.0018032590013044387)
      
    The t-test results for NUMBER_OF_TOP_Tens column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=4.11407182735084, pvalue=5.752419264872462e-05)
      
    The t-test results for MONEY column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=4.872578015825289, pvalue=2.2911126223226177e-06)


  ​        


Statistical significance seen between the two groups in all 3 variables with all p-values <= 0.05. All p-values indicating more statistical significance between over/under 300 yard groupings than between top/bottom 50% groupings. 


---

Summary and Conclusion of H1 Testing:


*   Higher mean and median wins, top 10s and money won in top 50% driving distance grouping compared to bottom 50% driving distance grouping.
*   Higher mean and median wins, top 10s and money won in over 300 yard average driving distance grouping compared to top 50% driving distance grouping.
*   High positive correlation across all 3 variables with increased driving distance.
*   Statistically significant with a p-value of 0.024, 0.0006, and 1.448e-05 in wins, top 10s, and money won between the top 50% and bottom 50% groupings, respectively.
*   More statistical significance between over/under 300 yard groupings than between top/bottom 50% groupings.
---
*   **Reject the null hypothesis and accept the alternate hypotheseis that players with higher average driving distance will have more success on PGA Tour.**



## Hypothesis 2 Analysis

*   **HO = Players with higher average driving distance will not have better strokes gained per round.**
*   **HA = Players with higher average driving distance will have better strokes gained per round.**

Create list of variables for H2 analysis for all strokes gained statistics


```python
h2_list = (['AVG_Driving_DISTANCE','SG_PER_ROUND', 'SG:OTT', 'SG:APR', 'SG:ARG', 'TOTAL_SG:PUTTING'])
```



---


Descriptive Statistics for H2 in Bottom 50%, Top 50%, and Over 300 yard driving distance groupings:


```python
bottom_50_df[h2_list].describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>SG_PER_ROUND</th>
      <th>SG:OTT</th>
      <th>SG:APR</th>
      <th>SG:ARG</th>
      <th>TOTAL_SG:PUTTING</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>98.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>290.298980</td>
      <td>-0.101908</td>
      <td>-0.139622</td>
      <td>0.008949</td>
      <td>0.028704</td>
      <td>4.740327</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.222143</td>
      <td>0.689527</td>
      <td>0.393084</td>
      <td>0.368360</td>
      <td>0.223597</td>
      <td>18.167656</td>
    </tr>
    <tr>
      <th>min</th>
      <td>278.400000</td>
      <td>-2.840000</td>
      <td>-1.443000</td>
      <td>-1.586000</td>
      <td>-0.924000</td>
      <td>-42.673000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>287.950000</td>
      <td>-0.486000</td>
      <td>-0.305000</td>
      <td>-0.166750</td>
      <td>-0.119000</td>
      <td>-6.083250</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>291.050000</td>
      <td>-0.020000</td>
      <td>-0.064500</td>
      <td>0.021500</td>
      <td>0.019500</td>
      <td>5.564000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>293.350000</td>
      <td>0.301250</td>
      <td>0.111750</td>
      <td>0.266250</td>
      <td>0.193000</td>
      <td>15.160000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>296.200000</td>
      <td>1.479000</td>
      <td>0.587000</td>
      <td>0.975000</td>
      <td>0.632000</td>
      <td>46.404000</td>
    </tr>
  </tbody>
</table>



```python
top_50_df[h2_list].describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>SG_PER_ROUND</th>
      <th>SG:OTT</th>
      <th>SG:APR</th>
      <th>SG:ARG</th>
      <th>TOTAL_SG:PUTTING</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>303.138384</td>
      <td>0.376444</td>
      <td>0.222525</td>
      <td>0.119495</td>
      <td>0.034374</td>
      <td>1.335343</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.687055</td>
      <td>0.734549</td>
      <td>0.362417</td>
      <td>0.414351</td>
      <td>0.217316</td>
      <td>18.327189</td>
    </tr>
    <tr>
      <th>min</th>
      <td>296.200000</td>
      <td>-3.586000</td>
      <td>-1.585000</td>
      <td>-1.504000</td>
      <td>-0.497000</td>
      <td>-40.194000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299.000000</td>
      <td>-0.046500</td>
      <td>0.027000</td>
      <td>-0.120500</td>
      <td>-0.124000</td>
      <td>-8.627000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>301.600000</td>
      <td>0.373000</td>
      <td>0.229000</td>
      <td>0.159000</td>
      <td>0.027000</td>
      <td>0.810000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>305.900000</td>
      <td>0.815000</td>
      <td>0.412000</td>
      <td>0.389500</td>
      <td>0.200500</td>
      <td>11.107000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>320.200000</td>
      <td>1.987000</td>
      <td>1.006000</td>
      <td>0.990000</td>
      <td>0.629000</td>
      <td>60.061000</td>
    </tr>
  </tbody>
</table>



```python
over_300_df[h2_list].describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>SG_PER_ROUND</th>
      <th>SG:OTT</th>
      <th>SG:APR</th>
      <th>SG:ARG</th>
      <th>TOTAL_SG:PUTTING</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>306.083871</td>
      <td>0.517323</td>
      <td>0.334113</td>
      <td>0.136629</td>
      <td>0.046548</td>
      <td>0.955419</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.240172</td>
      <td>0.689911</td>
      <td>0.312502</td>
      <td>0.425042</td>
      <td>0.212524</td>
      <td>19.276550</td>
    </tr>
    <tr>
      <th>min</th>
      <td>300.000000</td>
      <td>-1.020000</td>
      <td>-0.448000</td>
      <td>-1.039000</td>
      <td>-0.307000</td>
      <td>-40.194000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>302.425000</td>
      <td>0.005750</td>
      <td>0.172250</td>
      <td>-0.109250</td>
      <td>-0.121250</td>
      <td>-10.673750</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>303.800000</td>
      <td>0.490000</td>
      <td>0.318000</td>
      <td>0.154500</td>
      <td>0.000500</td>
      <td>0.779000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>309.475000</td>
      <td>1.165250</td>
      <td>0.551250</td>
      <td>0.408250</td>
      <td>0.204500</td>
      <td>10.946750</td>
    </tr>
    <tr>
      <th>max</th>
      <td>320.200000</td>
      <td>1.987000</td>
      <td>1.006000</td>
      <td>0.990000</td>
      <td>0.629000</td>
      <td>60.061000</td>
    </tr>
  </tbody>
</table>


We see increases in mean strokes gained per round from bottom 50% to top 50% to over 300 yard groupings. 



---

Look at correlations between all strokes gained variables


```python
h2_list_corr_df = df[h2_list].corr()
h2_list_corr_df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>SG_PER_ROUND</th>
      <th>SG:OTT</th>
      <th>SG:APR</th>
      <th>SG:ARG</th>
      <th>TOTAL_SG:PUTTING</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AVG_Driving_DISTANCE</th>
      <td>1.000000</td>
      <td>0.459917</td>
      <td>0.630791</td>
      <td>0.205962</td>
      <td>0.002735</td>
      <td>-0.140773</td>
    </tr>
    <tr>
      <th>SG_PER_ROUND</th>
      <td>0.459917</td>
      <td>1.000000</td>
      <td>0.777071</td>
      <td>0.839260</td>
      <td>0.429187</td>
      <td>-0.110525</td>
    </tr>
    <tr>
      <th>SG:OTT</th>
      <td>0.630791</td>
      <td>0.777071</td>
      <td>1.000000</td>
      <td>0.423156</td>
      <td>-0.005909</td>
      <td>-0.196581</td>
    </tr>
    <tr>
      <th>SG:APR</th>
      <td>0.205962</td>
      <td>0.839260</td>
      <td>0.423156</td>
      <td>1.000000</td>
      <td>0.263644</td>
      <td>-0.141153</td>
    </tr>
    <tr>
      <th>SG:ARG</th>
      <td>0.002735</td>
      <td>0.429187</td>
      <td>-0.005909</td>
      <td>0.263644</td>
      <td>1.000000</td>
      <td>0.248778</td>
    </tr>
    <tr>
      <th>TOTAL_SG:PUTTING</th>
      <td>-0.140773</td>
      <td>-0.110525</td>
      <td>-0.196581</td>
      <td>-0.141153</td>
      <td>0.248778</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>


*   We see positive correlations between average driving distance and strokes gained per round with .45 correlation.
*   Also notice a very high correlation with strokes gained off the tee but as shots are being hit closer to the green the correlation to driving distance lowers and is negative with total strokes gained putting.






Heatmap of correlations with the more red being closer to -.20 correlation and the more green close to 1.0 positive correlation


```python
sns.heatmap(h2_list_corr_df, annot = True, cmap = 'RdYlGn')
plt.title('Heatmap of Correlations')
plt.show()
```


​    
![png](images/output_103_0.png)
​    




---
Regression plot of Strokes Gained per round vs Driving Distance



```python
plt.figure(figsize=(12,6))
sns.regplot(x = 'SG_PER_ROUND', y='AVG_Driving_DISTANCE', data = df)
plt.xlabel('Strokes Gained per Round')
plt.ylabel('Average Driving Distance')
plt.title('Strokes Gained per Round vs Driving Distance')
plt.show()
```


​    
![png](images/output_105_0.png)
​    


Visually see the positive correlation between driving distance and strokes gained per round.  We can also see that as average driving distance increases so the min and max strokes gained per round, meaning that the players floors and ceilings also increase.


---



Visualizations of other strokes gained statistics vs driving distance 


```python
fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
fig.suptitle('Other Trends with Driving Distance and Strokes Gained Statistics')

#SG:OTT
sns.regplot(ax=axes[0,0], x = 'SG:OTT', y='AVG_Driving_DISTANCE', data = df)

#SG:APR
sns.regplot(ax=axes[0,1], data = df, x = 'SG:APR', y='AVG_Driving_DISTANCE')

#SG:ARG
sns.regplot(ax=axes[1,0], data = df, x = 'SG:ARG', y='AVG_Driving_DISTANCE')

#TOTAL_SG:PUTTING
sns.regplot(ax=axes[1,1], data = df, x = 'TOTAL_SG:PUTTING', y='AVG_Driving_DISTANCE')


plt.show()
```


​    
![png](images/output_108_0.png)
​    


Here we can visually see what was noticed in the correlation testing. As players with higher average driving distances off the tee hit shots closer to the green, their strokes gained statistics become worse and eventually have a negative trendline with total strokes gained putting.  


---



Run t-tests for statistical significance between top and bottom 50% players


```python
for x in h2_list:
  print(f'The t-test results for {x} column comparing the top and bottom 50% in average driving distance are:')
  print(stats.ttest_ind(top_50_df[x], bottom_50_df[x]))
  print('\n')
```

    The t-test results for AVG_Driving_DISTANCE column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=17.977004473009526, pvalue=2.9831462094179926e-43)
    
    The t-test results for SG_PER_ROUND column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=4.711458528569094, pvalue=4.666726706320066e-06)
        
    The t-test results for SG:OTT column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=6.723739272937005, pvalue=1.9097909964039726e-10)
        
    The t-test results for SG:APR column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=1.9782929293867122, pvalue=0.04930375538317859)
    
    The t-test results for SG:ARG column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=0.18047586471968233, pvalue=0.8569664244349027)
      
    The t-test results for TOTAL_SG:PUTTING column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=-1.3094745834320003, pvalue=0.19191441093996892)


  ​      

*   Statistically significant strokes gained per round between the top 50% and bottom 50% average driving distance groupings with a p-value of 4.66e-06.
*   Also see statistical significance between stroked gained off the tee and approach shot but not in strokes gained around the green and putting. 



Run t-tests for statistical significance between over and under 300 yards average driving distance


```python
for y in h2_list:
  print(f'The t-test results for {y} column comparing the over and under average driving distance of 300 yards are:')
  print(stats.ttest_ind(over_300_df[y], under_300_df[y]))
  print('\n')
```

    The t-test results for AVG_Driving_DISTANCE column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=17.283678491926892, pvalue=4.681984277618193e-41)
    
    The t-test results for SG_PER_ROUND column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=5.136166108405067, pvalue=6.827833962369511e-07)
     
    The t-test results for SG:OTT column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=7.5524519484132435, pvalue=1.6623456136910585e-12)
     
    The t-test results for SG:APR column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=1.8121738656511948, pvalue=0.07151351629223547)
      
    The t-test results for SG:ARG column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=0.622305786166279, pvalue=0.5344753280212086)
     
    The t-test results for TOTAL_SG:PUTTING column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=-1.1229272480016679, pvalue=0.2628633304520743)




*   Statistically significant strokes gained per round between the over 300 yard and under 300 yard average driving distance groupings with a p-value of 6.82e-07.
*   Also see statistical significance between stroked gained off the tee but not in strokes gained appoach, around the green, and putting. 


---



Summary and Conclusion of H2 Testing:




*   Higher strokes gained per round mean in top 50% vs bottom 50% driving distance groupings by .477 strokes, leading to 1.908 saved strokes per 4 round tournament
*   High positive correlation of .45 between average driving distance and strokes gained per round.
* Statistical signifance in strokes gained per round between top 50% and bottom 50% as well as over 300 and under 300 driving distance groupings with p values of 4.66e-06 and 6.82e-07, respectively. 
---
*   **Reject the null hypothesis and accept the alternate hypothesis that players with higher average driving distance will have better strokes gained per round.**

---
Other insights gained related to strokes gained statistics:
*   Closer shots occur to the hole, the lower the correlation of strokes gained to driving distance
*   Even though strokes gained per round were statistically significant between groupings, SG:Around the Green and Putting were not statistically significant.
*   Thus the data shows that even though the longer drivers lose their once held advantage off the tee, the down trend is very close to even and not significant and leads to their high strokes gained per round statistics.



## Hypothesis 3 Analysis

*   **HO = Players with higher average driving distance will not have lower scoring averages**
*   **HA = Players with higher average driving distance will have lower scoring averages**

---



Create list of variables for H3 analysis


```python
h3_list = (['AVG_Driving_DISTANCE', 'AVG_SCORE','PAR_OR_BETTER'])
```



---


Descriptive Statistics for H3 in Bottom 50%, Top 50%, and Over 300 yard driving distance groupings:

```python
bottom_50_df[h3_list].describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>AVG_SCORE</th>
      <th>PAR_OR_BETTER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>98.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>290.298980</td>
      <td>71.131173</td>
      <td>263.102041</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.222143</td>
      <td>0.755630</td>
      <td>58.600603</td>
    </tr>
    <tr>
      <th>min</th>
      <td>278.400000</td>
      <td>69.311000</td>
      <td>136.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>287.950000</td>
      <td>70.602750</td>
      <td>216.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>291.050000</td>
      <td>71.031000</td>
      <td>276.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>293.350000</td>
      <td>71.518250</td>
      <td>303.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>296.200000</td>
      <td>73.624000</td>
      <td>392.000000</td>
    </tr>
  </tbody>
</table>



```python
top_50_df[h3_list].describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>AVG_SCORE</th>
      <th>PAR_OR_BETTER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>303.138384</td>
      <td>70.658606</td>
      <td>249.373737</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.687055</td>
      <td>0.812773</td>
      <td>46.666114</td>
    </tr>
    <tr>
      <th>min</th>
      <td>296.200000</td>
      <td>68.702000</td>
      <td>156.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299.000000</td>
      <td>70.224000</td>
      <td>213.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>301.600000</td>
      <td>70.734000</td>
      <td>245.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>305.900000</td>
      <td>71.048500</td>
      <td>282.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>320.200000</td>
      <td>74.891000</td>
      <td>356.000000</td>
    </tr>
  </tbody>
</table>



```python
over_300_df[h3_list].describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>AVG_SCORE</th>
      <th>PAR_OR_BETTER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>62.000000</td>
      <td>62.00000</td>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>306.083871</td>
      <td>70.52900</td>
      <td>250.048387</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.240172</td>
      <td>0.77230</td>
      <td>49.008005</td>
    </tr>
    <tr>
      <th>min</th>
      <td>300.000000</td>
      <td>68.70200</td>
      <td>156.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>302.425000</td>
      <td>70.11425</td>
      <td>213.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>303.800000</td>
      <td>70.71100</td>
      <td>244.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>309.475000</td>
      <td>70.95825</td>
      <td>288.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>320.200000</td>
      <td>72.25100</td>
      <td>352.000000</td>
    </tr>
  </tbody>
</table>
 


*   Notice a lower mean and median of average score from the bottom 50% to top 50% and over 300 yard driving distance groupings.
*   Also see lower mean and median with par or better from the bottom 50% to top 50% and over 300 yard driving distance groupings.  Possibility indicating that longer drivers have more birdies and eagles but also more bogeys or worse too.

---



Look at correlations between average driving distance, average score and pars or better.


```python
h3_list_corr_df = df[h3_list].corr()
h3_list_corr_df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>AVG_SCORE</th>
      <th>PAR_OR_BETTER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AVG_Driving_DISTANCE</th>
      <td>1.000000</td>
      <td>-0.400943</td>
      <td>-0.152336</td>
    </tr>
    <tr>
      <th>AVG_SCORE</th>
      <td>-0.400943</td>
      <td>1.000000</td>
      <td>-0.077790</td>
    </tr>
    <tr>
      <th>PAR_OR_BETTER</th>
      <td>-0.152336</td>
      <td>-0.077790</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>


Confirm the negative correlation between average driving distance and average scoring, so the higher the driving distance the lower that player scores.




Heatmap of correlations with the more red being closer to -.40 correlation and the more green close to 1.0 positive correlation


```python
sns.heatmap(h3_list_corr_df, annot = True, cmap = 'RdYlGn')
plt.title('Heatmap of Correlations')
plt.show()
```


​    
![png](images/output_130_0.png)
​    




---
Regression plot of Average Score vs Driving Distance



```python
plt.figure(figsize=(12,6))
sns.regplot(x = 'AVG_SCORE', y='AVG_Driving_DISTANCE', data = df)
plt.xlabel('Average Score')
plt.ylabel('Average Driving Distance')
plt.title('Average Score vs Driving Distance')
plt.show()
```


​    
![png](images/output_132_0.png)
​    


We can see the negative relationship between driving distacne and average score. The further the player hits it, the lower their average score trends.



---



Regression plot of Par or Better Holes vs Driving Distance


```python
plt.figure(figsize=(12,6))
sns.regplot(x = 'PAR_OR_BETTER', y='AVG_Driving_DISTANCE', data = df)
plt.xlabel('Total Holes Par or Better')
plt.ylabel('Average Driving Distance')
plt.title('Total Pars or Better vs Driving Distance')
plt.show()
```


​    
![png](images/output_135_0.png)
​    


See that players with increased driving distances actually have less pars or better during the season than shorter drivers. It is clearly negative but has a low correlation of -.15.


---



Run t-tests for statistical significance between top and bottom 50% players


```python
for x in h3_list:
  print(f'The t-test results for {x} column comparing the top and bottom 50% in average driving distance are:')
  print(stats.ttest_ind(top_50_df[x], bottom_50_df[x]))
  print('\n')
```

    The t-test results for AVG_Driving_DISTANCE column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=17.977004473009526, pvalue=2.9831462094179926e-43)
    
    The t-test results for AVG_SCORE column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=-4.225368653932252, pvalue=3.6606668081070745e-05)
    
    The t-test results for PAR_OR_BETTER column comparing the top and bottom 50% in average driving distance are:
    Ttest_indResult(statistic=-1.8198240595544244, pvalue=0.07031918864409493)





Statistically significant for the average score between the top 50% and bottom 50% average driving distance groupings with a p-value of 3.66e-05.

Run t-tests for statistical significance between over and under 300 yards average driving distance


```python
for y in h3_list:
  print(f'The t-test results for {y} column comparing the over and under average driving distance of 300 yards are:')
  print(stats.ttest_ind(over_300_df[y], under_300_df[y]))
  print('\n')
```

    The t-test results for AVG_Driving_DISTANCE column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=17.283678491926892, pvalue=4.681984277618193e-41)
    
    The t-test results for AVG_SCORE column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=-4.431973933828141, pvalue=1.5640004669604865e-05)
    
    The t-test results for PAR_OR_BETTER column comparing the over and under average driving distance of 300 yards are:
    Ttest_indResult(statistic=-1.1746633958676564, pvalue=0.24157668212690542)





Statistically significant for the average score between the over 300 and under 300 yard average driving distance groupings with a p-value of 1.56e-05.

Summary and Conclusion of H3 Testing:

*   Lower mean average score in top 50% group vs bottom 50% group.
*   Even slightly lower mean score in over 300 year group compared to top 50% group.
*   Negative correlation seen between average score and driving distance. Meaning the longer the driving distance, the lower the score.
*   Statistical signifance for average score between top 50% and bottom 50% as well as over 300 and under 300 driving distance groupings with p values of 3.66e-05 and 1.56e-05, respectively. 

---


*   **Again reject the null hypothesis and accept the alternate hypothesis that players with higher average driving distance will have lower scoring averages.**

---
*   Interesting to see a negative correlation with driving distance and total pars or better, especially with a lower average score in longer driving distances
*   This likely means that players with increased driving distance score more birdies and eagles, but also more bogeys or worse

## Other Trends Related to Driving Distance

Create list of variables to analyze


```python
h4_list = (['AVG_Driving_DISTANCE', 'AGE','FAIRWAY_HIT_%','Three_PUTT%','GOING_FOR_GREEN_IN_2%','RTP-GOING_FOR_THE_GREEN','RTP-NOT_GOING_FOR_THE_GRN'])
```



---


Correlation testing of other numeric variables to see relationship to driving distance


```python
h4_list_corr_df = df[h4_list].corr()
h4_list_corr_df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>AGE</th>
      <th>FAIRWAY_HIT_%</th>
      <th>Three_PUTT%</th>
      <th>GOING_FOR_GREEN_IN_2%</th>
      <th>RTP-GOING_FOR_THE_GREEN</th>
      <th>RTP-NOT_GOING_FOR_THE_GRN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AVG_Driving_DISTANCE</th>
      <td>1.000000</td>
      <td>-0.251643</td>
      <td>-0.398160</td>
      <td>0.141663</td>
      <td>0.811614</td>
      <td>-0.544856</td>
      <td>0.298993</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>-0.251643</td>
      <td>1.000000</td>
      <td>0.131303</td>
      <td>-0.137907</td>
      <td>-0.262759</td>
      <td>0.227949</td>
      <td>-0.167852</td>
    </tr>
    <tr>
      <th>FAIRWAY_HIT_%</th>
      <td>-0.398160</td>
      <td>0.131303</td>
      <td>1.000000</td>
      <td>-0.001424</td>
      <td>-0.223826</td>
      <td>0.065038</td>
      <td>-0.392998</td>
    </tr>
    <tr>
      <th>Three_PUTT%</th>
      <td>0.141663</td>
      <td>-0.137907</td>
      <td>-0.001424</td>
      <td>1.000000</td>
      <td>0.150796</td>
      <td>0.011321</td>
      <td>0.247174</td>
    </tr>
    <tr>
      <th>GOING_FOR_GREEN_IN_2%</th>
      <td>0.811614</td>
      <td>-0.262759</td>
      <td>-0.223826</td>
      <td>0.150796</td>
      <td>1.000000</td>
      <td>-0.588978</td>
      <td>0.462078</td>
    </tr>
    <tr>
      <th>RTP-GOING_FOR_THE_GREEN</th>
      <td>-0.544856</td>
      <td>0.227949</td>
      <td>0.065038</td>
      <td>0.011321</td>
      <td>-0.588978</td>
      <td>1.000000</td>
      <td>-0.026066</td>
    </tr>
    <tr>
      <th>RTP-NOT_GOING_FOR_THE_GRN</th>
      <td>0.298993</td>
      <td>-0.167852</td>
      <td>-0.392998</td>
      <td>0.247174</td>
      <td>0.462078</td>
      <td>-0.026066</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>


See high correlations between driving distance and the following variables: fairways hit (- correlation), going for green in 2 (+ correlation), score relative to par when going for green (+ correlation).

Heatmap of correlations with the more red being closer to -.60 correlation and the more green close to 1.0 positive correlation


```python
sns.heatmap(h4_list_corr_df, annot = True, cmap = 'RdYlGn')
plt.show()
```


​    
![png](images/output_152_0.png)
​    




---
Visualizations of remaining numeric data related to driving distance



```python
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
fig.suptitle('Other Trends with Driving Distance')

#Age
sns.regplot(ax=axes[0,0], data = df, x = 'AGE', y='AVG_Driving_DISTANCE')

#FAIRWAY_HIT_%
sns.regplot(ax=axes[0,1], data = df, x = 'FAIRWAY_HIT_%', y='AVG_Driving_DISTANCE')

#Three_PUTT%
sns.regplot(ax=axes[0,2], data = df, x = 'Three_PUTT%', y='AVG_Driving_DISTANCE')

#GOING_FOR_GREEN_IN_2%
sns.regplot(ax=axes[1,0], data = df, x = 'GOING_FOR_GREEN_IN_2%', y='AVG_Driving_DISTANCE')

#RTP-GOING_FOR_THE_GREEN
sns.regplot(ax=axes[1,1], data = df, x = 'RTP-GOING_FOR_THE_GREEN', y='AVG_Driving_DISTANCE')

#'RTP-NOT_GOING_FOR_THE_GRN
sns.regplot(ax=axes[1,2], data = df, x = 'RTP-NOT_GOING_FOR_THE_GRN', y='AVG_Driving_DISTANCE')

plt.show()
```


​    
![png](images/output_154_0.png)
​    




*   As players get older, driving distance declines
*   Increased driving distance, leads to decreased accuracy which likely plays a role in more volatility in scoring in longer hitters as seen earlier in analysis.
*   Increased driving distances has slight increases in 3 putt percentage
*   As expected, increaed driving distances is highly correlated with going the green in 2 on par 5 holes.
*   Relative to par scoring when going for green in 2 is negatively correlated with driving distance. Thus going for a par 5 in 2 should be chosen whenever able to.
*   Interestingly, shorter average driving distance players score better on par 5 holes when not going for the green in 2 compared to longer hitters.  Likley because they are typically better wedge players and better putters.



## Factors Into Increased Driving Distance

Create list of variables in dataset to analyze to determine which variables lead to highest increase in driving distance


```python
driving_dist_factor = ['AVG_Driving_DISTANCE','AVG_CLUB_HEAD_SPEED', 
       'AVG_BALL_SPEED', 'AVG_SMASH_FACTOR', 'AVG_LAUNCH_ANGLE',
       'AVG_SPIN_RATE', 'AVG_HANG_TIME', 'AVG_CARRY_DISTANCE']
```




---


Correlations of variables related to driving distance


```python
driving_dist_factors_df = df[driving_dist_factor].corr()
driving_dist_factors_df.AVG_Driving_DISTANCE = driving_dist_factors_df.AVG_Driving_DISTANCE.abs()
driving_dist_factors_df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG_Driving_DISTANCE</th>
      <th>AVG_CLUB_HEAD_SPEED</th>
      <th>AVG_BALL_SPEED</th>
      <th>AVG_SMASH_FACTOR</th>
      <th>AVG_LAUNCH_ANGLE</th>
      <th>AVG_SPIN_RATE</th>
      <th>AVG_HANG_TIME</th>
      <th>AVG_CARRY_DISTANCE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AVG_Driving_DISTANCE</th>
      <td>1.000000</td>
      <td>0.864714</td>
      <td>0.914823</td>
      <td>0.151948</td>
      <td>-0.004169</td>
      <td>-0.020851</td>
      <td>0.420434</td>
      <td>0.924587</td>
    </tr>
    <tr>
      <th>AVG_CLUB_HEAD_SPEED</th>
      <td>0.864714</td>
      <td>1.000000</td>
      <td>0.960059</td>
      <td>-0.167782</td>
      <td>-0.035389</td>
      <td>0.247314</td>
      <td>0.289736</td>
      <td>0.845098</td>
    </tr>
    <tr>
      <th>AVG_BALL_SPEED</th>
      <td>0.914823</td>
      <td>0.960059</td>
      <td>1.000000</td>
      <td>0.114551</td>
      <td>-0.011768</td>
      <td>0.165379</td>
      <td>0.320358</td>
      <td>0.901337</td>
    </tr>
    <tr>
      <th>AVG_SMASH_FACTOR</th>
      <td>0.151948</td>
      <td>-0.167782</td>
      <td>0.114551</td>
      <td>1.000000</td>
      <td>0.090226</td>
      <td>-0.292332</td>
      <td>0.104247</td>
      <td>0.176145</td>
    </tr>
    <tr>
      <th>AVG_LAUNCH_ANGLE</th>
      <td>0.004169</td>
      <td>-0.035389</td>
      <td>-0.011768</td>
      <td>0.090226</td>
      <td>1.000000</td>
      <td>0.038425</td>
      <td>0.071029</td>
      <td>0.000931</td>
    </tr>
    <tr>
      <th>AVG_SPIN_RATE</th>
      <td>0.020851</td>
      <td>0.247314</td>
      <td>0.165379</td>
      <td>-0.292332</td>
      <td>0.038425</td>
      <td>1.000000</td>
      <td>0.177615</td>
      <td>0.004018</td>
    </tr>
    <tr>
      <th>AVG_HANG_TIME</th>
      <td>0.420434</td>
      <td>0.289736</td>
      <td>0.320358</td>
      <td>0.104247</td>
      <td>0.071029</td>
      <td>0.177615</td>
      <td>1.000000</td>
      <td>0.536967</td>
    </tr>
    <tr>
      <th>AVG_CARRY_DISTANCE</th>
      <td>0.924587</td>
      <td>0.845098</td>
      <td>0.901337</td>
      <td>0.176145</td>
      <td>0.000931</td>
      <td>0.004018</td>
      <td>0.536967</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>




All variable have a postive correlation to driving distance with club head speed, ball speed, hang time and carry distance all having the highest correlations.

Heatmap of correlations with the more red being closer to no correlation and the more green close to 1.0 positive correlation



```python
sns.heatmap(driving_dist_factors_df, annot = True, cmap = 'RdYlGn')

plt.show()
```


​    
![png](images/output_163_0.png)
​    




---


Visualizations of variables related to driving distance


```python
fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
fig.suptitle('Factors that Lead to Driving Distance')

#AVG_CLUB_HEAD_SPEED
sns.regplot(ax=axes[0,0], data = df, x = 'AVG_CLUB_HEAD_SPEED', y='AVG_Driving_DISTANCE')

#AVG_BALL_SPEED
sns.regplot(ax=axes[0,1], data = df, x = 'AVG_BALL_SPEED', y='AVG_Driving_DISTANCE')

#AVG_SMASH_FACTOR
sns.regplot(ax=axes[1,0], data = df, x = 'AVG_SMASH_FACTOR', y='AVG_Driving_DISTANCE')

#AVG_LAUNCH_ANGLE
sns.regplot(ax=axes[1,1], data = df, x = 'AVG_LAUNCH_ANGLE', y='AVG_Driving_DISTANCE')

plt.show()
```


![png](images/output_165_0.png)
​    


*   Easily see the positive correlation with driving distance and club head speed as well as ball speed which will always be similar to each other and have a correlation of .96.
*   Smash factor has slight uptrend with driving distance 
*   Launch angle appears to have no effect on driving distance 




```python
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
fig.suptitle('Factors that Lead to Driving Distance')

#AVG_SPIN_RATE
sns.regplot(ax=axes[0], data = df, x = 'AVG_SPIN_RATE', y='AVG_Driving_DISTANCE')

#AVG_HANG_TIME
sns.regplot(ax=axes[1], data = df, x = 'AVG_HANG_TIME', y='AVG_Driving_DISTANCE')

#AVG_CARRY_DISTANCE
sns.regplot(ax=axes[2], data = df, x = 'AVG_CARRY_DISTANCE', y='AVG_Driving_DISTANCE')


plt.show()
```


​    
![png](images/output_167_0.png)
​    




*   We see a positive correlation with driving distance and hang time as well as carry distance. Both of these will increase as a players clubhead speed increases. Since launch angle appears to have no correlation to driving distance, increased hang time and carry distance is likely related to club head speed.  This is seen in correlation testing as well. 

*   There is no correlation between driving distance and spin rate 

---

## Recommendations from hypotheses and other insights from analysis


Based on data analysis, all alternate hypotheses were accepted below:

1. Players with higher average driving distance will have more success on PGA Tour.
2. Players with higher average driving distance will have better strokes gained per round.
3. Players with higher average driving distance will have lower scoring averages


Thus increased driving distance leads to improved stats across most major statistical categories including:
1.   Wins
2.   Top 10s
3.   Money Won
4.   Strokes Gained per Round
5.   Average Score

Recommendations from analysis:

*   Any golfer, both professional and amateur, who wants to shoot lower scores and have more success on the golf course should work to increase their driving distance. This would include working with a fitness professional to improve power, strength, and flexibility as well as a swing coach to improve swing technique.

*   Validates the work of many fitness professionals who work with golfers virtually, at home, or in the gym to increase their strength, flexibility, and power to increased their distance to better success on the golf course. 

*   Work to increase club head speed as it lead to the highest increase is driving distance as is correlated with other factors that lead to increased distance as well.

*   Once increased driving distance is obtained, expect more missed fairways as well as more volatility in scoring during rounds.

*   Expect driving distance to decrease as golfers age. 

*   If able to go for the green in 2 on a par 5, always go for it as it will lead to a lower score more often than not.

*   Not all golfers will be able to increase driving distance due to their past medical history or prior injuires and should consult with a medical professional attempting.

Future Analysis Recommendations:

*   This analysis was only completed from statistics from one season and would be beneifical to include data from more seasons to verify this information.

*   Would also be benefical to complete a further analysis comparing importance of driving distance vs other aspects of golf including, approach shots, shots around the green, and putting.




