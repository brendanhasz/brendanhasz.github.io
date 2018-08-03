---
layout: post
title: "Nice Ride Bike Share EDA"
date: 2018-08-02
description: "Exploratory data analysis of Nice Ride MN bike share's system data for 2017."
github_url: https://github.com/brendanhasz/nice-ride
kaggle_url: https://www.kaggle.com/brendanhasz/nice-ride-mn-eda
tags: [eda, python, bokeh]
---

Bicycle ride-sharing systems have become increasingly popular in major cities.  They allow people to enjoy biking around the city without investing in buying a bike for themselves, by providing affordable bike rentals.  Here in the twin cities (Minneapolis/St. Paul, MN) we have the bike-sharing nonprofit [Nice Ride MN](https://www.niceridemn.org/).  Customers can rent bikes at stations, each of which has docks for several bikes, and are scattered throughout the cities.  Customers can then bike around, and return their bike at any other station (providing there's an empty dock for it).  

Nice Ride MN provides public access to their [historical data](https://www.niceridemn.org/data/), and that data is published under the [Nice Ride Minnesota Data License Agreement](https://www.niceridemn.org/data_license/).  In this EDA, we'll explore Nice Ride's data from the 2017 year by looking at things like bike demand across different stations, the flow of bikes from and to each station, seasonality and weather effects on ride patterns, and differences in ride patterns between members and non-members.  We'll be using Python with [pandas](https://pandas.pydata.org/) and [numpy](http://www.numpy.org/) for data manipulation; [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/index.html), and [bokeh](https://bokeh.pydata.org/en/latest/) for visualization; and [statsmodels](https://www.statsmodels.org/stable/index.html) for a small bit of statistical modeling.

## Outline
- [Data Loading and Cleaning](#loading-and-cleaning)
- [Station Locations](#station-locations)
- [Number of Bike Docks per Station](#number-of-docks)
- [Station Demand](#station-demand)
    - [Difference in demand](#difference-in-demand)
    - [Difference in demand over time](#demand-over-time)
    - [Cumulative difference in demand](#cumulative-difference)
    - [Optimal initial number of bikes](#optimal-initial-number)
    - [Demand Range](#demand-range)
- [Flow](#flow)
- [Ride Season](#ride-season)
- [Ride durations](#ride-durations)
- [Weather](#weather)
- [Rental dependence on season](#dependence-on-season)
- [Memberships](#memberships)
- [Conclusion](#conclusion)

Let's load the packages we'll use, and then we can get to the data.


```python
# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from numpy import log

# Matplotlib settings
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
sns.set()

# Bokeh settings
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, Circle, HoverTool, ColorBar, LinearColorMapper, LogColorMapper, CustomJS, Slider
from bokeh.palettes import Viridis256, brewer
from bokeh.tile_providers import CARTODBPOSITRON_RETINA
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column
fig_height = 500
fig_width = 800
output_notebook()
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="66ae79a3-e615-44a1-835a-dc3d0216646e">Loading BokehJS ...</span>
    </div>




<a id='loading-and-cleaning'></a>
## Data Loading and Cleaning

First, let's load the data and take a look at the data to see if it needs any cleaning.


```python
# Load data
stations = pd.read_csv('../input/Nice_Ride_2017_Station_Locations.csv')
trips = pd.read_csv('../input/Nice_ride_trip_history_2017_season.csv')
```

    /opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2698: DtypeWarning: Columns (2,5) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
# Show some of the stations data
stations.sample(n=5)
```




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
      <th>Number</th>
      <th>Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Total docks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>30040</td>
      <td>11th Street &amp; Marquette</td>
      <td>44.972650</td>
      <td>-93.274050</td>
      <td>19</td>
    </tr>
    <tr>
      <th>195</th>
      <td>30198</td>
      <td>Bohemian Flats</td>
      <td>44.976223</td>
      <td>-93.241600</td>
      <td>15</td>
    </tr>
    <tr>
      <th>168</th>
      <td>30171</td>
      <td>SE 3rd Ave &amp; University Ave SE</td>
      <td>44.985465</td>
      <td>-93.252102</td>
      <td>15</td>
    </tr>
    <tr>
      <th>122</th>
      <td>30125</td>
      <td>Cedar Street &amp; Columbus Ave</td>
      <td>44.952351</td>
      <td>-93.099549</td>
      <td>15</td>
    </tr>
    <tr>
      <th>71</th>
      <td>30073</td>
      <td>West 34th Street  &amp; Hennepin Ave S</td>
      <td>44.941388</td>
      <td>-93.298634</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print info about each column
for col in stations:
    print('\n',col,'\nNulls:',stations[col].isnull().sum(),'\n',stations[col].describe())
```

    
     Number 
    Nulls: 0 
     count       202
    unique      202
    top       30151
    freq          1
    Name: Number, dtype: object
    
     Name 
    Nulls: 0 
     count                202
    unique               202
    top       Minnehaha Park
    freq                   1
    Name: Name, dtype: object
    
     Latitude 
    Nulls: 0 
     count    202.000000
    mean      44.965178
    std        0.023582
    min       44.890527
    25%       44.948514
    50%       44.969745
    75%       44.980757
    max       45.042435
    Name: Latitude, dtype: float64
    
     Longitude 
    Nulls: 0 
     count    202.000000
    mean     -93.229178
    std        0.064316
    min      -93.322066
    25%      -93.274872
    50%      -93.251619
    75%      -93.200025
    max      -93.083433
    Name: Longitude, dtype: float64
    
     Total docks 
    Nulls: 0 
     count    202.000000
    mean      18.059406
    std        4.682606
    min        7.000000
    25%       15.000000
    50%       15.000000
    75%       19.000000
    max       41.000000
    Name: Total docks, dtype: float64
    

The stations dataset looks pretty clean - no missing values, and the latitude, longitude, and number of docks loaded as expected.  The only anomaly was that the last station ID number was a string ('NRHQ'), while all the others were integers.  But that's fine - pandas will just treat that column as a categorical object, which is what we want anyway.  

What about the trip data?


```python
# Show some of the trip data
trips.sample(n=5)
```




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
      <th>Start date</th>
      <th>Start station</th>
      <th>Start station number</th>
      <th>End date</th>
      <th>End station</th>
      <th>End station number</th>
      <th>Account type</th>
      <th>Total duration (Seconds)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>320410</th>
      <td>6/17/2017 2:35</td>
      <td>26th Street &amp; Lyndale</td>
      <td>30022</td>
      <td>6/17/2017 3:03</td>
      <td>MCAD</td>
      <td>30018</td>
      <td>Casual</td>
      <td>1651</td>
    </tr>
    <tr>
      <th>274048</th>
      <td>7/3/2017 0:11</td>
      <td>Sanford Hall</td>
      <td>30182</td>
      <td>7/3/2017 0:25</td>
      <td>7th Street SE &amp; 10th Ave SE</td>
      <td>30190</td>
      <td>Member</td>
      <td>805</td>
    </tr>
    <tr>
      <th>256030</th>
      <td>7/8/2017 13:31</td>
      <td>11th Street &amp; Marquette</td>
      <td>30040</td>
      <td>7/8/2017 13:37</td>
      <td>7th Street &amp; 4th Ave S</td>
      <td>30051</td>
      <td>Casual</td>
      <td>368</td>
    </tr>
    <tr>
      <th>192147</th>
      <td>7/30/2017 13:01</td>
      <td>Dupont Ave &amp; 22nd Street</td>
      <td>30048</td>
      <td>7/30/2017 13:17</td>
      <td>Nicollet Island</td>
      <td>30170</td>
      <td>Member</td>
      <td>985</td>
    </tr>
    <tr>
      <th>446736</th>
      <td>4/13/2017 16:30</td>
      <td>Hennepin County Government Center</td>
      <td>30029</td>
      <td>4/13/2017 16:37</td>
      <td>Washington Ave S &amp; 10th Ave S</td>
      <td>30046</td>
      <td>Member</td>
      <td>385</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print info about each column
for col in trips:
    print('\n',col,'\nNulls:',trips[col].isnull().sum(),'\n',trips[col].describe())
```

    
     Start date 
    Nulls: 0 
     count             460718
    unique            171626
    top       7/4/2017 22:29
    freq                  20
    Name: Start date, dtype: object
    
     Start station 
    Nulls: 0 
     count                       460718
    unique                         202
    top       Lake Street & Knox Ave S
    freq                         10747
    Name: Start station, dtype: object
    
     Start station number 
    Nulls: 0 
     count     460718
    unique       401
    top        30115
    freq        7350
    Name: Start station number, dtype: object
    
     End date 
    Nulls: 0 
     count             460718
    unique            170180
    top       9/9/2017 14:58
    freq                  25
    Name: End date, dtype: object
    
     End station 
    Nulls: 0 
     count                       460718
    unique                         202
    top       Lake Street & Knox Ave S
    freq                         11658
    Name: End station, dtype: object
    
     End station number 
    Nulls: 0 
     count     460718
    unique       401
    top        30158
    freq        7769
    Name: End station number, dtype: object
    
     Account type 
    Nulls: 0 
     count     460718
    unique         3
    top       Member
    freq      290070
    Name: Account type, dtype: object
    
     Total duration (Seconds) 
    Nulls: 0 
     count    4.607180e+05
    mean     2.276507e+03
    std      4.393244e+04
    min      6.000000e+01
    25%      4.080000e+02
    50%      7.640000e+02
    75%      1.483000e+03
    max      1.135480e+07
    Name: Total duration (Seconds), dtype: float64
    

The trips dataset looks pretty clean too.  The only anomalies I noticed were that there was a weird third Account type, and the dates were loaded as objects (instead of dates).

The possible types of accounts are "Member", "Casual" (which presumably just means non-member), and "Inconnu" (which I assume represents ["unknown"](https://en.wiktionary.org/wiki/inconnu#French) and not [the fish](https://en.wiktionary.org/wiki/inconnu#English)!).  But there were only two entries w/ an account type of "Inconnu", so we'll just set those entries to be "Casual" so that we only have members and non-members.


```python
# Plot types of accounts in the trips dataset
ax = sns.countplot(x='Account type', data=trips)
plt.title('Account types')
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('%d' % (y), (x.mean(), y), 
                ha='center', va='bottom')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_11_0.svg)



```python
# Set "Inconnu" memberships to "Casual"
trips.loc[trips['Account type']=='Inconnu', 'Account type'] = 'Casual'
```

Another small probem with the trips datset was that the trip start and end dates were loaded as generic objects, instead of dates.  Since we'll want to work with them as date/times, we'll convert them to the datetime datatype:


```python
# Convert start and end times to datetime
for col in ['End date', 'Start date']:
    trips[col] = pd.to_datetime(trips[col],
                                format='%m/%d/%Y %H:%M')
```

<a id='station-locations'></a>
## Station Locations

Let's plot the Nice Ride station locations on a map.  We'll use [Bokeh](https://bokeh.pydata.org/en/latest/) to display a map of the station locations.  To plot them on a map, however, we'll first have to transform the station locations from latitude+longitude coordinates to [Mercator (UTM) coordinates](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system), and make a function which generates a plot of points on the map.


```python
def lat_to_mercY(lat):
    """Convert Latitude to Mercator Y"""
    return np.log(np.tan(np.pi/4+np.radians(lat)/2))*6378137.0

def lon_to_mercX(lon):
    """Convert Longitude to Mercator X"""
    return np.radians(lon)*6378137.0

def MapPoints(lat, lon, size=10, color="green", alpha=0.8, 
              padding=0.1, tooltips=None, title=None, 
              width=None, height=None, clims=None,
              palette=Viridis256, symmetric_color=False):
    """Bokeh plot of points overlayed on a map"""
    
    # Convert lat,lon to UTM coordinates
    X = lon_to_mercX(lon)
    Y = lat_to_mercY(lat)
    
    # Set marker sizes
    if type(size) is int or type(size) is float:
        size = size*np.ones(len(lat))
    
    # Data source table for Bokeh
    source = ColumnDataSource(data=dict(
        X = X,
        Y = Y,
        size = size
    ))
    
    # Color limits
    if type(color) is not str:
        if clims is None:
            if symmetric_color:
                cmin = -max(abs(color))
                cmax = max(abs(color))
            else:
                cmin = min(color)
                cmax = max(color)
        else:
            cmin = clims[0]
            cmax = clims[1]
            
    # Set marker colors
    if type(color) is not str: #map colors to a colormap
        source.add(color, 'color') #add to source table
        mapper = LinearColorMapper(palette=palette, 
                                   low=cmin, 
                                   high=cmax)
        color = {'field': 'color', 'transform': mapper}
            
    # Plot the points
    p = figure(tools="pan,wheel_zoom,reset,hover,save", 
               active_scroll="wheel_zoom")
    p.add_tile(CARTODBPOSITRON_RETINA) #set background map
    p.circle('X', 'Y', source=source, size='size', 
             fill_color=color, fill_alpha=alpha, 
             line_color=None) #plot each station
    p.axis.visible = False
    
    # Colorbar
    if type(color) is not str:
        color_bar = ColorBar(color_mapper=mapper, 
                             location=(0, 0))
        p.add_layout(color_bar, 'right')
        
    # Tool tips
    if tooltips is not None:
        for T in tooltips: #add to Bokeh data source
            source.add(T[1].values.tolist(), name=T[0])
        hover = p.select_one(HoverTool) #set hover values
        hover.tooltips=[(T[0], "@"+T[0]) for T in tooltips]
        
    # Title
    if title is not None:
        p.title.text = title
        
    # Figure height
    if height is not None:
        p.plot_height = height
        
    # Figure width
    if width is not None:
        p.plot_width = width
    
    return p, source
```


```python
# On hover, show Station name
tooltips = [("Station", stations['Name'])]

# Plot the stations
p, _ = MapPoints(stations.Latitude, stations.Longitude, 
                 title="Nice Ride Station Locations",
                 tooltips=tooltips,
                 height=fig_height, width=fig_width)

show(p)
```

<iframe src="/assets/img/nice-ride-eda/station-locations-map.html" height="400" style="border:none;width:100%;"></iframe>
<a href="/assets/img/nice-ride-eda/station-locations-map.html" target="_blank">Full screen map</a>


Looks like most of the stations are scattered around Minneapolis, but there's also a cluster in downtown St. Paul, and several stations along University Ave and Grand Ave, which connect Minneapolis and St. Paul.

<a id='number-of-docks'></a>
## Number of Bike Docks per Station

Each station has a certain number of docks which people can rent bikes from or return them to at the end of their rentals.  How many docks do Nice Ride stations usually have?  We can plot a histogram of the number of docks at each station to get an idea of that distribution.


```python
# Plot histogram of # docks at each station 
plt.figure()
plt.hist(stations['Total docks'], 
         bins=np.arange(0.5, 42.5, 1.))
plt.ylabel('Number of Stations')
plt.xlabel('Number of Docks')
plt.title('Number of Docks Distribution')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_20_0.svg)


It looks like 15 is the standard station size, but with additional 4-dock add-ons (since there's also a bunch of stations with 19 docks, then less with 23, etc).  If the docks are most available/cheapest with 15-plus-increments-of-4 docks, then we'll have to take that into consideration later on, when we try to optimize the number of docks at each station.

Which stations have what number of docks, and how are those docks distributed geographically?  We can again plot the station locations, but now we'll also use the color and size of the station marker to indicate how many docks are at that station.


```python
# On hover, show Station name and the number of docks
tooltips = [("Station", stations['Name']), 
            ("Docks", stations['Total docks'])]

# Plot the stations
p, _ = MapPoints(stations.Latitude, stations.Longitude, 
                 tooltips=tooltips, color=stations['Total docks'],
                 size=4*np.sqrt(stations['Total docks']/np.pi),
                 title="Number of Docks at each Station",
                 height=fig_height, width=fig_width)

show(p)
```








  <div class="bk-root" id="88ae5581-e510-4180-aa92-9d34bc42f1f4"></div>





<a id='station-demand'></a>
## Station Demand

Let's also take a look at the demand at each station.  What I mean by that is the number of bikes which users take from each station and leave at each station.  By grouping the trips dataset by the station name and counting the number of trips in each group, we can find how many trips have left from or ended at each station.


```python
# Count incoming and outgoing trips for each station
demand_df = pd.DataFrame({'Outbound trips': trips.groupby('Start station').size(),
                          'Inbound trips': trips.groupby('End station').size()
                      })
demand_df['Name'] = demand_df.index
sdf = stations.merge(demand_df, on='Name')
```

How many trips are started from each station?  We'll take a look at the distribution (as a histogram) and then plot the number of trips started from each station on a map.


```python
# Plot num trips started from each station 
plt.figure()
plt.hist(sdf['Outbound trips'], bins=20)
plt.ylabel('Number of Stations')
plt.xlabel('Number of outbound rentals')
plt.title('Outbound trip distribution')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_26_0.svg)


And the number of trips which *end* at each station?


```python
# Plot num trips started from each station 
plt.figure()
plt.hist(sdf['Inbound trips'], bins=20)
plt.ylabel('Number of Stations')
plt.xlabel('Number of inbound rentals')
plt.title('Inbound trip distribution')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_28_0.svg)


Nice Ride MN has to re-distribute bikes from stations which have extra bikes to stations which don't have enough.  Stations which have more rides ending at that station than starting there will end up with extra bikes, and Nice Ride will have to re-distribute those extra bikes to the stations which are more empty!  What does this distribution look like? That is, which stations have more rides ending at that station than starting there, or vice versa? 


```python
# Compute the DIFFERENCE between #incoming and #outgoing trips
sdf['demand_diff'] = sdf['Inbound trips']-sdf['Outbound trips']

# Sanity / valid data check
print('Number of outbound trips: %d' % (sdf['Outbound trips'].sum()))
print('Number of inbound trips: %d' % (sdf['Inbound trips'].sum()))

# Plot histogram of difference in demand
plt.figure()
plt.hist(sdf['demand_diff'], bins=20)
plt.ylabel('Number of Stations')
plt.xlabel('Num Inbound - Num Outbound trips')
plt.title('Distribution of Demand Difference')
plt.show()
```

    Number of outbound trips: 460718
    Number of inbound trips: 460718
    


![svg](/assets/img/nice-ride-eda/output_30_1.svg)


Luckily most stations have about as many rides ending there as starting from that station.  However, there are definitely a few stations which are imbalanced!  That is, there are some stations which have more *inbound* rides than *outbound* rides, or vice-versa.  But which ones are those?

We can plot these distributions on a map to see which are the imbalanced stations.  We'll use Bokeh to plot the number of outbound trips, inbound trips, and the difference in demand in three separate tabs.  The color and area of the circles represent the vaue (number of outbound trips, inbound trips, or the difference) in the corresponding tab.  For the "Difference", the size of the circle represents the *absolute* difference (so we can see which stations are the most imbalanced, and the color tells us in which direction they are imbalanced).  Click each tab at the top of the plot to see the number of outbound trips, inbound trips, or the difference between the two.


```python
# On hover, show station info
tooltips = [("Station", sdf['Name']), #show on hover
            ("Docks", sdf['Total docks']), 
            ("Num_Outbound", sdf['Outbound trips']),
            ("Num_Inbound", sdf['Inbound trips']),
            ("Inbound_minus_Outbound", sdf['demand_diff'])]

# Outbound trips 
p1, _ = MapPoints(sdf.Latitude, sdf.Longitude, 
                  tooltips=tooltips, color=sdf['Outbound trips'],
                  size=0.3*np.sqrt(sdf['Outbound trips']/np.pi),
                  title="Number of Outbound Trips",
                  height=fig_height, width=fig_width)
tab1 = Panel(child=p1, title="Outbound")

# Inbound trips
p2, _ = MapPoints(sdf.Latitude, sdf.Longitude, 
                  tooltips=tooltips, color=sdf['Inbound trips'],
                  size=0.3*np.sqrt(sdf['Inbound trips']/np.pi),
                  title="Number of Inbound Trips",
                  height=fig_height, width=fig_width)
tab2 = Panel(child=p2, title="Inbound")

# Difference in demand (Inbound - Outbound) trips
p3, _ = MapPoints(sdf.Latitude, sdf.Longitude, 
                  tooltips=tooltips, color=sdf['demand_diff'],
                  size=1*np.sqrt(np.abs(sdf['demand_diff'])/np.pi),
                  title="Inbound - Outbound Trips",
                  palette=brewer['Spectral'][11], #divergent palette
                  symmetric_color=True, #centered @ 0
                  height=fig_height, width=fig_width)
tab3 = Panel(child=p3, title="Difference")

show(Tabs(tabs=[tab1, tab2, tab3]))
```








  <div class="bk-root" id="fb8b71d2-e909-4249-b359-0a84b92965b7"></div>





In the "Difference" tab, we can immediately identify some problem stations.  There are some stations at which far more people are ending their trips than starting them (e.g. the station at Lake St & Knox Ave, at the northeast corner of Bde Maka Ska, or the Minnehaha Park station).  There are also stations at which far more people are *starting* their trips than ending them (e.g. the stations at Coffman Union and Wiley Hall on the Univeristy of Minnesota campus).  But the majority of stations have about as many inbound as outbound rides (e.g. the two stations by Como Lake, which you can barely see on the differences map!).  

Plotting the difference in demand by location also reveals an interesting rental flow pattern: it appears that, on average, more rides start from downtown Minneapolis or the U of M campus and go away from downtown.  Notice how there's a lot of large blue circles (stations with more outbound rides) clustered around downtown Minneapolis, but most of the large red circles (stations with more inbound rides) lie away from downtown, and tend to be "destinations" and parks (such as Minnehaha Park, Bde Maka Ska, Logan Park, and North Mississippi Regional Park)

<a id='difference-in-demand'></a>
### Difference in demand

Is this difference in demand going to be a problem for Nice Ride?  Ideally, Nice Ride will want to have more docks at stations where there is large difference between the number of incoming and outgoing rides.  This is because if more rides are starting at a given station than ending there, the number of bikes at that station will decrease as the day goes on.  So, there needs to be enough docks at that station to hold enough bikes so the station isn't empty by the end of the day!  On the other hand, if more rides are *ending* at a station than are beginning there, all the docks at that station will fill up and people won't be able to end their rides there!  So those stations must have enough docks to absorb the amount of incoming traffic over the course of a day.

Stations which have a good balance of the number of rides coming in to the number of rides going out don't need quite as many docks - because about as many bikes are being taken from that station as are being left there.  Of course, there's also the issue of time - some stations may see different demand depending on the time of day, week, or season - but we'll get to that later. With a good match of the number of docks at each station to the difference between incoming and outgoing trips, Nice Ride won't have to spend as much time during prime riding hours re-distributing bikes from low-demand stations (with extra unused bikes) to high-demand stations (with not enough bikes!).  How well does this distribution of demand differences match up with the distribution of the number of docks at each station? 


```python
# Compute the ABSOLUTE difference between #incoming and #outgoing trips
sdf['abs_diff'] = sdf['demand_diff'].abs()

# Distributions of docks and absolute demand difference across stations
sdf['Docks'] = sdf['Total docks']/sdf['Total docks'].sum()
sdf['DemandDiff'] = sdf['abs_diff']/sdf['abs_diff'].sum()

# Demand direction (more outgoing vs incoming)
sdf['demand_dir'] = sdf['Name']
sdf.loc[sdf['demand_diff']<0, 'demand_dir'] = 'More Outgoing'
sdf.loc[sdf['demand_diff']>0, 'demand_dir'] = 'More Incoming'
sdf.loc[sdf['demand_diff']==0, 'demand_dir'] = 'Balanced'

# Tidy the data for seaborn
tidied = (
    sdf[['Name', 'Docks', 'DemandDiff']]
       .set_index('Name')
       .stack()
       .reset_index()
       .rename(columns={'level_1': 'Distribution', 0: 'Proportion'})
)

# Show the distributions
plt.figure(figsize=(4.5, 35))
station_list = sdf.sort_values('DemandDiff', ascending=False)['Name'].tolist()
sns.barplot(y='Name', x='Proportion', hue='Distribution', 
            data=tidied, order=station_list)
plt.title('Proportion Docks vs Demand Difference')
locs, labels = plt.yticks()
plt.yticks(locs, tuple([s[:15] for s in station_list]))
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_34_0.svg)


There's not a really good match between the number of docks at each station and the overall difference in demand.  But don't go telling Nice Ride to re-allocate their docks just yet!  Keep in mind that we were looking at the *overall* difference in demand - but that difference in demand probably changes over time.  For example, some stations may have more outbound trips in the morning and more inbound trips in the evening, or vice-versa.

<a id='demand-over-time'></a>
### Difference in demand over time

Let's look at how the demand at each station changes over time.  First we need to compute the number of inbound and outbound trips, and the difference, at each station by the hour:


```python
# Compute hourly trips at each station
trips['Start hour'] = trips['Start date'].dt.hour
trips['End hour'] = trips['End date'].dt.hour
outbound = ( #num trips in and out by hour for each station
    trips.groupby(['Start station', 'Start hour'])['Start date']
         .count().to_frame().reset_index()
         .rename(columns={"Start date": "Outbound trips", 
                          "Start station": "Station name",
                          "Start hour": "Hour"})
)
inbound = ( #num trips in and out by hour for each station
    trips.groupby(['End station', 'End hour'])['End date']
         .count().to_frame().reset_index()
         .rename(columns={"End date": "Inbound trips", 
                          "End station": "Station name",
                          "End hour": "Hour"})
)
trips_h = ( #num trips in and out by hour for each station
    pd.merge(outbound, inbound, how='outer', 
             on=['Station name', 'Hour'])
      .fillna(value=0)
      .sort_values(['Station name', 'Hour'])
)
trips_h['Difference'] = trips_h['Inbound trips'] - trips_h['Outbound trips']

# Pivot to get Nstations-by-Nhours arrays for in, out, and diff
trips_hp = trips_h.pivot(index='Station name', columns='Hour').fillna(value=0)

# Normalize by number of days in the season
ndays = (max(trips['Start date'])-min(trips['Start date'])).days
trips_hp = trips_hp/ndays
```

Now we can see how the demand at each station changes over time.  For example, let's take a look at the busy downtown station at the Hennepin County Government Center.


```python
station_name = 'Hennepin County Government Center'

# Plot demand at Hennepin County Government Center
plt.figure()
plt.subplot(2, 1, 1)
inds = trips_hp['Outbound trips'].index==station_name
plt.plot(trips_hp['Outbound trips'].loc[inds,:].values[0])
plt.plot(trips_hp['Inbound trips'].loc[inds,:].values[0])
plt.xticks(np.arange(0, 25, 6))
plt.xlabel('Hour')
plt.ylabel('Average\nNumber of trips')
plt.legend(['Outbound', 'Inbound'])
plt.title('Trips at '+station_name)
plt.show()

# Plot difference in demand at Hennepin County Government Center
plt.subplot(2, 1, 2)
inds = trips_hp['Difference'].index==station_name
plt.plot(trips_hp['Difference'].loc[inds,:].values[0], 'y')
plt.fill_between(np.arange(24), 0, 
                 trips_hp['Difference'].loc[inds,:].values[0],
                 color='yellow', alpha=0.2)
plt.xticks(np.arange(0, 25, 6))
plt.xlabel('Hour')
plt.ylabel('Average Difference\n(Inbound - Outbound)')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_38_0.svg)



![svg](/assets/img/nice-ride-eda/output_38_1.svg)


This station sees a large volume of both inbound and outbound rentals over the course of a day.  But the balance between inbound and outbound trips isn't constant - it changes with time!  At around 8am, there are more people ending their rentals at the station, probably people who are commuting to work.  But at the end of the day, around 5p, people are usually *starting* their rentals from that station, probably for their commute home.

What about a station in a more residential area?  Let's take a look at how the demand changes for a station in the Marcy Holmes neighborhood, just across the river from downtown, and down the street from the University of Minnesota.


```python
station_name = '6th Ave SE & University Ave'

# Plot demand at 6th Ave SE & University Ave
plt.figure()
plt.subplot(2, 1, 1)
inds = trips_hp['Outbound trips'].index==station_name
plt.plot(trips_hp['Outbound trips'].loc[inds,:].values[0])
plt.plot(trips_hp['Inbound trips'].loc[inds,:].values[0])
plt.xticks(np.arange(0, 25, 6))
plt.xlabel('Hour')
plt.ylabel('Average\nNumber of trips')
plt.legend(['Outbound', 'Inbound'])
plt.title('Trips at '+station_name)
plt.show()

# Plot difference in demand at 6th Ave SE & University Ave
plt.subplot(2, 1, 2)
inds = trips_hp['Difference'].index==station_name
plt.plot(trips_hp['Difference'].loc[inds,:].values[0], 'y')
plt.fill_between(np.arange(24), 0, 
                 trips_hp['Difference'].loc[inds,:].values[0],
                 color='yellow', alpha=0.2)
plt.xticks(np.arange(0, 25, 6))
plt.xlabel('Hour')
plt.ylabel('Average Difference\n(Inbound - Outbound)')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_40_0.svg)



![svg](/assets/img/nice-ride-eda/output_40_1.svg)


Here the pattern is the opposite - more trips depart from the station in the morning and end there in the afternoon.  There's also a lot more inbound trips late into the evening.  That's probably students from the Univeristy of Minnesota coming home.

But instead of looking through all 202 stations individually, let's just plot on a map how the difference in demand changes across the day at all stations.  In the map below, drag the slider to see how station demands change over the course of the day.

(Note: you might want to pan the map a bit before messing with the slider.  It gets a bit glitchy otherwise.)


```python
def HourSlider(loc, val, title='', height=500, width=800,
               palette=Viridis256, symmetric_color=False):
    """Plot map of points w/ values which change hourly
    
    Parameters
    ----------
    loc : pd.DataFrame
        Locations and names of stations.
        Must have Name, Latitude, and Longitude columns
        Size: (Nstations, 3)
    val : pd.DataFrame
        Values which change hourly
        Size: (Nstations, Nhours)
        
    Returns
    -------
    bokeh.models.layouts.Column
    Bokeh layout with the plot and slider
    """
    
    # On hover, show station name
    tooltips = [("Station", loc.Name)]

    # Plot trips
    V = np.transpose(val.values)
    p, src = MapPoints(loc.Latitude, loc.Longitude, 
                       tooltips=tooltips, color=V[12,:],
                       size=10*np.sqrt(np.abs(V[12,:])),
                       symmetric_color=symmetric_color,
                       palette=palette, title=title, 
                       height=height, width=width)
    
    # Make slider to control hour
    cb = CustomJS(args=dict(source=src, colors=V), code="""
        var h = cb_obj.value;
        var c = source.data['color'];
        var s = source.data['size'];
        for (var i = 0; i < c.length; i++) {
            c[i] = colors[h][i];
            s[i] = 10*Math.sqrt(Math.abs(colors[h][i]));
        }
        source.change.emit();
    """)
    slider = Slider(start=0, end=23, 
                    value=12, step=1, title="Hour")
    slider.js_on_change('value', cb)
    
    # Return figure w/ slider
    return column(p, slider)

# Outbound
p1 = HourSlider(stations.sort_values('Name'), 
                trips_hp['Outbound trips'], 
                title='Number of Outbound Trips, by Hour')
tab1 = Panel(child=p1, title="Outbound")

# Inbound
p2 = HourSlider(stations.sort_values('Name'), 
                trips_hp['Inbound trips'], 
                title='Number of Inbound Trips, by Hour')
tab2 = Panel(child=p2, title="Inbound")

# Difference
p3 = HourSlider(stations.sort_values('Name'), 
                trips_hp['Difference'], 
                title='Difference (Inbound - Outbound), by Hour',
                palette=brewer['Spectral'][11],
                symmetric_color=True)
tab3 = Panel(child=p3, title="Difference")

# Show all tabs
show(Tabs(tabs=[tab3, tab1, tab2]))
```








  <div class="bk-root" id="3b77cd8f-b73e-4dd2-89e1-c6d4e677d0e2"></div>





There's a pretty obvious commuter trend: stations downtown and at the Univeristy of Minnesota have more incoming rides during the morning rush hour (~7-8am), while residential areas surrounding downtown have more outbound rides.  During the evening commute however (~4-5pm), stations downtown and at the U of M are depleted and there is an influx of rides to stations in surrounding residential areas.

This pattern can actually work in Nice Ride's favor.  Suppose a station has a bunch of bikes leaving in the morning.  As long as there were lots of bikes there in the morning, if they all come back in the evening, then that's great for Nice Ride - they don't have to re-distribute the bikes to that station! The only time when bike distribution problems arise is when (1) there is an imbalance in the ride distribution on average, or (2) there aren't enough docks at a station to buffer the daily fluxuations in demand at that station. 

Earlier we looked at which stations have imbalances in the ride distribution on average (the horizontal bar plot with difference in demand and # docks for each station).  That analysis tells us about which stations we can expect to see become depleted over the course of the day, and *not* be replenished, or vice versa.  This problem can be alleviated at a given station by increasing the number of docks at a station, but that won't *fix* the problem.  Even with a large number of docks at a station, the imbalance will add up over time.  No matter what, Nice Ride will have to re-distribute bikes to/from these stations.  That's why we shouldn't worry as much about trying to match the number of docks at a station to its demand difference on average.  However, the other problem *can* be fixed by optimizing the number of docks at each station.  

<a id='cumulative-difference'></a>
### Cumulative difference in demand

That second problem is the issue of the demand changing over time.  Even when a station has an even balance in inbound to outbund trips on average, we want there to be enough docks at a station to buffer imbalances that happen over the course of a day.  For example, suppose a station has, on average, 10 outbound rentals in the morning, and then 10 inbound rentals in the afternoon.  There's a perfect balance on average, so Nice Ride doesn't have to worry too much about re-distributing bikes to or from that station.  

But only if the station has enough docks!  What if the station only has 5 docks?  Then the station will run out of available bikes halfway through the morning, and Nice Ride will have to bring 5 bikes to the station from some other station so that more people can rent from that station.  If they don't, Nice Ride will miss out on revenue!  And in the afternoon, the station will fill up, and Nice Ride will have to take 5 bikes *away* from the station to a different station, so that there are empty docks where people can end their rides.  If they don't, customers will be pissed that they can't end their ride where they wanted to.  So if the station doesn't have enough docks, Nice Ride will have to re-distribute bikes to and from that station *every day!*

We need to optimize the number of docks at each station in order to minimize the amount of bike re-distribution that's required.  To do that, we'll take a look at the cumulative difference in demand at each station.  The cumulative difference in demand is, at a given time of day, how many more (or less) bikes are at station than were there at the beginning of the day.  Using this analysis, we can answer some important questions about bike re-distribution of bikes at each station:
- Which stations have a large imbalance?
- In which direction is that imbalance? (i.e. does Nice Ride need to bring bikes to that station, or take them away?)
- When is that imbalance worst?
- What is the optimal number of bikes to have at each station at the beginning of the day?
- What is the optimal number of docks to have at each station?

Let's compute the cumulative difference in demand at each station.  We'll also normalize by the number of days in the Nice Ride season, so that our values reflect how many bikes are at each station across time on an average day (relative to the number of bikes that were there at the beginning of the day).


```python
# Compute the cumulative difference in demand
cdiff = trips_hp['Difference'].apply(np.cumsum, axis=1)
```

What does the cumulative difference in demand look like for the station at the station at 6th Ave SE & University Ave, which we were looking at earlier?


```python
station_name = '6th Ave SE & University Ave'

# Plot difference in demand at 6th Ave SE & University Ave
plt.figure()
plt.subplot(2, 1, 1)
inds = trips_hp['Difference'].index==station_name
plt.plot(trips_hp['Difference'].loc[inds,:].values[0], 'y')
plt.fill_between(np.arange(24), 0, 
                 trips_hp['Difference'].loc[inds,:].values[0],
                 color='yellow', alpha=0.2)
plt.xticks(np.arange(0, 25, 6))
plt.xlabel('Hour')
plt.ylabel('Average Difference\n(Inbound - Outbound)')
plt.show()

# Plot cumulative difference in demand at Lake Street & Knox Ave S
plt.subplot(2, 1, 2)
plt.plot(cdiff.loc[station_name, :].index,
         cdiff.loc[station_name, :].values, 'b')
plt.fill_between(cdiff.loc[station_name, :].index, 0,
                 cdiff.loc[station_name, :].values,
                 color='blue', alpha=0.2)
plt.xticks(np.arange(0, 25, 6))
plt.xlabel('Hour')
plt.ylabel('Cumulative Difference\n(Num bikes)')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_46_0.svg)



![svg](/assets/img/nice-ride-eda/output_46_1.svg)


Looking at the cumulative demand difference makes it obvious that a lot of bikes are being taken away from this station during the morning, and then brought back in the evening.  So if Nice Ride doesn't re-distribute bikes to this station, there will be a lot fewer bikes here between 9a and 4p than there are at night.

Let's plot the cumulative difference in demand at all stations over the course of a day.  Blue will show when a station usually has less bikes than it had at the beginning of the day, red will show when a station has more bikes than it started with, and white will show  when the station has about as many bikes as it started with.


```python
# Sort stations by overall demand difference
cdiff = cdiff.iloc[np.argsort(-np.sum(cdiff.values, axis=1)), :]

# Plot cumulative difference for each station
sns.set_style("dark")
plt.figure(figsize=(7, 35))
plt.imshow(cdiff, 
           aspect='auto',
           interpolation="nearest")
plt.set_cmap('seismic')
mav = np.amax(np.abs(cdiff.values))
plt.clim(-mav, mav)
cbar = plt.colorbar()
cbar.set_label('Number of bikes relative to beginning of day')
plt.ylabel('Station')
plt.xlabel('Hour')
plt.xticks(np.arange(0, 24, 6))
plt.yticks(np.arange(cdiff.shape[0]),
           tuple([s[:15] for s in cdiff.index.tolist()]))
plt.title('Average Number of Bikes at each station\nrelative to start of day')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_48_0.svg)


From this plot we can see which stations have the largest imbalances of incoming to outbound rides (stations with dark red or blue), the direction of that imbalance (red=more bikes, blue=less bikes), and at what time during the day that imbalance is the worst.  

Using this matrix, we could work out the best times to move bikes between stations, how many to move, and between which stations.  Unfortunately the times at which Nice Ride actually re-distributed bikes isn't included in the historical datset, although that information could be collected from their [General Bikeshare Feed](https://api-core.niceridemn.org/gbfs/gbfs.json).

<a id='optimal-initial-number'></a>
### Optimal initial number of bikes

We can also use this cumulative difference information to figure out the optimal number of bikes to have at each station at the beginning of the day.  We want to minimize the amount of time that stations are empty when customers are wanting to rent, and the amount of time stations are full when customers are wanting to return their rentals.  To do that we'll define a "loss" function for each station.  This loss function takes a hypothetical initial number of bikes, and returns a value which is higher when a station is close to being full and there are more inbound trips than outbound trips, and also higher when a station is close to being empty and there are more outbound trips than inbound trips:

$$ 
L(N_0) = \int_t \alpha \max(0, D_t) \exp ( N_0 + C_t - N_d ) - (1-\alpha) \min(0, D_t) \exp ( - N_0 - C_t )  ~ dt
$$

where 

- \\( L(N_0) \\) is the loss of starting with N_0 docks at the beginning of the day
- \\( D_t \\) is the demand difference at time \\( t \\) (inbound rides minus outbound rides)
- \\( C_t \\) is the cumulative demand difference (\\( C_t = \int_0^t D_x ~ dx \\))
- \\( N_d \\) is the number of docks at the station
- \\( \alpha \\) is a parameter which controls how bad it is for a station to be full vs being empty

Let's look at each part of that overly-complicated equation.  The integral in the loss equation sums over all the hours of a day (but you could use a finer time bin if you wanted to).  The first half of what's inside the integral (\\( \alpha \max(0, D_t) \exp ( N_0 + C(t) - N_d ) \\)) is a value which will be large when the station is close to being full and there are more inbound rides than outbound rides.  \\( \max(0, D_t) \\) is just the difference in demand, but only when it's positive (and 0 when it's negative).  \\( N_0 + C(t) - N_d \\) is the (negative of the) expected number of *empty docks* left at the station.  That way, when the expected number of empty docks at the station is very low, \\( \exp ( N_0 + C_t - N_d ) \\) will be very high, and vice-versa.  So, when the expected number of empty docks at the station is low *and* the number of inbound trips is high, the value of \\( \max(0, D_t) \exp ( N_0 + C_t - N_d ) \\) will be large.

But we also want to penalize having an empty station when people want to rent bikes!  The second half of what's in the integral (\\( - (1-\alpha) \min(0, D_t) \exp ( - N_0 - C_t ) \\)) is a value which will be large when the station is close to being empty and there are more outbound rides than inbound rides.  \\( \min(0, D_t) \\) is the difference in demand, but only when it's negative (and 0 when it's positive).  \\( (- N_0 - C_t) \\) is the (negative of the) expected number of bikes left at the station.  That way, when the expected number of bikes available at a station is low, \\( \exp ( - N_0 - C_t ) \\) will be very high, and vice-versa.  So when the expected number of available bikes at the station is low *and* the number of outbound trips his high, the value of \\( - \min(0, D_t) \exp ( - N_0 - C_t ) \\) will be large.

The \\( \alpha \\) parameter controls how bad it is to have stations fill up with bikes compared to running out of bikes, and should take values between 0 and 1.  If \\( \alpha<0.5 \\), that means we consider it "more bad" when the station is empty but there are people wanting to rent from that station than when it is full and people are wanting to end their rides there.  I'm going to set \\( \alpha=0.3 \\), because to me it seems slightly worse to have a station be empty when someone wants to rent a bike than to have a station be full where someone wants to return a bike.  Either way the customer is pissed, but if the station they want to rent from is empty, they probably won't end up renting a bike, and Nice Ride will miss out on revenue.  If the station they want to return their bike to is full, they'll just have to find another station and return it there.

(Keep in mind that optimizing this jerry-built loss function won't technically give us the "optimal" number of initial bikes.  To get that we would want to build a predictive model and run a full cost-benefit analysis.  More on that later.)

Let's compute the optimal number of bikes for each station to have at the beginning of the day using that loss function.


```python
# Create dataframe to store optimal initial # bikes
OptBikes = pd.DataFrame(index=trips_hp.index.tolist(), 
                        columns=['Opt Bikes'])

# Set alpha
alpha = 0.3

# Compute optimal initial number of bikes
for s in OptBikes.index: #for each station,
    Nd = stations.loc[stations.Name==s, 'Total docks'].values[0] #num docks @ this station
    bL = np.inf #best loss so far 
    bB = Nd/2   #best #bikes for that loss
    for N0 in range(Nd+1): #for each possible initial # of bikes,
        L = 0 #loss w/ this # initial bikes
        for h in np.arange(24): #for each hour,
            Dt = trips_hp['Difference'].loc[s, h] #diff @ now
            Ct = cdiff.loc[s, h] #cum diff @ now
            L = ( L + #integrate the loss function!
                  alpha*max(0,Dt)*np.exp(N0+Ct-Nd) -
                  (1.0-alpha)*min(0,Dt)*np.exp(-N0-Ct))
        if L<bL: #if best loss so far,
            bL = L #save this loss 
            bB = N0 #and #bikes
    OptBikes.loc[s,'Opt Bikes'] = bB
```

Now we can see, for each station, the optimal number of bikes for Nice Ride to have at that station at the beginning of a day, such that the least amount of time passes when the station is empty but people want to rent from that station, or the station is full but people want to end their rentals there.


```python
# Plot the optimal number of initial bikes
sns.set()
plt.figure(figsize=(4.5, 35))
station_list = (OptBikes
                    .sort_values('Opt Bikes')
                    .index
                    .tolist())
sns.barplot(y=OptBikes.index.tolist(), 
            x=OptBikes['Opt Bikes'], 
            order=station_list)
plt.title('Optimal Initial Number of Bikes\n' +
          'at each station at beginning of day')
locs, labels = plt.yticks()
plt.yticks(locs, tuple([s[:15] for s in station_list]))
plt.xlabel('Number of Bikes')
plt.ylabel('Station')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_52_0.svg)


These values are a good first estimation at a target number of bikes for Nice Ride to have at each station at the end of the day.  However, keep in mind that Nice Ride's employees are re-distributing bikes throughout the day (and not all at once at midnight!).  An even more useful analysis would be to determine when Nice Ride employees should re-distribute bikes, and from/to which stations.  We won't do that here, but I hope to do that analysis in a future post.

<a id='demand-range'></a>
### Demand Range

Finally, we can also compute the optimal number of *docks* for each station to have, given the cumulative difference in demand.  Of course, ideally each station would have *infinite* docks, but Nice Ride doesn't have infinite money.  The best we can do is determine how to optimally *distribute* the docks that Nice Ride can afford across their stations.

Each station should have enough docks to buffer the daily fluxuations in bikes.  Specifically, to avoid re-distribution of bikes, a station needs to have (on average) as many docks as the difference between the maximum and minimum of its cumulative demand difference (or the demand range). 

Let's look at this value for an example station.  Below is the cumulative difference in demand for the station at 6th Ave SE & University Ave (cumulative number of inbound minus outbound rides per hour).  The demand range for this station is the difference between the max and minimum of this cumulative difference - the difference between the two dashed lines in the plot below:


```python
# Plot cumulative difference at 6th Ave SE & University Ave
station_name = '6th Ave SE & University Ave'
plt.figure()
RH = cdiff.loc[station_name, :].index
RC = cdiff.loc[station_name, :].values
plt.plot(RH, RC, 'b')
plt.fill_between(RH, 0, RC, color='blue', alpha=0.2)
plt.plot([0, 23], [RC.max(), RC.max()], 'k--')
plt.plot([0, 23], [RC.min(), RC.min()], 'k--')
plt.annotate(s='', xy=(13,RC.min()), xytext=(13,RC.max()), 
             arrowprops=dict(arrowstyle='<->'))
plt.xticks(np.arange(0, 25, 6))
plt.xlabel('Hour')
plt.ylabel('Cumulative Difference\n(Num bikes)')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_54_0.svg)


The larger that demand range value is, the more docks the station should have in order to buffer that fluxuation in the number of bikes at the station.

Let's compute the demand range for each station, and see how the distribution of demand ranges compare to the distribution of docks across all stations


```python
# Create dataframe to store demand ranges
range_vs_docks = pd.DataFrame(index=stations['Name'].tolist(), 
                              columns=['Range', 'Docks'])

# Compute demand range for each station
for s in range_vs_docks.index:
    RC = cdiff.loc[s, :].values
    range_vs_docks.loc[s,'Range'] = RC.max() - RC.min()
    D = stations.loc[stations.Name==s,'Total docks'].values[0]
    range_vs_docks.loc[s,'Docks'] = D
    
# Normalize distributions
range_vs_docks = range_vs_docks.apply(lambda x: x/np.sum(x))
    
# Tidy the data for seaborn
tidified = (
    range_vs_docks
       .stack()
       .reset_index()
       .rename(columns={'level_0': 'Name',
                        'level_1': 'Distribution', 
                        0: 'Proportion'})
)

# Show the distributions
plt.figure(figsize=(4.5, 35))
station_list = range_vs_docks.sort_values('Range', ascending=False).index.tolist()
sns.barplot(y='Name', x='Proportion', hue='Distribution', 
            data=tidified, order=station_list)
plt.title('Proportion Docks vs Demand Range')
locs, labels = plt.yticks()
plt.yticks(locs, tuple([s[:20] for s in station_list]))
plt.tick_params(axis='y', labelsize=5)
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_56_0.svg)


There's not a great match between the number of docks and the demand range.  Ideally, those two distributions would look similar.  But it's not an awful match, either - notice how the stations with the lowest demand range usually have the smallest number of docks, and stations with the highest demand range usually have the most docks.

Why don't the two distributions match better?  Hasn't Nice Ride done their homework?  There are a few reasons why the actual distribution of docks might not align to the demand range as we computed it.

First of all, it seems like it would be costly for each station to have a custom number of docks.  As we saw earlier, most stations have 15 docks, with additional increments of 4 docks.  It would seem that the standard station size is 15, with 4-dock add-ons.  It is likely more expensive for Nice Ride to custom-order each station than it is for them to buy mostly stations with the standard number of docks.

Secondly, it is probably less expensive for Nice Ride to pay a few employees to re-distribute bikes over the course of the day than it is for them to buy stations with a bajillion docks.  Ideally, stations with large demand ranges (such as the stations at Hennepin County Gov't Center, Washingon Ave, Coffman Union, etc) would have far more docks, to match their fluxuations in bikes over the course of the day.  However that isn't nessesary - a few employees can re-distribute bikes from and to these stations over the course of the day so that the stations don't fill up or empty.

Also keep in mind that these demand ranges were calculated using the average over all days of the season!  In reality there's probably a difference based on *what* day it is (e.g. a weekday vs a weekend), whether the day is a holiday, the season/time of year, the weather, and other factors.  It may be most cost-effective for Nice Ride to optimize the number of docks during, say, only the part of the season when they have the most customers.   Or the day of the week, etc.

Computing the average demand range is a simple way to preliminarily estimate how many docks Nice Ride should have at each station.  While it was simple enough to do in a few lines of code, there are more complex ways to more accurately predict optimal parameters for Nice Ride's system.  Ideally, one would want to train a machine learning algorithm to predict the volume of inbound and outbound rides at each station.  Instead of using averages, as we used above, this algorithm could take into consideration the season, day of the week, weather, etc, and output demand predictions.  From those demand predictions we could compute the demand difference and the cumulative demand difference.  Given the average revenue per ride, and the estimated cost of having someone end up at a full station, we could perform a cost-benefit analysis in order to optimize the initial number of bikes at each station, the number of docks at each station, the best times to re-distribute bikes, *and* from/to which stations.  Again, we won't do that here, but I hope to build such a model in a future post.

<a id='flow'></a>
## Flow

We looked at where the Nice Ride stations are located, the number of docks at each station, and the demand at each station and how that demand changes over time.  However, how are bikes flowing from each station to the other?  That is, what does the distribution of *trips* look like?  What are the most and least common destinations from each station?

Let's compute how many trips go from each station to each other station.


```python
# Compute number of trips from each station to each other station
flow = (
    trips.groupby(['Start station', 'End station'])['Start date']
    .count().to_frame().reset_index()
    .rename(columns={"Start date": "Trips"})
    .pivot(index='Start station', columns='End station')
    .fillna(value=0)
)
```

What does the distribution of trips look like?  We can visualize this information as a matrix, where each row corresponds to trips *from* a given station, and each column corresponds to trips *to* a given station.  So the value in row \\( i \\), column \\( j \\), is the number of trips from station number \\( i \\) to station number \\( j \\).


```python
# Plot trips to and from each station
sns.set_style("dark")
plt.figure(figsize=(10, 8))
plt.imshow(np.log10(flow.values+0.1),
           aspect='auto',
           interpolation="nearest")
plt.set_cmap('plasma')
cbar = plt.colorbar(ticks=[-1,0,1,2,3])
cbar.set_label('Number of trips')
cbar.ax.set_yticklabels(['0','1','10','100','1000'])
plt.ylabel('Station Number FROM')
plt.xlabel('Station Number TO')
plt.title('Number of trips to and from each station')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_61_0.svg)


There's a clear diagonal line across the plot - this indicates that many trips end at the same station at which they started, because the diagonal corresponds to where the row number is the same as the column number (the start station is the same as the end station).

Is this the norm?  Do most trips actually return to the station they started from?


```python
# Plot num trips returning to same station
sns.set()
plt.figure()
plt.bar([0, 1], 
        [np.trace(flow.values), 
         flow.values.sum()-np.trace(flow.values)],
        tick_label=['Same as start', 'Other'])
plt.xlabel('End station')
plt.ylabel('Number of trips')
plt.title('Number of trips which end\n'+
          'at same station they started from')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_63_0.svg)


Nope - most trips don't actually return to the station they started from.  But how popular is the most popular destination from each station relative to coming back to that station?


```python
# Compute num trips to self + most popular other
V = flow.copy().values
num_same = np.trace(V)
np.fill_diagonal(V, np.nan)
second_best = np.nanmax(V, axis=0).sum()

# Plot num trips returning to same station
plt.figure()
plt.bar([0, 1], [num_same, second_best],
        tick_label=['Same as start', 'Most popular other'])
plt.xlabel('End station')
plt.ylabel('Number of trips')
plt.title('Trips which end at same station\n'+
          'vs most popular other station')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_65_0.svg)


Coming back to the same station is the most popular trip, although there are nearly as many trips as to a second-most-popular station.

Are there clusters of stations that customers ride between?  For example, do people mostly ride from the downtown stations to other downtown stations?  Or, say, between Uptown and Bde Maka Ska?  The matrix from before didn't group stations in any way, or arrange them in any particular order, so we weren't able to get this information from that plot.  However, we can hierarchically cluster the flow patterns between stations, in order to see which groups of stations have many rides between themselves, but not as many to other stations outside the group.

We'll compute the "distance" or "disconnectedness" between a pair of stations by taking the inverse of the number of trips between that pair of stations.  Then we can plot a [dendrogram](https://en.wikipedia.org/wiki/Dendrogram) which shows how disconnected different groups of stations are.


```python
# Normalized flow (proportion trips to OR from)
names = flow.index.tolist() #station names
counts = flow.values #trip counts
sflow = counts+np.transpose(counts) #symmetric
dist = 1.0/(sflow+1) #"distance"
np.fill_diagonal(dist, 0.0) #0 distance to same station

# Dendrogram
sns.set_style("dark")
plt.figure(figsize=(5,20))
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
sdist = squareform(dist)
Z = hierarchy.linkage(sdist, 'ward')
dg = hierarchy.dendrogram(Z, labels=names,
                          orientation='right')
plt.xlabel('Disconnectedness')
plt.ylabel('Station')
plt.title('Hierarchical Clustering of Stations'+
           '\nby traffic between them')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_67_0.svg)



```python
# Get the ordering of the dendrogram
inds = [names.index(sn) for sn in dg['ivl']]

# Get trip counts ordered by dendrogram
counts = flow.copy().values[inds,:]
counts = counts[:,inds]
counts = np.log10(counts)
counts[counts<0] = -1

# Plot the log flow matrix
plt.figure(figsize=(10, 8))
plt.imshow(counts,
           aspect='auto',
           interpolation="nearest")
plt.set_cmap('plasma')
cbar = plt.colorbar(ticks=[-1,0,1,2,3])
cbar.set_label('Number of trips')
cbar.ax.set_yticklabels(['0','1','10','100','1000'])
plt.ylabel('Station FROM')
plt.xlabel('Station TO')
plt.yticks([], [])
snums = range(len(dg['ivl']))
plt.xticks(snums[0::5], dg['ivl'][0::5],
           rotation='vertical')
plt.yticks([], [])
plt.title('Number of trips to and from each station'+
          '\nsorted according to hierarchical clustering')
plt.show()
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:7: RuntimeWarning: divide by zero encountered in log10
      import sys
    


![svg](/assets/img/nice-ride-eda/output_68_1.svg)


The flow patterns between stations are somewhat clustered.  Stations in St. Paul see more traffic between themselves than to stations in Minneapolis (the cluster on the upper left in the matrix plot and the green cluster at the bottom in the dendrogram).  Stations on the University of Minnesota campus see lots of traffic between themselves (the cluster on the middle in the matrix plot), and stations in downtown Minneapolis see mostly traffic between themselves (large cluster in the lower-right).  However, these clusters aren't completely distinct, and there is obviously a lot of traffic even between stations not in the same "cluster", especially across Minneapolis.

We can also visualize the flow of rides between stations on a map!  Let's plot lines connecting each station, where the line width and color corresponds to how many rides there were from a station to all other stations.  Hover over a station in the map below to see where rides from that station went.


```python
# Get number of trips from each station to each other station
# (row i is num trips from station i)
RC = flow.copy().values + 1

# Re-order data in C to match data in stations df
flow_idx = flow.index.tolist()
flow_col = flow['Trips'].columns.tolist()
inds_idx = [flow_idx.index(sn) for sn in stations.Name.tolist()]
inds_col = [flow_col.index(sn) for sn in stations.Name.tolist()]
RC = RC[inds_idx,:]
RC = RC[:,inds_col]

# Convert lat,lon to UTM coordinates
X = lon_to_mercX(stations.Longitude)
Y = lat_to_mercY(stations.Latitude)

# Data for bokeh
source = ColumnDataSource(data=dict(
    name = stations.Name.tolist(),
    X = X,
    Y = Y,
    x0 = X[0]*np.ones_like(X),
    y0 = Y[0]*np.ones_like(Y),
    V = RC[0,:],
    W = RC[0,:]*0.01,
))

# Log color map for line color
mapper = LogColorMapper(palette=Viridis256, 
                        low=0, high=5000)
color = {'field': 'V', 'transform': mapper}

# Plot the points and segments
p = figure(tools="pan,wheel_zoom,reset,save", 
           active_scroll="wheel_zoom")
p.add_tile(CARTODBPOSITRON_RETINA) #background map
p.axis.visible = False
sr = p.segment(x0='x0', y0='y0', x1='X', y1='Y',
               source=source, line_color=color,
               line_width='W', line_cap='round')
cr = p.circle(x='X', y='Y', source=source,
              size=20, color='gray', alpha=0.4, 
              hover_color='gray', hover_alpha=1.0)

# Set lines on hover
min_trips = 10 #don't plot if <10 trips
code = """
var x0 = source.data['x0']
var y0 = source.data['y0']
var X = source.data['X']
var Y = source.data['Y']
var V = source.data['V']
var W = source.data['W']
var indices = cb_data.index['1d'].indices;
if (indices.length > 0) {
    ind = indices[0]
    for (i=0; i<x0.length; i++) {
        x0[i] = X[ind]
        y0[i] = Y[ind]
        V[i] = C[ind][i]
        if (C[ind][i]>%d) {
            W[i] = C[ind][i]*0.01
        } else {
            W[i] = 0.001
        }
    }
}
source.change.emit();
""" % min_trips
callback = CustomJS(args=dict(source=source, C=RC), 
                    code=code)
tooltips = [("Station", "@name")]
p.add_tools(HoverTool(tooltips=tooltips, 
                      callback=callback, 
                      renderers=[cr]))

# Color bar
color_bar = ColorBar(color_mapper=mapper, 
                     location=(0, 0))
p.add_layout(color_bar, 'right')

# Show plot
p.title.text = 'Flow between stations'
p.plot_width = 800 
p.plot_height = 500
show(p)
```








  <div class="bk-root" id="3a5a7318-90cb-4590-bdc2-2c86e6907ab7"></div>





<a id='ride-durations'></a>
## Ride durations

How long did rentals of Nice Ride's bikes usually last? There were an oddly large number of trips larger than 24 hours (but a very small proportion of the total number of trips).


```python
# Show number of trips longer than 24hrs
Ntd = np.count_nonzero(trips['Total duration (Seconds)']>(24*60*60))
print("Number of trips longer than 24 hours: %d ( %0.2g %% )"
      % (Ntd, 100*Ntd/float(len(trips))))
```

    Number of trips longer than 24 hours: 812 ( 0.18 % )
    

But most trips were under an hour:


```python
# Show number of trips shorter than 1hr
Ntd = np.count_nonzero(trips['Total duration (Seconds)']<(24*60))
print("Number of trips shorter than 1 hour: %d ( %0.2g %% )"
      % (Ntd, 100*Ntd/float(len(trips))))
```

    Number of trips shorter than 1 hour: 340375 ( 74 % )
    

Let's look at the distribution of trip durations, but only for those under 24 hours.


```python
# Plot histogram of ride durations
plt.figure()
sns.distplot(trips.loc[trips['Total duration (Seconds)']<(4*60*60),
                       'Total duration (Seconds)']/3600)
plt.xlabel('Ride duration (hrs)')
plt.ylabel('Number of Trips')
plt.title('Ride durations')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_76_0.svg)


<a id='ride-season'></a>
## Ride Season

When are customers renting bikes across the course of the 2017 season?  The first rental is in the beginning of April, and the last rental in the beginning of November:


```python
# Show first and last rentals of the season
print('First rental: ', trips['Start date'].min())
print('Last rental: ', trips['Start date'].max())
```

    First rental:  2017-04-03 09:19:00
    Last rental:  2017-11-05 21:45:00
    

We can also look at the number of rides by month:


```python
# Plot number of rides per month
plt.figure()
sns.countplot(trips['Start date'].dt.month)
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.title('Number of rentals by month in 2017')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_80_0.svg)


The most popular month to rent is July, though there are still a lot of rentals in non-prime months - there are still almost half as many rentals in April and October as in July.

We can also look at the number of rentals by the week in the year, and the day of the year:


```python
# Plot number of rides per week
plt.figure(figsize=(6, 7))
sns.countplot(y=trips['Start date'].dt.week)
plt.ylabel('Week')
plt.xlabel('Number of Trips')
plt.title('Number of rentals by week in 2017')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_82_0.svg)



```python
# Plot number of rides per day of the year
trips.groupby(trips['Start date'].dt.dayofyear)['Start date'].count().plot()
plt.xlabel('Day of the year')
plt.ylabel('Number of rentals')
plt.title('Number of rentals by day of the year in 2017')
holidays = [("Mother's day", 134),
            ("Memorial day", 149),
            ("4th of July", 185),
            ("Labor day", 247), 
            ("Oct 27", 300)]
for name, day in holidays:
    plt.plot([day,day], [0,6000], 
             'k--', linewidth=0.2)
    plt.text(day, 6000, name, fontsize=8, 
             rotation=90, ha='right', va='top')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_83_0.svg)


Like the plot by months, we can see that the number of rentals ramps up and then down again over the course of the summer, with peak rentals on the week leading up to the 4th of July.  There's also a clear increase in the number of rentals on holiday weekends.

There also seems to be an oscillatory pattern to the volume of rentals, which could perhaps be due to the day of the week.  Let's take a look at the number of rentals by the day of the week:


```python
# Plot number of rides per day of the week
plt.figure()
sns.countplot(trips['Start date'].dt.weekday)
plt.xlabel('Day')
plt.ylabel('Number of Trips')
plt.title('Number of rentals by day of the week in 2017')
plt.xticks(np.arange(7),
           ['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'])
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_85_0.svg)


Indeed it looks like there are more rentals on the weekends (and Fridays) than during weekdays, though not by all that much.

Looking at the rides across the year made me wonder if people would continue riding bikes later into the season as long as the weather was nice.  Nice Ride ended their season in early November, but there were still around 1000 rider per day through most of October.  Notice the sharp drop in rentals on Oct 27th, when there was a [sudden cold snap (pdf)](https://www.weather.gov/media/mpx/Climate/MSP/oct2017.pdf) and daily high temperatures dropped from the low 60s (brisk but still pleasant biking weather) to the low 30s.  

What if that cold snap hadn't happened?  It's possible that people may have continued renting bikes, and Nice Ride could have continued earning revenue.  Could Nice Ride optimize when they end their season for the year in order to earn more revenue?  Let's first see how weather affected the number of rentals, and then we'll get back to that question.

<a id='weather'></a>
## Weather

How does riding activity depend on the weather?  Presumably people ride more when it is nice out and not raining, but how large is that effect?

NOAA's National Centers for Environmental Information provides daily [historical weather data](https://www.ncdc.noaa.gov/cdo-web/).  Let's load their data from a weather station about halfway between St. Paul and Minneapolis.  The data includes the daily high and low temperatures, as well as daily precipitation in inches.


```python
# Load weather data
weather = pd.read_csv('../input/WeatherDailyMinneapolis2017.csv')
```


```python
# Convert weather date to datetime
weather['DATE'] = pd.to_datetime(weather['DATE'],
                                 format='%Y-%m-%d')
```


```python
# Plot daily min + max temperature
plt.plot(weather.DATE.dt.dayofyear,
         weather.TMAX, 'C2')
plt.plot(weather.DATE.dt.dayofyear,
         weather.TMIN, 'C1')
plt.legend(['Max', 'Min'])
plt.xlabel('Day of year')
plt.ylabel('Temperature (degrees F)')
plt.title('Daily temperatures in Minneapolis for 2017')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_90_0.svg)



```python
# Plot daily min + max temperature
plt.plot(weather.DATE.dt.dayofyear,
         weather.PRCP)
plt.xlabel('Day of year')
plt.ylabel('Precipitation (in)')
plt.title('Daily precipitation in Minneapolis for 2017')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_91_0.svg)


How does the temperature correlate with the number of rides in a day?  We'll plot the number of rides per day against the daily high temperature (which probably more closely reflects the temperature during most rides than the daily low temperature). 


```python
# Count number of rides per day
trips_per_day = (
    trips
    .groupby(trips['Start date'].dt.dayofyear)
    .size()
    .rename('trips')
)

# Add the trips/day to weather table
weather['dayofyear'] = weather.DATE.dt.dayofyear
weather = (
    weather
    .set_index('dayofyear')
    .join(trips_per_day)
)
```


```python
# Plot max temp vs #trips per day
sns.set_style("darkgrid")
sns.regplot(x='TMAX', y='trips', data=weather)
plt.xlabel('Daily maximum temperature (degrees F)')
plt.ylabel('Number of rentals per day')
plt.xlim([20, 100])
plt.ylim([0, 5000])
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_94_0.svg)


It definitely looks like people ride more often when it's warmer.  Though the peak riding temperature may be around 75-80 degrees, because the number of rides per day appear to peter off when temperatures get into the 90's.

What about precipitation?  Presumably there are less rentals when it's raining.


```python
# Plot precipitation vs #trips per day
sns.regplot(x=weather.PRCP+0.001, 
            y=weather.trips, 
            logx=True, truncate=True)
plt.xlabel('Daily precipitation (in)')
plt.ylabel('Number of rentals per day')
plt.title('Precipitation vs Rentals per day')
plt.xlim([0, 2.1])
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_96_0.svg)


Precipitation didn't have as large an effect on the number of rentals as one might expect.  Even on days with two inches of rain, there are only about 1/3rd less rentals than average.  This could be due to the possibility that, even on days with a large amount of precipitation, it may not have been raining *all* day.  Rentals could still be happening at their normal rates for the part of the day without rain.

Temperatures were highly correlated with the number of rides in a given day, but do temperatures also influence the *duration* of those rentals?


```python
# Get max temp on the day of each rental
max_temps = weather.loc[trips['Start date'].dt.dayofyear, 'TMAX'].values

# Get trip durations
trip_durs = trips['Total duration (Seconds)'].values

# Plot max temp vs ride duration
sns.set_style("white")
h = sns.jointplot(x=max_temps, 
                  y=np.log10(trip_durs), 
                  kind='hex',
                  joint_kws=dict(gridsize=20))
h.set_axis_labels('Daily maximum temperature (degrees F)', 
                  'Duration of rental ( log10(s) )')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_98_0.svg)


There may be a small correlation between the temperature and ride duration, but if so the effect isn't very large.

What about precipitation - does precipitation negatively correlate with ride durations?  The colors of the bins in the plot below represent log10(number of rentals), because with a linear color scale all we can see is a blue line on the left side of the plot! (because the precipitation distribution is so skewed).


```python
# Get precipitation on the day of each rental
precip = weather.loc[trips['Start date'].dt.dayofyear, 'PRCP'].values

# Plot max temp vs ride duration
sns.set_style("white")
h = sns.jointplot(x=precip,
                  y=np.log10(trip_durs), 
                  kind='hex',
                  joint_kws=dict(gridsize=10, bins='log'))
h.set_axis_labels('Daily precipitation (in)', 
                  'Duration of rental ( log10(s) )')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_100_0.svg)


There may be a negative correlation between precipitation and ride duration, but again, if there is one it's very small.  This is more obvious when comparing ride durations on days where there was rain to days where there wasn't any.  The vertical bars in the plot below represent 95% confidence intervals on the mean.


```python
# Show ride durations split by precipitation
precip_bool = precip > 0
sns.pointplot(x=precip_bool, y=trip_durs)
plt.xlabel('Any Precipitation')
plt.ylabel('Mean Ride Duration (s)')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_102_0.svg)


Overall, it looks like temperature is strongly correlated with the number of rentals in a day, while precipitation is negatively correlated with the number of rides, but neither have a large effect on ride durations.  This could be because when it's raining or overly cold, customers decide not to ride *at all* instead of riding for shorter amounts of time.

<a id='dependence-on-season'></a> 
 ## Rental dependence on season
 
Now that we've loaded the weather data, we can get back to our question of whether there is a seasonal effect on the number of rides, independent from the effect of weather.  That is, if the weather was exactly the same, would we still see a change in the number of rides per day over the year?  

The reason that's important is becuase using that information along with weather predictions, Nice Ride would be able to better decide when to end their rentals for the season.  If there lots of rentals in the fall as long as it's nice out, and weather predictions look pleasant, then Nice Ride might want to extend their rental season.  On the other hand, if customers usually stop renting in fall even when it's nice out, Nice Ride might want to end their season early and save on operating costs.

There are a few different ways we could go about determining the effect of season on the number of rides, independent from weather.  We could build a model to predict the number of rides per day, which includes season as a predictor, along with the weather and other factors.  We'd probably want to use a 2nd order polynomial to model the seasonal effect, since we're only looking at data from one year:

$$
\text{RidesPerDay} \sim \beta + \text{DayOfYear} + \text{DayOfYear}^2 + \text{Weather} + \text{etc...}
$$

However, obviously the season will be highly correlated with the temperature!  That means that some of the temperature's effect on the number of rides may be attributed to season if we try to fit both in the same model.  Just to be safe, let's instead fit a model which includes as predictors everything we care about *except* season, and then look at the residuals of that model as a function of season.  Basically what we'll be doing is "removing the effect" of weather from the number of rides, and then looking at how that leftover information changes with season.

First, let's fit an ordinary least squares regression model to predict the number of rides per day from the day of the week, the max daily temperature, and the daily precipitation.


```python
# Build dataframe with day and weather
df = pd.DataFrame()
df['Trips'] = weather.trips
df['Date'] = weather.DATE
df['Day'] = weather.DATE.dt.dayofweek
df['Temp'] = weather.TMAX
df['Precip'] = weather.PRCP + 0.001

# Only fit model on days with trips
df = df.loc[~np.isnan(df.Trips), :]
df.reset_index(inplace=True, drop=True)

# Fit the linear regression model
olsfit = smf.ols('Trips ~ Temp + log(Precip) + C(Day)', data=df).fit()

# Show a summary of the fit
print(olsfit.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  Trips   R-squared:                       0.516
    Model:                            OLS   Adj. R-squared:                  0.498
    Method:                 Least Squares   F-statistic:                     27.76
    Date:                Fri, 03 Aug 2018   Prob (F-statistic):           4.37e-29
    Time:                        03:45:18   Log-Likelihood:                -1701.1
    No. Observations:                 217   AIC:                             3420.
    Df Residuals:                     208   BIC:                             3451.
    Df Model:                           8                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept   -1235.7491    254.655     -4.853      0.000   -1737.784    -733.714
    C(Day)[T.1]   207.5403    159.740      1.299      0.195    -107.377     522.457
    C(Day)[T.2]    22.9151    159.490      0.144      0.886    -291.509     337.339
    C(Day)[T.3]   114.9055    159.346      0.721      0.472    -199.234     429.045
    C(Day)[T.4]   396.5971    159.895      2.480      0.014      81.374     711.820
    C(Day)[T.5]   624.8469    159.983      3.906      0.000     309.451     940.242
    C(Day)[T.6]   200.6730    159.395      1.259      0.209    -113.563     514.909
    Temp           39.8840      3.194     12.489      0.000      33.588      46.180
    log(Precip)   -77.9814     16.684     -4.674      0.000    -110.873     -45.090
    ==============================================================================
    Omnibus:                        4.482   Durbin-Watson:                   1.429
    Prob(Omnibus):                  0.106   Jarque-Bera (JB):                5.960
    Skew:                          -0.002   Prob(JB):                       0.0508
    Kurtosis:                       3.812   Cond. No.                         551.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

Our model definitely captured an effect of the weather.  In the summary table above, the `coef` column contains the coefficients for the variables in the leftmost column.  The coefficient for temperature was \\( \approx 39.9 \\), meaning that for every 10 degree increase in the temperature, Nice Ride can expect \\( \approx 400 \\) more rides per day!  But not *exactly* 400 more rides.  The two rightmost columns show the 95% confidence interval, which show that the model is 95% sure the temperature coefficient is between 33.6 and 46.2.  So the model is very sure that the number of rides per day *increases* with temperature (because the 95% confidence interval is completely above 0), but there is uncertainty as to the exactly how strong that relationship is.

Similarly the coefficient for precipitation is significantly negative, meaning more precipitation means less rides per day.  This makes sense and matches up with our weather analysis from before.

The `C(Day)` rows show the coefficients for each day of the week.  For example the `C(Day)[T.1]` row is for Tuesday (0 is Monday and 6 is Sunday).  The coefficients for these rows correspond to how may more rides per day we can expect on that day than on a Monday.  We can see that only Friday and Saturday have a lot more rides than Monday (because their 95% confidence intervals are completely above 0, while for the other days the 95% confidence intervals encompass 0).

But most of that we already knew!  Now let's look at the residuals of the model as a function of season.


```python
# Predict num rides and compute residual
y_pred = olsfit.predict(df)
resid = df.Trips-y_pred

# Plot Predicted vs actual
sns.set_style("darkgrid")
plt.plot(df.Date.dt.dayofyear, y_pred)
plt.plot(df.Date.dt.dayofyear, df.Trips)
plt.legend(['Predicted', 'Actual'])
plt.xlabel('Day of the Year')
plt.ylabel('Number Daily Rentals')
plt.title('Actual vs Predicted Rides Per Day')
plt.show()

# Plot residuals
plt.figure()
sns.regplot(x=df.Date.dt.dayofyear, 
            y=df.Trips-y_pred,
            order=2, truncate=True,
            line_kws={'color':'C2'})
plt.xlabel('Day of the Year')
plt.ylabel('Residual (Actual-Predicted)')
plt.plot([0,360], [0,0], 'k:')
plt.title('Residuals')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_107_0.svg)



![svg](/assets/img/nice-ride-eda/output_107_1.svg)


It actually looks like rides peter off towards the end of fall, even after taking into account the weather.  This may be because people think of fall as being "not biking season" and so are less likely to rent bikes.  If Nice Ride had continued their season, there likely would have continued to be very few rides per day, even if the weather was nice.  For 2017 at least, it seems Nice Ride made the right choice to end their season when they did.

A more reliable way to do this analysis would be to use data over multiple years.  Nice Ride provides their [historical system data](https://www.niceridemn.org/data/), which goes all the way back to 2010.  To really answer the question of whether there's a seasonal effect, one would want to analyse data from multiple years to see what happened during fall seasons where it *was* nice out late into the season.

<a id='memberships'></a>
## Memberships

Nice Ride also has a membership program, where customers can pay to become members to get discounted rides in the future.  Most of Nice Ride's rentals are actually by customers who have memberships:


```python
# Plot counts of account types
ax = sns.countplot(x='Account type', data=trips)
plt.title('Account types')
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('%d' % (y), (x.mean(), y), 
                ha='center', va='bottom')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_110_0.svg)


Is there a difference between the ride patterns of members and non-members?  Rides by members were much more common during weekdays, while non-member activity peaked on the weekends:


```python
# Plot number of rides per day of the week
sns.set()
plt.figure()
sns.countplot(x=trips['Start date'].dt.weekday,
              hue=trips['Account type'])
plt.xlabel('Day')
plt.ylabel('Number of Trips')
plt.title('Number of rentals by day of the week in 2017')
plt.xticks(np.arange(7),
           ['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'])
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_112_0.svg)


This probably indicates that members more often used bike rentals for commuting, while non-members used bike rentals for recreation on the weekends.  This becomes more obvious when we look at rentals by members and non-members as a function of the hour of the day.


```python
# Plot number of rides by hour
plt.figure()
sns.countplot(x=trips['Start date'].dt.hour,
              hue=trips['Account type'])
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.title('Number of rentals by hour in 2017')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_114_0.svg)


For members, there was a clear increase in traffic around the morning rush hour, lunch, and evening rush hour, while non-members showed a less specific activity pattern which started in the late morning and ended in the evening.

There wasn't a huge difference between members and non-members over the course of the season, besides that more members are still riding towards the end of the season in October:


```python
# Plot number of rides per week
plt.figure(figsize=(6, 7))
sns.countplot(y=trips['Start date'].dt.week,
              hue=trips['Account type'])
plt.ylabel('Week')
plt.xlabel('Number of Trips')
plt.title('Number of rentals by week in 2017')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_116_0.svg)


Members also had shorter trips on average than non-members.  Most member rentals lasted around 10 minutes, while most non-member trips lasted between 10 and 30 minutes.


```python
# Plot histogram of ride durations
plt.figure()
sns.violinplot(x='Total duration (Seconds)',
               y='Account type',
               data=trips.loc[trips['Total duration (Seconds)']<6000,:])
plt.xticks([0,1800,3600,5400])
plt.xlim([0,6000])
plt.title('Ride durations\nMembers vs non-members')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_118_0.svg)


Are there certain stations at which members are more likely to start or end their rentals than non-members?  Let's plot the distribution of rides from and to each station for members as compared to non-members


```python
# Get trips FROM each station, split by account type
trips_from = ( trips
    .groupby(['Start station', 'Account type'])
    .size().to_frame('count').reset_index()
)

# Get trips TO each station, split by account type
trips_to = ( trips
    .groupby(['End station', 'Account type'])
    .size().to_frame('count').reset_index()
)

# Get station list sorted by total # rides from
order = ( trips
    .groupby('Start station')
    .size().to_frame('count').reset_index()
    .sort_values('count')
    ['Start station'].tolist()
)

# Plot trips FROM each station
plt.figure(figsize=(5, 35))
plt.subplot(1,2,1)
sns.barplot(x='count', 
            y='Start station', 
            hue='Account type', 
            data=trips_from,
            order=order)
locs, labels = plt.yticks()
plt.yticks(locs, tuple([s.get_text()[:20] for s in labels]))
plt.tick_params(axis='y', labelsize=5)
plt.title('Trips FROM')

# Plot trips TO each station
plt.subplot(1,2,2)
sns.barplot(x='count', 
            y='End station', 
            hue='Account type', 
            data=trips_to,
            order=order)
plt.yticks([])
plt.title('Trips TO')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_120_0.svg)


There are definitely some stations which have different usage by members as compared to non-members.  For example, the station at Lake Street & Knox Ave sees far more use by non-members, while the station at Washington Ave SE & Union Street sees far more use by members than non-members.  But, it's a bit hard to see any systematic differences in the plot above.

We can also visualize this difference by taking the percent difference between rides by members and non-members. This allows us to more clearly see broad patterns in the data:


```python
# Get percent difference FROM each station
pd_from = (trips_from
           .pivot(index='Start station', 
                  columns='Account type', 
                  values='count')
           .reset_index()
           .sort_values('Start station') )
pd_from['Percent Difference'] = ( 100*
    (pd_from['Casual'] - pd_from['Member']) / 
    (pd_from['Casual'] + pd_from['Member']) )

# Get trips TO each station, split by account type
pd_to = (trips_to
         .pivot(index='End station', 
                columns='Account type', 
                values='count')
           .reset_index()
           .sort_values('End station') )
pd_to['Percent Difference'] = ( 100*
    (pd_to['Casual'] - pd_to['Member']) / 
    (pd_to['Casual'] + pd_to['Member']) )

# Get station list sorted by percent diff from
order = (pd_from
         .sort_values('Percent Difference')
         ['Start station'].tolist() )

# Plot percent difference FROM each station
plt.figure(figsize=(7.2, 35))
plt.subplot(1,2,1)
sns.barplot(x='Percent Difference', 
            y='Start station', 
            data=pd_from,
            order=order)
locs, labels = plt.yticks()
plt.yticks(locs, tuple([s.get_text()[:20] for s in labels]))
plt.tick_params(axis='y', labelsize=5)
plt.ylim([201.5, -2])
plt.text(-6, -1, 
         (r'More Members \\( \leftarrow \\)' + 
          r'\\( \rightarrow \\) More Casual'),
         horizontalalignment='center', 
         color='gray', size=10)
plt.title('Trips FROM')

# Plot trips TO each station
plt.subplot(1,2,2)
sns.barplot(x='Percent Difference', 
            y='End station', 
            data=pd_to,
            order=order)
plt.yticks([])
plt.ylim([201.5, -2])
plt.text(-6, -1, 
         (r'More Members \\( \leftarrow \\)' + 
          r'\\( \rightarrow \\) More Casual'),
         horizontalalignment='center', 
         color='gray', size=10)
plt.title('Trips TO')
plt.show()
```


![svg](/assets/img/nice-ride-eda/output_122_0.svg)


From the plot above, we can clearly see that while some stations are used more by members and others more by casual users, stations tend to be either member- or casual-user-dominated both in terms of outbound and inbound trips.  That is, stations which have more members than casual users starting their trips at that station are more likely to also have more members than casual users ending their trips there, and vice-versa.  There are a few exceptions, of course (for example the station at Dale street and Grand Ave has more member trips which begin at that station but more casual user trips which end at that station).

We can also plot these percent differences on a map to see if there are any geographical patterns.  Red stations in the plot below are stations which see more use by non-members than members, and blue markers are the stations which see more use by members than non-members.


```python
# Get lat + lon of stations
lat = (stations
       .set_index('Name')
       .loc[pd_from['Start station'], 'Latitude']
       .values)
lon = (stations
       .set_index('Name')
       .loc[pd_from['Start station'], 'Longitude']
       .values)

# On hover, show station info
tooltips = [("Station", pd_from['Start station']),
            ("PercDiffFrom", pd_from['Percent Difference']),
            ("PercDiffTo", pd_to['Percent Difference'])]

# Outbound trips 
p1, _ = MapPoints(lat, lon, tooltips=tooltips, 
                  color=pd_from['Percent Difference'],
                  title="Difference between members and non-members",
                  palette=brewer['Spectral'][11], #divergent palette
                  symmetric_color=True, #centered @ 0
                  height=fig_height, width=fig_width)
tab1 = Panel(child=p1, title="Outbound")

# Inbound trips
p2, _ = MapPoints(lat, lon, tooltips=tooltips, 
                  color=pd_to['Percent Difference'],
                  title="Difference between members and non-members",
                  palette=brewer['Spectral'][11], #divergent palette
                  symmetric_color=True, #centered @ 0
                  height=fig_height, width=fig_width)
tab2 = Panel(child=p2, title="Inbound")

show(Tabs(tabs=[tab1, tab2]))
```








  <div class="bk-root" id="25a8a0db-6821-40ef-ad2b-c6f91c4e23f0"></div>





We can see from the map that stations downtown see more use by members than non-members, whereas stations near parks or far from downtown (e.g. by the lakes west of Minneapolis, down by Fort Snelling, Lake Como, etc) see more use by non-members than members.

Nice ride could use this information about member vs. non-member rental patterns to optimize where to put advertisements (or even to decide where to put stations).  For example, Nice Ride might want to put advertisements for their membership program at stations which get traffic from members, so that people with riding patterns similar to that of members will see them, as they may be the most likely portion of Nice Ride's customer base to get a membership.  Of course a better way to determine the best add placement would be to actually place ads, and observe how membership rates change, and decide which add placements are the most effective via A/B testing or a multi-armed bandit algorithm.

<a id='conclusion'></a>
## Conclusion

So we looked at a bunch of data, and plotted a bunch of plots.  We learned some obvious things, such as the fact that most members appear to use Nice Ride for commuting, and the fact that there are more rentals when it's warm out.  We also learned some counter-intuitive things (to me, at least), like the fact that precipitation didn't seem to have a huge impact on ride durations, and that there was actually a seasonal effect on the number of rides which was independent from the weather.

But how can all this information be used to help Nice Ride's business?  Here are some reccomendations for Nice Ride based on our analyses:

**Continuing the rental season late into the year probably isn't worth it.** There appeared to be a seasonal effect on the volume of rentals which was independent of the weather.  Specifically, there were more rides than would be expected (given the weather) in the summer than in early spring and late fall.  Honestly, I was expecting to see *more* rentals in the fall after accounting for weather - both because bike-riding habits formed during the summer die hard (or so I thought), and because I figured there was a continuous growth in Nice Ride's customer base.  However, it appears this intuition was wrong, and that there's even fewer rentals in late fall and early spring than would be expected based on the weather.  This means that is probably isn't worth it for Nice Ride to continue renting bikes late into fall, because with fewer rides, Nice Ride's operations costs may outweigh their revenue from the decreased volume of rentals.

**Nice Ride may want to consider more optimally distributing their docks.** We found that the distribution of the number of docks at each station didn't match up very well to the difference in cumulative demand over the course of the day.  This means that Nice Ride might be spending more time re-distributing their bikes between stations than they would have to if they more optimally distributed their docks.  Of course, there are restrictions on how Nice Ride can distribute their docks, for example it appears the default number of docks for a station is 15, plus increments of 4.  While it might not be worth it for Nice Ride to perfectly match their docks distribution to the difference in cumulative demands, they may want to take a look at stations where the discrepancy is especially bad, and consider adding docks to those stations.  However, they'll also have to balance the costs of new docks against the costs of simply paying employees to re-distribute the bikes more often.

**It might be worth building a model to suggest how and when to more optimally re-distribute bikes.**  We worked out the cumulative demand at each station over the course of the day using the number of inbound and outbound trips on average over the entire year.  However, one could instead build a model which uses the season, day of the week, time relative to certain holidays, the weather, etc, to forecast how many bikes will be available at each station at a given point in time in the future, and also how many inbound and outbound rides would occur as long as there were enough docks or bikes.  From that information, Nice Ride could optimize how many bikes to move between stations, and when!  This could both cut down on how many re-distribution employees they require, and also help alleviate the need to optimally distribute docks among their stations.

**Advertising for memberships will probably be most effective in the downtown and U of M areas.**  We found that members were far more likely than non-members to start and end rides at stations near downtown and the University of Minnesota.  Assuming that advertisements are more likely to be effective on potential customers who behave similarly to current customers (an assumption which may or may not be valid...), then Nice Ride should target any advertising for memberships at stations near downtown and the U of M, because that is where most current members are starting and ending their trips.  Therefore, non-members with member-like riding patterns (for example, a non-member who often commutes to work using Nice Ride) will see the advertisements and may be more likely to sign up for memberships.  However, in order to truly evaluate what advertisements and which placements are the most effective, Nice Ride will want to run a controlled advertisement experiment.
