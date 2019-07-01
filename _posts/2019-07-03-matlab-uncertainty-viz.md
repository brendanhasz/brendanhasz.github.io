---
layout: post
title: "Visualizing multiple sources of uncertainty with semitransparent confidence intervals"
date: 2019-07-03
description: "Performing hyperparameter optimization, and creating ensemble and stacking models to predict customer loyalty."
img_url: /assets/img/uncertainty-viz-matlab/output_57_1.svg
github_url: http://github.com/brendanhasz/matlab-uncertainty-viz
tags: [visualization]
language: [matlab]
comments: true
published: false
---


TODO: image of the egg splat plot
![svg](/assets/img/uncertainty-viz-matlab/histogram1.svg)


Matlab comes with several built-in functions for visualizing undertainty: [histogram](https://www.mathworks.com/help/matlab/ref/matlab.graphics.chart.primitive.histogram.html) for static 1D distributions, [errorbar](http://www.mathworks.com/help/matlab/ref/errorbar.html) for visualizing 1D uncertainty in time series data, and [contour](https://www.mathworks.com/help/matlab/ref/contour.html).  Unfortunately sometimes these default functions for make things a bit... more uncertain than they need to be.  Here I've written some functions which make visualizing multiple sources of uncertainty more clear, and perhaps even aesthetically pleasing!


**Outline**

- [Histogram](#histogram)
- [Error bars](#error-bars)
- [Contour plots](#contour-plots)


## Histogram

The built-in [histogram](https://www.mathworks.com/help/matlab/ref/matlab.graphics.chart.primitive.histogram.html) function is actually pretty great.  By default, it'll plot two overlapping distributions with semitransparency and using different colors:

```matlab
% Generate samples from two distributions
X1 = randn(1000,1);
X2 = 2+randn(1000,1);

% Plot both distributions
figure
histogram(X1)
hold on
histogram(X2)
```

![svg](/assets/img/uncertainty-viz-matlab/histogram1.svg)

If it ain't broke...


## Error bars

Unfortunately, using the [errorbar](http://www.mathworks.com/help/matlab/ref/errorbar.html) function works less well.  If we generate two time series data with error, then we can plot them on top of each other with errorbar:

```matlab
% Generate sample time series data with error
N = 60;
X = linspace(-2, 6, N)';
Y1 = exp(-X)-exp(-2*X);
Y1(X<0) = 0;
Y1 = Y1 + 0.01*randn(N, 1);
E1 = 0.02+0.1*rand(N, 1);
Y2 = exp(-X+1)-exp(-2*(X-1));
Y2(Y2<0) = 0;
Y2 = Y2 + 0.01*randn(N, 1);
E2 = 0.03+0.05*rand(N, 1);

% Plot the time series
figure
errorbar(X, Y1, E1)
hold on
errorbar(X, Y2, E2)
```

![svg](/assets/img/uncertainty-viz-matlab/errorbar1.svg)

Oh ew.  My eyes.  Though it does at least choose different colors for subsequent lines by default, which is nice.  But the error bars are often ovrlapping, which makes it slightly difficult to see what's going on.  What would be better if we could display uncertainty in the form of shaded, semitransparent bounds.  We can do that using the [fill]() plotting function, which plots a function given x,y coordinates of the vertexes.  Therefore, we'll have to plot the upper error bounds from left to right, and then the lower bounds from right to left.  Then we can layer the mean line on top, like this:

```matlab
% Plot shaded, semitransparent error bounds
figure
blue = [0.35 0.7 0.9];
orange = [0.9,0.6,0];
fill([X; flipud(X)], [Y1+E1; flipud(Y1-E1)], blue, ...
     'EdgeColor', 'none', 'facealpha', 0.3)
hold on
plot(X, Y1, 'Color', blue, 'LineWidth', 2)
fill([X; flipud(X)], [Y2+E2; flipud(Y2-E2)], orange, ...
     'EdgeColor', 'none', 'facealpha', 0.3)
plot(X, Y2, 'Color', orange, 'LineWidth', 2)
```

![svg](/assets/img/uncertainty-viz-matlab/errorbar2.svg)


It looks great, and it's a lot easier to tell what's going on.
But, that's a lot of work just for two lines with error bounds!
I wrote a function so I didn't have to worry about all that every time I want to plot a line with error bounds.  It makes plotting time series with error bounds a lot easier:


```matlab
% Plot the time series
figure
ploterr(X, Y1, E1)
ploterr(X, Y2, E2, 'Color', 'orange')
```

![svg](/assets/img/uncertainty-viz-matlab/errorbar2.svg)


You can even use it to plot the error given multiple datapoints, without having to compute the error yourself.  For example, if you have several signals (each trace being a signals in the matrix `Y`):


```matlab
% Multiple time series
Nt = 10; %number of signals
N = 60;
X = linspace(-2, 6, N);
Y1 = exp(-X)-exp(-2*X);
Y1(X<0) = 0;
Y1 = Y1 + 0.1*Y1.*randn(Nt, N) + 0.05*randn(Nt, N);

% Plot the signals
figure
plot(X, Y1')
```

![svg](/assets/img/uncertainty-viz-matlab/signals.svg)

Then you can use `ploterr` to show the standard deviation:

```matlab
ploterr(X, Y1, 'Error', 'std')
```

![svg](/assets/img/uncertainty-viz-matlab/signal_std.svg)

Or the standard error of the mean (the default):

```matlab
ploterr(X, Y1, 'Error', 'sem')
```

![svg](/assets/img/uncertainty-viz-matlab/signal_sem.svg)

And you can have it show the individual points:

```matlab
ploterr(X, Y1, 'Error', 'std', 'ShowPoints', true)
```

![svg](/assets/img/uncertainty-viz-matlab/signal_points.svg)

It returns a line handle so you can draw a legend:

```matlab
h1 = ploterr(X, Y1, 'Error', 'std', 'Color', 'blue');
h2 = ploterr(X, -Y1, 'Error', 'std', 'Color', 'orange');
legend([h1 h2], {'Thing 1', 'Thing 2'})
```

![svg](/assets/img/uncertainty-viz-matlab/ploterr_legends.svg)

It'll even auto-generate numbers given an index!

```matlab
x = linspace(0, 1, 30);
for iL = 1:5
    y = iL+x*iL+0.7*randn(10, length(x));
    ploterr(x, y, 'Color', iL, 'Error', 'std')
end
```

![svg](/assets/img/uncertainty-viz-matlab/rainbow.svg)

There's a bunch of other features too, including support for categorical X variables, and the option to set colors in several different ways, control the line style, the transparency level, plotting percentiles instead of std/sem, etc, etc.  It's up on [my GitHub](https://github.com/brendanhasz/matlab-uncertainty-viz/blob/master/ploterr.m).


## Contour plots

TODO

    2D (kscontour)
        e.g. contour only shows 1
        tho we could show 2 by setting the color
        but if you have 2 peaks, this is ambiguous (does it go up or down?)
        can fix by adding 'ShowText','on'
        introduce kscontour, for density plots after KDE smoothing