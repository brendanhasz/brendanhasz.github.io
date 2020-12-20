---
layout: post
title: "Visualizing multiple sources of uncertainty with semitransparent confidence intervals"
date: 2019-07-03
description: "Improving on Matlab's default plotting tools for uncertainty visualization."
img_url: /assets/img/uncertainty-viz-matlab/ksc2.svg
github_url: http://github.com/brendanhasz/matlab-uncertainty-viz
tags: [visualization]
language: [matlab]
comments: true
published: true
---


![svg](/assets/img/uncertainty-viz-matlab/ksc2.svg)


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

Oh ew.  My eyes.  Though it does at least choose different colors for subsequent lines by default, which is nice.  But the error bars are often overlapping, which makes it slightly difficult to see what's going on.  What would be better if we could display uncertainty in the form of shaded, semitransparent bounds.  We can do that using the [fill]() plotting function, which plots a function given x,y coordinates of the vertexes.  Therefore, we'll have to plot the upper error bounds from left to right, and then the lower bounds from right to left.  Then we can layer the mean line on top, like this:

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

It'll even auto-generate colors given an index!

```matlab
x = linspace(0, 1, 30);
for iL = 1:5
    y = iL+x*iL+0.7*randn(10, length(x));
    ploterr(x, y, 'Color', iL, 'Error', 'std')
end
```

![svg](/assets/img/uncertainty-viz-matlab/rainbow.svg)

There's a bunch of other features too, including support for categorical X variables, and the option to set colors in several different ways, control the line style, the transparency level, plotting percentiles instead of std/sem, etc, etc.  The code is up on [my GitHub](https://github.com/brendanhasz/matlab-uncertainty-viz/blob/master/ploterr.m).


## Contour plots

Similarly, creating contour plots of 2D distributions can be a pain using Matlab's built-ins.  Supposing we have two sets of points drawn from two distributions:

```matlab
% Generate samples from 2D distributions
X1 = randn(100, 2);
X2 = 2+randn(1000, 2);

% Show the points
figure
plot(X1(:, 1), X1(:, 2), '.')
hold on
plot(X2(:, 1), X2(:, 2), '.')
```

![svg](/assets/img/uncertainty-viz-matlab/con1_scatter.svg)

Then we can plot two separate histograms of their densities:

```matlab
% Plot 2 histograms
figure
edges = {linspace(-4, 4, 11), linspace(-4, 4, 11)};
subplot(1, 2, 1)
    [N1, c] = hist3(X1, 'Edges', edges);
    imagesc(c{1}, c{2}, N1)
    set(gca, 'YDir', 'normal')
    title('X1')
subplot(1, 2, 2)
    [N2, c] = hist3(X2, 'Edges', edges);
    imagesc(c{1}, c{2}, N2)
    set(gca, 'YDir', 'normal')
    title('X2')
```

![svg](/assets/img/uncertainty-viz-matlab/con2_hist.svg)

Unfortunately we can't superimpose the two histograms to get a better idea of how well they overlap.  We can use contour plots, however, to visualize the overlapping distributions.  Though by default Matlab's [`contour`](https://www.mathworks.com/help/matlab/ref/contour.html) function uses the same colormap for both...

```matlab
% Plot 2 contours
figure
subplot(1, 2, 1)
    [N1, c] = hist3(X1, 'Edges', edges);
    contour(c{1}, c{2}, N1, 5)
    title('X1')
subplot(1, 2, 2)
    [N2, c] = hist3(X2, 'Edges', edges);
    contour(c{1}, c{2}, N2, 5)
    title('X2')
```

![svg](/assets/img/uncertainty-viz-matlab/con3_2contour.svg)

We can manually set the color of the lines for both plots, but then we loose information about in what direction the contours are going.  For example, in the plot below, are the two small contour lines at the top of X2 peaks, or are they valleys?

```matlab
% Plot 2 contours
figure

[N1, c] = hist3(X1, 'Edges', edges);
contour(c{1}, c{2}, N1, 5, 'r')
title('X1')

[N2, c] = hist3(X2, 'Edges', edges);
contour(c{1}, c{2}, N2, 5), 'b'
title('X2')
```

![svg](/assets/img/uncertainty-viz-matlab/con5_2color.svg)

To get each contour to have its own colormap, we need to create two separate axes for each contour, and then assign the colormap for each independently.  This gets a bit messy, because we then have to set one or the other to be invisible, make custom colormaps (because Matlab doesn't really come with different categories of continuous colormaps...), etc.

```matlab
% Plot the contours on two separate axes
figure
ax1 = axes;
[N1, c] = hist3(X1, 'Edges', edges);
[~, h1] = contour(c{1}, c{2}, N1, 5);
view(2)
hold on
ax2 = axes;
[N2, c] = hist3(X2, 'Edges', edges);
[~, h2] = contour(c{1}, c{2}, N2, 5);
linkaxes([ax1,ax2])
legend([h1 h2], {'X1', 'X2'})
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];

% Custom color maps
reds = [linspace(0.87, 0.9, 64)', ...
        linspace(0.17, 0.8, 64)', ...
        linspace(0.15, 0.75, 64)'];
blues = [linspace(0.19, 0.80, 64)', ...
         linspace(0.51, 0.85, 64)', ...
         linspace(0.74, 0.90, 64)'];

% Set different colormaps for the two axes
colormap(ax1, reds)
colormap(ax2, blues)
```

![svg](/assets/img/uncertainty-viz-matlab/con6_2cmap.svg)

What I've found to be the least visually painful, and the most interperatable, is to use semi-transparent filled contours.  I wrote a Matlab script which uses kernel density estimation to smooth the inupt datapoints, and then computes the [contour matrix](https://www.mathworks.com/help/matlab/ref/contour.html#mw_6bc1813f-47cf-4c44-a454-f937ea210ab6) as output by `contour` to generate the contours with [`patch`](https://www.mathworks.com/help/matlab/ref/patch.html).  This results in much nicer-looking contour plots which require less boilerplate code (well, once you have the function...):

```matlab
figure
h1 = kscontour(X1);
h2 = kscontour(X2, 'Color', 'orange');
legend([h1 h2], {'X1', 'X2'})
```

![svg](/assets/img/uncertainty-viz-matlab/ksc1.svg)

Ths makes it easy to see what's going on even when you have a bunch of different distributions:

```matlab
X3 = mvnrnd([3, 0], eye(2), 100);

figure
kscontour(X1, 'Color', 'blue');
kscontour(X2, 'Color', 'orange');
kscontour(X3, 'Color', 'green');
```

![svg](/assets/img/uncertainty-viz-matlab/ksc2.svg)

It'll take a cell array of matrixes, and auto-color the resulting contours, which makes things even easier when you have many distributions:

```matlab
figure
kscontour({X1, X2, X3});
```

![svg](/assets/img/uncertainty-viz-matlab/ksc2.svg)

You can view all the datapoints which went into the kernel density estimation:

```matlab
figure
kscontour(X1, 'Color', 'blue',   'ShowPoints', true);
kscontour(X2, 'Color', 'orange', 'ShowPoints', true);
```

![svg](/assets/img/uncertainty-viz-matlab/ksc3.svg)

And you can also control the number of contours:

```matlab
figure
kscontour(X1, 'Nlevels', 8);
```

![svg](/assets/img/uncertainty-viz-matlab/ksc4.svg)

`kscontour` is also available [on my GitHub](https://github.com/brendanhasz/matlab-uncertainty-viz/blob/master/kscontour.m).
