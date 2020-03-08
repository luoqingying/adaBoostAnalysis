# adaBoostAnalysis

Data set description:

* The first part is about using adaboost to approximate a linear function y = 3x and a nonlinear function y = x^2. 
For the linear one, X ranges from 20 to 220, a total of 200 points each having 1 margin.
For the nonlinear one, X ranges from 0 to 2, a total of 200 points each having 0.01 margin.
We input X and get the corresponding function y, which is the label.

* The second part is working a real world data set, a california housing price from python sklearn package.
X has 20640 data points, each having 8 dimensions. 
y, the housing price, divided by $100000, ranges from 0.15 to 5.00 with an average of 2.07.

Approaches:

1. We use L2 loss function here, which is 1/n * (y-y_hat)^2. We find the optimal direction by considering both the 
positive and negative direction of alpha. We found the optimal step size by formula derived from this particular 
loss function.

2. We used 10 weak learners. For each feature/dimension, we sort the x values first and then find the corresponding 
10 quantile values to be the thresholds and construct the weak learners based on that.

3. For plotting the graph, we centered the y values and also the cumulative alphas to have a better visualization so 
that we can know the rough trend of these two values. The graph is the step function of each feature, the x axis is the 
feature values quantiles, blue line is the cumulative alphas.

Concusion:

1. For the first part, it shows that the number of weak learners is the pivotal factor in how well you approximate 
the function. If we don't have enough weak learners, no matter how many iterations we go, the algorithm cannot cconverge 
to a good performance.

2. For the second part, the graph shows how each feature contributes to the housing price, we can also find the most 
important variable by calculating variable importance. For each feature, we shuffled the entries of it, and recalculated 
the MSE again. The one with the bigger MSE change is the one with more importance.

3. Comparison with linear model also shows that adaBoost is better at capturing nonlinearity in the data.
