# adaBoostAnalysis

Data set description:

* The first part is about using adaboost to approximate a linear function y = 3x and a nonlinear function y = x^2. 
For the linear one, X ranges from 20 to 220, a total of 200 points each having 1 margin.
For the nonlinear one, X ranges from 0 to 2, a total of 200 points each having 0.01 margin.
We input X and get the corresponding function y, which is the label.

* The second part is working with a real world data set, a California Housing Price from python sklearn package.
X has 20640 data points, each having 8 dimensions. 
y, the housing price, divided by $100000, ranges from 0.15 to 5.00 with an average of 2.07.

Approaches:

1. We used L2 loss function here, which is 1/n * (y-y_hat)^2. We used coordinate descent to find the next weak learner  
by considering both the positive and negative direction of alpha (taking absoute value of it). We found the optimal step size by a closed form formula derived from this particular loss function.

2. We used 10 weak learners. For each feature/dimension, we sort the x values first and then find the corresponding 
10 quantile values to be the thresholds and construct the weak learners based on that, each will assign either 1 or 0 
to the datapoints. Added bias as one independent column into the matrix so we have a total of 81 columns.

3. For visualization, each feature should plot the dot values of y first, then plot the weights associated with each 
learner, which will be a piece-wise constant function. Centered the data to better visualize them in the same range.

Conclusion:

1. For the first part, it shows that the number of weak learners is the pivotal factor in how well you approximate 
the function. If we don't have enough weak learners, no matter how many iterations we go, the algorithm cannot converge 
to a good performance.

2. For the second part, the algorithm drops down the error fast. We achieved a good performance with 30 iterations only.
It will converge extremely fast in the classification problem with exponential loss also. The graph shows how each feature 
contributes to the housing price, we can also find the most important variable by calculating variable importance. For 
each feature, we shuffled the entries of it, and recalculated the MSE again. The one with the bigger MSE change is the one 
with more importance.

3. Comparison with linear model also shows that adaBoost is better at capturing nonlinearity in the data.
