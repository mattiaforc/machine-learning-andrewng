# Machine Learning Notes
## Introduction
Tom Mitchell provides a more modern definition of Machine Learning:
> *"A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E."*

In general, any machine learning problem can be assigned to one of two broad classifications:
-   Supervised learning 
-   Unsupervised learning.

## Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output. Supervised learning problems are categorized into:
-   **Regression problem**: In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.
-   **Classification problem**: In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.
#### Examples:
-   **Regression**: Given a picture of a person, we have to predict their age on the basis of the given picture.
-   **Classification**: Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.



## Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by **clustering** the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.
#### Examples: 
-   **Clustering**: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.
-   **Non-clustering**: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

## Model Representation
To establish notation for future use, we’ll use $x^{(i)}$
to denote the “input” variables, also called input features, and $y^{(i)}$ to denote the “output” or target variable that we are trying to predict. A pair $(x^{(i)} , y^{(i)} )$ is called a **training example**, and the dataset that we’ll be using to learn - a list of $m$ training examples $(x(i),y(i));i=1,...,m$ - is called a **training set**. Note that the superscript “$(i)$” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values. In this example, $X = Y = ℝ.$

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $h : X → Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. 

## Cost Function
We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average difference of all the results of the hypothesis with inputs from $x$'s and the actual output $y$'s.

$J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x_i)-y_i)^2$

To break it apart, it is $\frac{1}{2}\bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta (x_{i}) - y_{i}$, or the difference between the predicted value and the actual value.

This function is otherwise called the "**Squared error function**", or "**Mean squared error**". The mean is halved $\left(\frac{1}{2}\right)$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term.

If we try to think of it in visual terms, our training data set is scattered on the $x$ - $y$ plane. We are trying to make a straight line (defined by $h_\theta(x)$ ) which passes through these scattered data points.

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_0, \theta_1)$ will be 0.

## Gradient Descent 
So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields $\theta_0$ and $\theta_1$ (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing $x$ and $y$ itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters. We put $\theta_0$ on the $x$ axis and $\theta_1$ on the $y$ axis, with the cost function on the vertical $z$ axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup:

![](Pictures/1-1.png) 

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $\alpha$, which is called the **learning rate**.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter $\alpha$. A smaller $\alpha$ would result in a smaller step and a larger $\alpha$ results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(\theta_0,\theta_1)$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places. The **gradient descent algorithm** is:

*Repeat until convergence:*

$\theta_j := \theta_j -\alpha\frac{∂}{∂\theta_j}J(\theta_0, \theta_1)$

*where $j, i$ represents the feature index number*.

At each iteration $j$, one should simultaneously update the parameters $\theta_0, \theta_1, ..., \theta_n$. 
Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation.

On a side note, we should adjust our parameter $\alpha$ to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong. But, how does gradient descent converge with a fixed step size $\alpha$?

The intuition behind the convergence is that $\frac{d}{d\theta_1} J(\theta_1)$ approaches $0$ as we approach the bottom of our convex function. At the minimum, the derivative will always be $0$ and thus we get:

$\theta_1 := \theta_1-\alpha*0$

This means gradient descent can converge to a local minimum, even with the **learning rate **$\alpha$** fixed**. As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease $\alpha$ over time.

## Gradient Descent For Linear Regression
When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :

*Repeat until convergence: {*

$\theta_0 := \theta_0 -\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x_i)-y_i)$

$\theta_1 := \theta_1 -\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x_i)-y_i) x_i$

*} where $m$ is the size of the training set, $\theta_0$ a constant that will be changing simultaneously with $\theta_1$ and $x_{i}, y_{i}$ are values of the given training set (data)*.

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate. So, this is simply gradient descent on the original cost function $J$. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate $\alpha$ is not too large) to the global minimum. Indeed, $J$ is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function:

![](Pictures/1-2.png)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at $(48,30)$. The $x$’s in the figure (joined by straight lines) mark the successive values of $\theta$ that gradient descent went through as it converged to its minimum.

## Matrix Operations

### Matrix Vector Moltiplication

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$$\begin{pmatrix}a & b\\\ c & d\\e & f\end{pmatrix}*\begin{pmatrix}x \\ y\end{pmatrix}=\begin{pmatrix}a*x+b*y\\c*x+d*y\\e*x+f*y\end{pmatrix}$$

The result is a vector. The **number of columns** of the **matrix** must **equal the number of rows of the vector**.
An $m$ x $n$ matrix multiplied by an $n$ x $1$ vector results in an $m$ x $1$ vector.

### Matrix Matrix Moltiplication
We multiply two matrices by breaking it into several vector multiplications and concatenating the result.
$$\begin{pmatrix}a & b\\\ c & d\\e & f\end{pmatrix}*\begin{pmatrix}w & x\\ y & z\end{pmatrix}=\begin{pmatrix}a*w+b*y & a*x+b*z\\c*w+d*y & c*x+d*z\\e*w+f*y & e*x+f*z\end{pmatrix}$$

An $m$ x $n$ matrix multiplied by an $n$ x $o$ matrix results in an $m$ x $o$ matrix. In the above example, a $3$ x $2$ matrix times a $2$ x $2$ matrix resulted in a $3$ x $2$ matrix.

To multiply two matrices, the **number of columns of the first matrix** must **equal** the **number of rows of the second matrix**.