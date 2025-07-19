# Math Cheatsheet for ML
- [Math Cheatsheet for ML](#math-cheatsheet-for-ml)
  - [**Calculus**](#calculus)
    - [Gradient](#gradient)
    - [Lagrange Multiplier](#lagrange-multiplier)
    - [Taylor Expansion](#taylor-expansion)
    - [Newton's Method](#newtons-method)
      - [Gauss-Newton Method](#gauss-newton-method)
  - [**Linear Algebra**](#linear-algebra)
    - [Differentiation](#differentiation)
      - [Numerator Layout](#numerator-layout)
        - [Gradients of Vectors](#gradients-of-vectors)
        - [Jacobian](#jacobian)
        - [Hessian](#hessian)
        - [Chain Rule](#chain-rule)
        - [Gradients of Matrices](#gradients-of-matrices)
      - [Denominator Layout](#denominator-layout)
        - [Chain Rule](#chain-rule-1)
        - [Gradients of Matrices](#gradients-of-matrices-1)
    - [Least Squares](#least-squares)
    - [Determinant](#determinant)
      - [3 basic properties](#3-basic-properties)
      - [derived properties](#derived-properties)
      - [how to calculate](#how-to-calculate)
    - [Eigenvalues \& Eigenvectors](#eigenvalues--eigenvectors)
    - [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
  - [**Probability**](#probability)
    - [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
    - [Covariance](#covariance)
    - [Gaussian Distribution](#gaussian-distribution)
      - [$\\Delta$: Mahalanobis distance](#delta-mahalanobis-distance)
      - [Exponent](#exponent)
      - [Conditional](#conditional)
      - [Marginal](#marginal)
      - [Maximum Likelihood](#maximum-likelihood)
  - [**Information Theory**](#information-theory)
    - [Entropy](#entropy)
    - [Cross Entropy](#cross-entropy)
    - [KL Divergence](#kl-divergence)

## **Calculus**

### Gradient

def $z = f(\bold{x})$ with $\bold{x} = \begin{bmatrix}
  x_1\\x_2\\...\\x_n
\end{bmatrix}$

$\nabla f = \frac{\partial f}{\partial \bold{x}} = [\frac{\partial f}{\partial x_1},..., \frac{\partial f}{\partial x_n}]$ is the formula of the function's gradient and also the partial derivative of the input vector, **output**:
- is a **row vector** in the input space.
- go to the **direction** where goes **upward** fastest.
- **magnitude** (length) is the gradient of this direction.

in addition:

- $ \quad \nabla f(\bold{x}) + \nabla g(\bold{x}) \\
= (\frac{\partial f(\bold{x})}{\partial x_1} + \frac{\partial g(\bold{x})}{\partial x_1},..., \frac{\partial f(\bold{x})}{\partial x_n} + \frac{\partial g(\bold{x})}{\partial x_n}) \\
= (\frac{\partial f(\bold{x}) + g(\bold{x})}{\partial x_1},...,\frac{\partial f(\bold{x}) + g(\bold{x})}{\partial x_n}) \\
= \nabla (f(\bold{x}) + g(\bold{x}))$

- $\nabla(fg) = f\nabla g + g\nabla f$
- $\nabla(\frac{f}{g}) = \frac{g\nabla f - f\nabla g}{g^2}$
- $\partial (x^T) = (\partial x)^T$
- $\frac{\partial x^T A x}{\partial x} = x^T(A + A^T)$



### Lagrange Multiplier

**Q1**: constrain $g(x,y) = 0$, find extreme value of $f(x,y)$

$\to$ (at extreme value point $(x_0,y_0)$, $\nabla f(x_0,y_0)$ and $\nabla g(x_0,y_0)$ should have same or opposite direction)

$\to$ $\nabla f(x_0,y_0) = \lambda \nabla g(x_0,y_0), \lambda \in R$

$\to$ def $F(x,y) = f(x,y) - \lambda g(x,y)$, $\nabla F(x_0,y_0) = \vec{0}$

$\to$ solve the equations:
$\quad\nabla F(x_0,y_0) = \vec{0} \quad g(x,y) = 0$

**Q2**: constrains $g_1(x,y) = 0$, $g_2(x,y) = 0$, find extreme value of $f(x,y)$

$\to$ $F(x,y) = f(x,y) - \lambda_1 g_1(x,y) - \lambda_2 g_2(x,y)$

$\to$ ...



### Taylor Expansion
for a given $y = f(x)$

$$f(x_1)-f(x_0) = f'(x_0)(x_1-x_0) + o(x_1-x_0)$$

![taylor1](taylor1.png)

then we get:

$f(x_1)-f(x_0) = f'(x_0)(x_1-x_0) + \frac{f''(x_0)}{2!}(x_1-x_0)^2 + \frac{f^3(x_0)}{3!}(x_1-x_0)^3 + ... + \frac{f^n(x_0)}{n!}(x_1-x_0)^n + ...$

then we get:

$f(x_1) = f(x_0) + f'(x_0)(x_1-x_0) + \frac{f''(x_0)}{2!}(x_1-x_0)^2 + \frac{f^3(x_0)}{3!}(x_1-x_0)^3 + ... + \frac{f^n(x_0)}{n!}(x_1-x_0)^n + ...$

$n \to \infty$

or: 

$f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f^3(0)}{3!}x^3 + ... + \frac{f^n(0)}{n!}x^n + ...$

$n \to \infty$

$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + ...$


### Newton's Method
given function $f: R^n \to R$, its secondary Taylor expansion:
$$f(x_1) = f(x_0) + \nabla f(x_0)(x_1 - x_0) + \frac{1}{2}(x_1 - x_0)^T \nabla^2f(x_0) (x_1 - x_0)$$

we wish to know when does $\frac{\partial f}{\partial x} = 0$, so we differentiate both sides:
$$\nabla f(x_1) = \nabla f(x_0) + (\nabla^2f(x_0) (x_1 - x_0))^T = 0$$
after solving the equation: 
$$x_1 = x_0 - (\nabla^2 f(x_0))^{-1}(\nabla f(x_0))^T$$
we get iteration formula (introducing [Hessian matrix](#Hessian) to replace $\nabla^2$):
$$x_{k + 1} = x_{k} - (H_f(x_{k}))^{-1}(\nabla f(x_{k}))^T$$

now, in order to answer where the minimum of $f$ is, we could do the following things:
- initiate a $x_0$ as a first approximation of the min point.
- (for better explanation) get $x_1 = x_0 - (H_f(x_0))^{-1}(\nabla f(x_0))^T$
- iteratively do $x_{k+1} = x_{k} - (H_f(x_{k}))^{-1}(\nabla f(x_{k}))^T$ until the value is satisfiable.

#### Gauss-Newton Method
optimization on least square problem $f(x) = y - Ax$, minimize $||f(x)||^2$, Taylor expand $f(x)$ only:
$$\frac{\partial || f(x_0) + \nabla f(x_0)(x_1 - x_0)||^2}{\partial x_1} = 0$$

$$x_1 = x_0 - (\nabla f^T(x_0)\nabla f(x_0))^{-1}\nabla f^T(x_0)f(x_0)$$

Then $\nabla f(x_0)$ and $f(x_0)$ are needed only.

## **Linear Algebra**

### Differentiation
#### Numerator Layout
##### Gradients of Vectors
Given function $\mathbf{f}: R^n \to R^m$, $\mathbf{x} \in R^n$, $\mathbf{f}(\mathbf{x}) \in R^m$

$$\frac{d \mathbf{f}(\mathbf{x})}{d \mathbf{x}} = \begin{bmatrix}
  \frac{\partial f_1(\mathbf{x})}{\partial x_1} & ... & \frac{\partial f_1(\mathbf{x})}{\partial x_n} \\
  ... & ... & ... \\
  \frac{\partial f_m(\mathbf{x})}{\partial x_1} & ... & \frac{\partial f_m(\mathbf{x})}{\partial x_n} \\
\end{bmatrix}_{m \times n}$$

> Sometimes it will be written as $\frac{d \mathbf{f}(\mathbf{x})}{d \mathbf{x}^T}$. But in *Mathematics for Machine Learning, Deisenroth et al.*, it is written as $\frac{d \mathbf{f}(\mathbf{x})}{d \mathbf{x}}$.

##### Jacobian
The jacobian matrix of function $\mathbf{f}$ is defined as follows, which is the local approximated linear transformation of function $\mathbf{f}$ (maybe nonlinear transformation).

$$J_{\mathbf{f}(\mathbf{x})} = \frac{d \mathbf{f}(\mathbf{x})}{d \mathbf{x}} = \begin{bmatrix}
  \frac{\partial f_1(\mathbf{x})}{\partial x_1} & ... & \frac{\partial f_1(\mathbf{x})}{\partial x_n} \\
  ... & ... & ... \\
  \frac{\partial f_m(\mathbf{x})}{\partial x_1} & ... & \frac{\partial f_m(\mathbf{x})}{\partial x_n} \\
\end{bmatrix}_{m \times n}$$

Consider scalar Newton's Method:
$$
f(x_1) \approx f(x_2) + f'(x_2)(x_1 - x_2)
$$

Now for $\mathbf{f}: R^n \to R^m$:
$$
\underset{m}{\mathbf{f}(\mathbf{x_1})} \approx \underset{m}{\mathbf{f}(\mathbf{x_2})} + \underset{m \times n}{J_{\mathbf{f}(\mathbf{x_2})}} \underset{n}{(\mathbf{x_1} - \mathbf{x_2})}
$$

##### Hessian
Given another function $f: R^n \to R$, $\mathbf{x} \in R^n$, hessian matrix is:
$$
H_{f(\bold{x})} = \nabla^2f(\bold{x}) = \begin{bmatrix}
  \frac{\partial^2 f}{\partial x_1^2}&...&\frac{\partial^2 f}{\partial x_1x_n}\\
  ...&...&...\\
  \frac{\partial^2 f}{\partial x_nx_1}&...&\frac{\partial^2 f}{\partial x_n^2}\\
\end{bmatrix}_{n \times n}
$$

##### Chain Rule
Given function $\mathbf{h}: R^n \to R^m$, $\mathbf{x} \in R^n$, $\mathbf{h}(\mathbf{x}) = \mathbf{f}(\mathbf{g}(\mathbf{x}))$, where $\mathbf{g}: R^n \to R^k, \mathbf{f}: R^k \to R^m$.
$$
\underset{m \times n}{\frac{d \mathbf{h}(\mathbf{x})}{d \mathbf{x}}} = \underset{m \times k}{\frac{\partial \mathbf{f}(\mathbf{g}(\mathbf{x}))}{\partial \mathbf{g}(\mathbf{x})}} \underset{k \times n}{\frac{\partial \mathbf{g}(\mathbf{x})}{\partial \mathbf{x}}}
$$

##### Gradients of Matrices
Given function $\mathbf{f}: R^{m \times n} \to R^{p \times q}, X \in R^{m \times n}$, the differentiation is a tensor:

$$
\frac{d \mathbf{f}(X)}{d X} \in R^{(p \times q) \times (m \times n)}
$$

#### Denominator Layout
Given function $\mathbf{f}: R^n \to R^m$, $\mathbf{x} \in R^n$, $\mathbf{f}(\mathbf{x}) \in R^m$

$$\frac{d \mathbf{f}(\mathbf{x})}{d \mathbf{x}} = \begin{bmatrix}
  \frac{\partial f_1(\mathbf{x})}{\partial x_1} & ... & \frac{\partial f_m(\mathbf{x})}{\partial x_1} \\
  ... & ... & ... \\
  \frac{\partial f_1(\mathbf{x})}{\partial x_n} & ... & \frac{\partial f_m(\mathbf{x})}{\partial x_n} \\
\end{bmatrix}_{n \times m}$$

##### Chain Rule
Given function $\mathbf{h}: R^n \to R^m$, $\mathbf{x} \in R^n$, $\mathbf{h}(\mathbf{x}) = \mathbf{f}(\mathbf{g}(\mathbf{x}))$, where $\mathbf{g}: R^n \to R^k, \mathbf{f}: R^k \to R^m$.
$$
\underset{n \times m}{\frac{d \mathbf{h}(\mathbf{x})}{d \mathbf{x}}} = \underset{n \times k}{\frac{\partial \mathbf{g}(\mathbf{x})}{\partial \mathbf{x}}} \underset{k \times m}{\frac{\partial \mathbf{f}(\mathbf{g}(\mathbf{x}))}{\partial \mathbf{g}(\mathbf{x})}}
$$

##### Gradients of Matrices
Given function $\mathbf{f}: R^{m \times n} \to R^{p \times q}, X \in R^{m \times n}$, the differentiation is a tensor:

$$
\frac{d \mathbf{f}(X)}{d X} \in R^{(m \times n) \times (p \times q)}
$$

### Least Squares
There are $m$ data, each with $n$ features.

Data: $X_{m\times (n+1)} = \begin{bmatrix}
  1 & x_{11} & x_{12} & ... & x_{1n}\\
  1 & x_{21} & x_{22} & ... & x_{2n}\\
  ... & ... & ... & ... & ...\\
  1 & x_{m1} & x_{m2} & ... & x_{mn}
\end{bmatrix}$

Label: $Y_{m \times 1} = \begin{bmatrix}
  y_1\\
  y_2\\
  ...\\
  y_m
\end{bmatrix}$

to predict $\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$, construct: $\theta_{(n+1) \times 1} = \begin{bmatrix}
  \theta_0\\
  \theta_1\\
  ...\\
  \theta_n
\end{bmatrix}$

therefore $\hat{Y} = X\theta$

we want least square error, loss function $L = \frac{1}{2}(X\theta - Y)^T(X\theta - Y)$ to be minimized.

$\to\frac{\partial L}{\partial \theta} = 0$

$\to\theta = (X^TX)^{-1}X^TY$


### Determinant
**square** matrix $A$, its determinant $\det A$ is a number, $A = \begin{bmatrix}
  a & b\\
  c & d
\end{bmatrix}$, $\det A = \begin{vmatrix}
  a & b\\
  c & d
\end{vmatrix}$

#### 3 basic properties
- $\det I = 1$

- row exchanges, sign changes: eg. $\begin{vmatrix}
  c & d\\
  a & b
  \end{vmatrix} = - \begin{vmatrix}
  a & b\\
  c & d
  \end{vmatrix}$

- add vector in row, add determinant in row: eg. $\begin{vmatrix}
  a + a' & b + b'\\
  c & d
  \end{vmatrix} = \begin{vmatrix}
  a & b\\
  c & d
  \end{vmatrix} + \begin{vmatrix}
  a' & b'\\
  c & d
  \end{vmatrix}$

#### derived properties
- if $A$ is singular, then $\det A = 0$.
- ...

#### how to calculate
$A = \begin{bmatrix}
  a & b\\
  c & d
\end{bmatrix}, \det A = ad - bc$

If: $A = \begin{bmatrix}
  a_{11} & a_{12} & ... & a_{1n}\\
  a_{21} & a_{22} & ... & a_{2n}\\
  ... & ... & ... & ...\\
  a_{n1} & a_{n2} & ... & a_{nn}
\end{bmatrix}$

let: $M_{ij} = \begin{bmatrix}
  a_{11} & ... & a_{1(j-1)} & a_{1(j+1)} & ... & a_{1n}\\
  ... & ... & ... & ... & ... & ...\\
  a_{(i-1)1} & ... & a_{(i-1)(j-1)} & a_{(i-1)(j+1)} & ... & a_{(i-1)n}\\
  a_{(i+1)1} & ... & a_{(i+1)(j-1)} & a_{(i+1)(j+1)} & ... & a_{(i+1)n}\\
  ... & ... & ... & ... & ... & ...\\
  a_{n1} & ... & a_{n(j-1)} & a_{n(j+1)} & ... & a_{nn}
\end{bmatrix}$

let: $c_{ij} = (-1)^{j + 1} \det M_{ij}$

then: $\det A = \sum_{r = 1}^{n} a_{ir}c_{ir}$

in another word: $\det A = a_{11}\det M_{11} - a_{12}\det M_{12} + a_{13}\det M_{13} - ... + (-1)^{n+1} a_{1n}\det M_{1n}$

### Eigenvalues & Eigenvectors
**square** matrix $A$, its eigenvalue $\lambda$ its eigenvector $v$, $Av = \lambda v$.

$\to (A - \lambda I) v = 0$

$v$ other than $ 0 \to (A - \lambda I)$ singular

$\to \det(A - \lambda I) = 0$

### Singular Value Decomposition (SVD)

## **Probability**
### Maximum Likelihood Estimation (MLE)
Given probability distribution $P(x|\mu)$ and $N$ experiment results $X$. Parameter $\mu$ is estimated most likely to be $\mu_{ML}$

Do it step by step:
1. Probability of these $N$ experiments: $P(X|\mu) = \prod_{n = 1}^{N}P(x_n|\mu)$
2. Log it: $\ln P(X|\mu) = \sum_{n = 1}^{N}\ln P(x_n|\mu)$
3. Maximum $P(X|\mu)$, $\frac{\partial \ln P(X|\mu)}{\partial \mu} = \frac{\partial \sum_{n = 1}^{N}\ln P(x_n|\mu)}{\partial \mu} = 0$
4. Solve equation: $\mu = \mu_{ML}$

### Covariance
$D$-dimensional vector $\bold{x} = [x_1,x_2,...,x_D]^T$

Covariance:
$$
cov(x_i,x_j) = E((x_i - E(x_i))(x_j - E(x_j))), i,j\in\set{1,2,...,D}
$$
Properties:
$cov(x_i,x_i) = E((x_i - E(x_i))^2) = var(x_i)$
$cov(x_i,x_j) = cov(x_j,x_i)$

Covariance matrix:
$$
\Sigma = \begin{bmatrix}
  cov(x_1,x_1) & ... & cov(x_1,x_D)\\
  ...&...&...\\
  cov(x_D,x_1) & ... & cov(x_D,x_D)\\
\end{bmatrix}
$$
Properties:
$\Sigma = \Sigma^T$
$\Sigma^{-1} = \Lambda$

### Gaussian Distribution
Single variable $x$ with mean $\mu$ and variance $\sigma^2$:
$$
N(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\{-\frac{1}{2\sigma^2}(x - \mu)^2\}
$$

$D$-dimensional vector $\bold{x}$ with mean $\mu$ ($D$-dimensional vector) and covariance matrix $\Sigma$ (matrix $D\times D$):
$$
N(\bold{x}|\mu,\Sigma) = \frac{1}{(2\pi)^{D/2} \det(\Sigma)^{1/2}}\exp\{-\frac{1}{2}(\bold{x} - \mu)^T\Sigma^{-1}(\bold{x} - \mu)\}
$$

#### $\Delta$: Mahalanobis distance
$\Delta$: Mahalanobis distance from $\mu$ to $\bold{x}$, $\Delta^2 = (\bold{x} - \mu)^T\Sigma^{-1}(\bold{x} - \mu)$

#### Exponent
$C$ is constant.
$$
\ln N(\bold{x}|\mu,\Sigma) = -\frac{1}{2}(\bold{x} - \mu)^T\Sigma^{-1}(\bold{x} - \mu) + C\\
\qquad\qquad\quad = -\frac{1}{2}\bold{x}^T\Sigma^{-1}\bold{x} + \bold{x}^T\Sigma^{-1}\mu + C
$$
which ease the calculation.

#### Conditional
Partition $\bold{x}$ into two disjoint subsets $\bold{x}_a, \bold{x}_b$, $\bold{x} = \begin{pmatrix}
  \bold{x}_a\\
  \bold{x}_b\\
\end{pmatrix}$

Conditional distribution is also a Gaussian: $p(\bold{x}_a|\bold{x}_b) = N(\bold{x}_a|\mu_{a|b}, \Sigma_{a|b})$
$\mu_{a|b} = \mu_{a} + \Sigma_{ab}\Sigma_{bb}^{-1}(\bold{x}_b - \mu_b)$
$\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$

#### Marginal
Marginal distribution is also a Gaussian: $p(\bold{x}_a) = \int p(\bold{x}_a,\bold{x}_b)d\bold{x}_b = N(\bold{x}_a|\mu_a,\Sigma_{aa})$

#### Maximum Likelihood
Given $N$ observations $X = [\bold{x}_1,\bold{x}_2,...,\bold{x}_N]^T$ to estimate maximum likelihood of $\mu$ and $\Sigma$:
$\mu_{ML} = \frac{1}{N}\sum_{n = 1}^{N}\bold{x}_n$
$\Sigma_{ML} = \frac{1}{N}\sum_{n = 1}^{N}(\bold{x}_n - \mu_{ML})(\bold{x}_n - \mu_{ML})^T$

## **Information Theory**
### Entropy
$$
x \sim P(x)\\
H(X) = -\sum_{x} P(x)\log P(x) = E(- \log P(x))
$$
if $\log$ is $\log_2$ then unit is in bit.

### Cross Entropy
Random variables $x$ and $y$ have the same number of probable values.
$$
x \sim P(x), y \sim Q(y)\\
H(x, y) = -\sum P(x)\log Q(y) = E_{x \sim P}(- \log Q(y))
$$
Encode $x$ by $y$.

### KL Divergence
$$
x \sim P(x), y \sim Q(y)\\
D_{KL}(x||y) = -\sum_{x} P(x)\log \frac{Q(y)}{P(x)} = E_{x \sim P}(\log \frac{Q(y)}{P(x)}) = H(x, y) - H(x)
$$
Difference of distribution $x$ from $y$

