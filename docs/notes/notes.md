# Math Cheatsheet for ML

## Gradient

def $z = f(x,y)$ a function with two dimensional input $x,y$

$\nabla f = (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})$ formula of the function's gradient, **output**:
- is a **vector** in the input space.
- go to the **direction** where goes **upward** fastest.
- **magnitude** (length) is the gradient of this direction.

$\nabla f(x_1,y_1) = (\frac{\partial f(x_1,y_1)}{\partial x}, \frac{\partial f(x_1,y_1)}{\partial y})$ function's gradient at $(x_1,y_1)$




## Taylor Expansion
for a given $y = f(x)$

$f(x_1)-f(x_0) = f'(x_0)(x_1-x_0) + o(x_1-x_0)$

![taylor1](../assets/img/taylor1.png)

then we get:

$f(x_1)-f(x_0) = f'(x_0)(x_1-x_0) + \frac{f''(x_0)}{2!}(x_1-x_0)^2 + \frac{f^3(x_0)}{3!}(x_1-x_0)^3 + ... + \frac{f^n(x_0)}{n!}(x_1-x_0)^n + ...$

then we get:

$f(x_1) = f(x_0) + f'(x_0)(x_1-x_0) + \frac{f''(x_0)}{2!}(x_1-x_0)^2 + \frac{f^3(x_0)}{3!}(x_1-x_0)^3 + ... + \frac{f^n(x_0)}{n!}(x_1-x_0)^n + ...$

$n \to \infty$

or: 

$f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f^3(0)}{3!}x^3 + ... + \frac{f^n(0)}{n!}x^n + ...$

$n \to \infty$


## Lagrange Multiplier


