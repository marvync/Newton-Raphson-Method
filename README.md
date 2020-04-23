# Newton-Raphson Method

Newton-Raphson method implementation in Python using SymPy. This Python library for symbolic mathematics allows us to derived whatever may be the function passed into the method.



```python
from sympy import symbols, diff, lambdify
```


```python
x = symbols('x')
```


```python
def NewRaph(f, x0, ϵ=1e-3, Nmax=100):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    x0 : number
        Initial guess for a solution f(x)=0.
    ϵ : number
        Stopping criteria is abs(f(x)) < epsilon.
    Nmax : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
    
        Implement Newton's method: compute the linear approximation of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None.
        If the number of iterations exceeds Nmax, then return None.

    Examples
    --------
    f = x**3 - x**2 - 1
    NewRaph(f, x0=1)
    Found solution after 6 iterations.
    1.4655713749070918
    '''
    xn = x0
    df = diff(f,x)
    for n in range(Nmax):
        fxn = lambdify(x, f, 'math')(xn)
        if abs(fxn) < ϵ:
            print('Found solution after',n,'iterations.')
            return xn
        dfxn = lambdify(x, df, 'math')(xn)
        if dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None
```

```python
f = x**3 - x**2 - 1
NewRaph(f, x0=1)
```
    Found solution after 5 iterations.
    
    1.4655713749070918
