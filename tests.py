import unittest
import math
from math import sqrt
import numpy as np


def ternary(f, l, r, eps=2e-8):
    intervals = []
    while (r - l) / 2 >= eps:
        intervals += [[l, r]]
        m = (l + r) / 2
        l1 = m - eps / 2
        r1 = m + eps / 2

        if f(l1) > f(r1):
            l = l1
        else:
            r = r1

    return (l + r) / 2, intervals


def golden(f, a, b, eps=2e-8):
    fi = (1 + 5 ** 0.5) / 2
    intervals = []

    d = (b - a) / fi
    x1 = b - d
    x2 = a + d

    fx1, fx2 = f(x1), f(x2)

    while (b - a) / 2 >= eps:
        intervals += [[a, b]]
        if fx1 > fx2:
            a, b = x1, b
            x1 = x2
            x2 = a + (b - a) / fi

            fx1 = fx2
            fx2 = f(x2)
        else:
            a, b = a, x2
            x2 = x1
            x1 = b - (b - a) / fi

            fx2 = fx1
            fx1 = f(x1)

    return (a + b) / 2, intervals


def fibonacci(f, a, b, n):
    def fib(n):
        return 1 / sqrt(5) * (((1 + sqrt(5)) / 2) ** n - ((1 - sqrt(5)) / 2) ** n)

    f_n_2, f_n_1, f_n = fib(n - 2), fib(n - 1), fib(n)

    x1 = a + (b - a) * f_n_2 / f_n
    x2 = a + (b - a) * f_n_1 / f_n

    fx1 = f(x1)
    fx2 = f(x2)

    intervals = []
    for i in range(1, n - 2):
        intervals += [[a, b]]

        if fx1 > fx2:
            a = x1
            x1 = x2

            x2 = a + fib(n - i - 1) / fib(n - i) * (b - a)

            fx1 = fx2
            fx2 = f(x2)
        else:
            b = x2
            x2 = x1

            x1 = a + fib(n - i - 2) / fib(n - i) * (b - a)

            fx2 = fx1
            fx1 = f(x1)

    return (x1 + x2) / 2, intervals


def fibonacci_eps(f, a, b, eps=2e-8):
    fib0, fib1 = 0, 1
    n = 1
    while (b - a) / fib1 > eps:
        fib0, fib1 = fib1, fib0 + fib1
        n += 1

    return fibonacci(f, a, b, n)


def ternary_searcher(f, lr, i):
    l, r = lr
    return ternary(f, l, r)[0]


def grad_descent(
        f, df, start,
        step_searcher=ternary_searcher,
        eps=1e-6,
        max_iters=100
):
    x = start
    prev = np.zeros_like(x)

    points = [np.append(start, f(start))]

    while abs(f(x) - f(prev)) > eps:
        if max_iters is not None and len(points) > max_iters:
            break

        dfx = df(x)
        dfx = dfx / np.sqrt(np.sum(dfx ** 2))
        step = step_searcher(lambda s: f(x - s * dfx), [0, 1], len(points))
        prev = x
        x = x - step * dfx
        points.append(np.append(x, f(x)))

    return np.array(points)


class Test(unittest.TestCase):
    eps = 2e-8

    def test_bp(self):
        self.assertAlmostEqual(ternary(math.sin, math.pi, 2 * math.pi, eps=self.eps)[0], 1.5 * math.pi, delta=self.eps)
        self.assertAlmostEqual(ternary(lambda x: x ** 2, -123, 12.3, eps=self.eps)[0], 0, delta=self.eps)

    def test_golden(self):
        self.assertAlmostEqual(golden(math.sin, math.pi, 2 * math.pi, eps=self.eps)[0], 1.5 * math.pi, delta=self.eps)
        self.assertAlmostEqual(golden(lambda x: x ** 2, -123, 12.3, eps=self.eps)[0], 0, delta=self.eps)

    def test_fibb(self):
        self.assertAlmostEqual(fibonacci_eps(math.sin, math.pi, 2 * math.pi, eps=self.eps)[0], 1.5 * math.pi,
                               delta=self.eps)
        self.assertAlmostEqual(fibonacci_eps(lambda x: x ** 2, -123, 12.3, eps=self.eps)[0], 0, delta=self.eps)

    def test_grad_descent(self):
        self.assertAlmostEqual(grad_descent(lambda x: x ** 2, lambda x: 2 * x, 7.7)[-1, 0], 0)

        def f(xy: np.array):
            x, y = xy
            return np.sin(x) * np.cos(y)

        def df(xy: np.array):
            x, y = xy
            return np.array([math.cos(x) * math.cos(y), - math.sin(x) * math.sin(y)])

        self.assertAlmostEqual(grad_descent(f, df, [3, -3])[-1, 0], math.pi / 2)


if __name__ == '__main__':
    unittest.main()
