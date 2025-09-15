import numpy as np


class QuinticPolynomial:
    # 5th order polynomial
    # x0, v0, a0 as start point
    # x1, v1, a1 as end point
    def __init__(self, x0, v0, a0, x1, v1, a1, t):

        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        A = np.array(
            [
                [t**3, t**4, t**5],
                [3 * t**2, 4 * t**3, 5 * t**4],
                [6 * t, 12 * t**2, 20 * t**3],
            ]
        )
        b = np.array(
            [
                x1 - self.a0 - self.a1 * t - self.a2 * t**2,
                v1 - self.a1 - 2 * self.a2 * t,
                a1 - 2 * self.a2,
            ]
        )
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def interpolate(self, t):

        return (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )

    def interpolate_derivative(self, t):
        return (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t**2
            + 4 * self.a4 * t**3
            + 5 * self.a5 * t**4
        )

    def interpolate_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

    def interpolate_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2


class QuarticPolynomial:
    # 4th order polynomial
    def __init__(self, x0, v0, a0, v1, a1, time):
        # calc coefficient of quartic polynomial

        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        A = np.array([[3 * time**2, 4 * time**3], [6 * time, 12 * time**2]])
        b = np.array([v1 - self.a1 - 2 * self.a2 * time,  a1 - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def interpolate(self, t):
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

    def interpolate_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

    def interpolate_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

    def interpolate_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t
