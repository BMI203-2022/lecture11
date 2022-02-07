#!/usr/bin/env python3
import numpy as np
from typing import Tuple

class NelderMead:
    def __init__(
            self, 
            f, 
            n_args=2, 
            seed=42, 
            initial_bounds=(-10, 10), 
            tolerance=1e-6,
            alpha=1,
            gamma=2,
            rho=0.5,
            sigma=0.5,
            max_iter=1e2):
        
        self.f = f
        self.n_args = n_args
        self.seed = seed
        self.initial_bounds = initial_bounds
        self.tolerance = tolerance
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.max_iter = int(max_iter)

        self.simplex_list = []
        self.score_list = []

        self._isfit = False

    def _initialize_simplex(self):
        """
        initialize the simplex as random vertices in some range
        """
        return np.random.uniform(
            low=self.initial_bounds[0],
            high=self.initial_bounds[1],
            size=(self.n_args, self.n_args+1)
        )

    def _sort_simplex(self):
        """
        sort simplex and scores 
        """
        mask = np.argsort(self.eval)
        self.simplex = self.simplex[:, mask]
        self.eval = self.eval[mask]

    def _terminate(self):
        """
        calculates the standard deviation of the calculated
        scores of the simplex vertices
        """
        self.simplex_list.append(self.simplex)
        self.score_list.append(self.eval)
        return np.std(self.eval) < self.tolerance

    def _centroid(self):
        """
        calculates the centroid on all points except the worst
        """
        self._sort_simplex()
        return np.mean(self.simplex[:,:-1], axis=0)

    def _reflection(self) -> Tuple[np.ndarray, float]:
        """
        generates and scores the reflection of the worst point
        across the centroid
        """
        xr = self.centroid + (self.alpha * (self.centroid - self.simplex[:,-1]))
        score = self.f(*xr)
        return (xr, score)

    def _expansion(self) -> Tuple[np.ndarray, float]:
        """
        generates and scores the expansion of the worst point
        across the centroid
        """
        xe = self.centroid + (self.gamma * (self.reflection - self.centroid))
        score = self.f(*xe)
        return (xe, score)

    def _contraction(self, outer=True) -> Tuple[np.ndarray, float]:
        """
        generates and scores the contraction of the worst point
        across the centroid
        """
        if outer:
            xc = self.centroid + (self.rho * (self.reflection - self.centroid))
        else:
            xc = self.centroid + (self.rho * (self.simplex[:,-1] - self.centroid))
        score = self.f(*xc)
        return (xc, score)

    def _shrink(self):
        """
        shrinks and replace all points on the simplex except the best point
        """
        for idx in np.arange(1, self.eval.size):
            point = self.simplex[:, idx]
            self.simplex[:, idx] = point + (self.sigma * (point - self.simplex[:, 0]))
            self.eval[idx] = self.f(*self.simplex[:, idx])

    def _update_simplex(self, point, score):
        """
        replaces the worst point and worse score with a new point and score
        """
        self.simplex[:,-1] = point
        self.eval[-1] = score


    def fit(self):
        self._isfit = False
        np.random.seed(self.seed)

        # initialize simplex
        self.simplex = self._initialize_simplex()
        self.eval = self.f(*self.simplex)
        num_iter = 0

        while True:
            if self._terminate():
                # print("Terminated!")
                break
            elif num_iter == self.max_iter:
                print("Maximum Iterations reached")
                break
        
            self.centroid = self._centroid()
            self.reflection, self.reflection_score = self._reflection()
            
            # expansion
            if self.reflection_score < self.eval[0]:
                expansion, expansion_score = self._expansion()
                if expansion_score < self.reflection_score:
                    # print("Accept Expansion")
                    self._update_simplex(expansion, expansion_score)
                else:
                    # print("Accept Reflection After Expansion")
                    self._update_simplex(self.reflection, self.reflection_score)
            
            # accept reflection
            elif self.reflection_score < self.eval[-2]:
                # print("Accept Reflection")
                self._update_simplex(self.reflection, self.reflection_score)

            # contraction
            else:
                xco, xco_score = self._contraction(outer=True)
                if xco_score < self.reflection_score:
                    # print("Accept XCO")
                    self._update_simplex(xco, xco_score)

                xci, xci_score = self._contraction(outer=False)
                if xci_score < self.reflection_score:
                    # print("Accept XCI")
                    self._update_simplex(xci, xci_score)

                self._shrink()

            num_iter += 1

        self._isfit = True

    def get_simplexes(self) -> np.ndarray:
        if not self._isfit:
            raise AttributeError("Must fit the optimization before calling")
        return np.array(self.simplex_list)

    def get_scores(self) -> np.ndarray:
        if not self._isfit:
            raise AttributeError("Must fit the optimization before calling")
        return np.array(self.score_list)

def f(x, y):
    return np.sqrt(x**2 + y**2)

def main():
    nm = NelderMead(f, tolerance=1e-6)
    nm.fit()
    simplexes = nm.get_simplexes()
    scores = nm.get_scores()


if __name__ == "__main__":
    main()
