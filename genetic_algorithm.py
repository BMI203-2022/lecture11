#!/usr/bin/env python3

import numpy as np

def f(x, y):
    return np.sqrt(x**2 + y**2)

class GeneticAlgorithm:
    def __init__(
            self,
            f,
            n_args=2,
            population_size=10,
            elitism=True,
            initial_bounds=(-10, 10),
            tolerance=1e-3,
            seed=42,
            max_iter=1e2):

        self.f = f
        self.n_args = int(n_args)
        self.population_size = int(population_size)
        self.elitism = elitism
        self.initial_bounds = initial_bounds
        self.tolerance = tolerance
        self.seed = seed
        self.max_iter = max_iter

        self.best_list = []
        self.best_score = []

    def _initialize_population(self) -> np.ndarray:
        """
        initialize a population with random values
        """
        return np.random.uniform(
                low=self.initial_bounds[0],
                high=self.initial_bounds[1],
                size=(self.n_args, self.population_size))

    def _terminate(self) -> bool:
        """
        checks if ready to terminate
        """
        self.fitness.std() < self.tolerance
        

    def _sort(self):
        """
        sorts population based on fitness
        """
        mask = np.argsort(self.fitness)
        self.population = self.population[:, mask]
        self.fitness = self.fitness[mask]

        self.best_list.append(self.population[:, 0])
        self.best_score.append(self.fitness[0])

    def _selection(self) -> (np.ndarray, np.ndarray):
        """
        selects the best performing individuals in population
        """
        top = int(np.floor(self.population_size / 2))
        if top % 2 != 0:
            top += 1
        return self.population[:, :top], self.fitness[:top]

    def _pairing(self, size: int) -> np.ndarray:
        """
        pairs the selected individuals in the population
        in a rank sort manner (i.e. fittest with fittest)
        """
        mask = np.arange(size).reshape((int(size/2), 2))
        return mask

    def _mating(
            self, 
            selection: np.ndarray, 
            scores: np.ndarray, 
            pairs: np.ndarray) -> np.ndarray:
        """
        performs mating between the selected pairs
        with two children per pair
        """

        f1 = np.zeros((selection.shape[0], pairs.shape[0] * 2))
        idx = 0
        for mask in pairs:
            parents = selection[:, mask]
            for _ in np.arange(2):
                inheritance = np.random.choice(mask.size, mask.size)
                genes = parents[np.arange(mask.size), inheritance]
                f1[:, idx] = genes
                idx += 1
        return f1

    def _merge_populations(
                self,
                parents: np.ndarray,
                children: np.ndarray):
        """
        merges the populations into a single population
        """
        idx = 0
        for p in parents.T:
            self.population[:, idx] = p
            idx += 1
        for c in children.T:
            try: 
                self.population[:, idx] = c
            except IndexError:
                pass
            idx += 1

    def _mutate(self):
        """
        applies a mutation across the population by sampling from a gaussian
        """
        for idx, val in enumerate(self.population.T):
            if self.elitism and idx == 0:
                continue
            sample = np.random.normal(val, size=2)
            self.population[:, idx] = sample
            

    def fit(self):
        np.random.seed(self.seed)

        # initialize population
        self.population = self._initialize_population()

        num_iter = 0
        while True:

            self.fitness = self.f(*self.population)
            if self._terminate():
                break

            if num_iter == self.max_iter:
                break
            
            self._sort()
            selection, scores = self._selection()
            pairs = self._pairing(scores.size)
            f1 = self._mating(selection, scores, pairs)
            self._merge_populations(selection, f1)
            self._mutate()

            num_iter += 1

    def get_positions(self) -> np.ndarray:
        """
        return the positions of the best members
        """
        return np.array(self.best_list)

    def get_scores(self) -> np.ndarray:
        """
        return the scores of the best members
        """
        return np.array(self.best_score)


def main():
    ga = GeneticAlgorithm(f, population_size=1000, max_iter=1e4)
    ga.fit()


if __name__ == "__main__":
    main()
