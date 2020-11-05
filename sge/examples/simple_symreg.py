import numpy as np
import sge

class SimpleSymbolicRegression():
    def __init__(self, num_fitness_cases=20, invalid_fitness=9999999):
        self.invalid_fitness = invalid_fitness
        self.function = lambda x: (x + 1) * (x - 3)
        self.fitness_cases = num_fitness_cases
        self.x_points = np.asarray([x for x in range(self.fitness_cases)])
        self.y_points = np.asarray([self.function(x) for x in self.x_points])
        self.x_evals = np.empty(self.fitness_cases)

    def evaluate(self, individual):
        try:
            code = compile('result = ' + individual, 'solution', 'exec')
            globals = {}
            for i in range(self.fitness_cases):
                locals = {'x': self.x_points[i]}
                exec(code, globals, locals)
                self.x_evals[i] = locals['result']
            error = np.sum(np.sqrt(np.square(self.x_evals - self.y_points)))
        except (OverflowError, ValueError) as e:
            error = self.invalid_fitness
        return error, {'generation': 0, "evals": 1}


if __name__ == "__main__":
    fitness = SimpleSymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=fitness, parameters_file="parameters/standard.yml")
