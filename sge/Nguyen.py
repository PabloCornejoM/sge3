import random
from numpy import cos, sin
import sge
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import argparse
import numpy as np
from multiprocessing import Process, Queue
import time
from sklearn.metrics import mean_squared_error
import sys
#from problems.SR import SymbolicRegression

import warnings
warnings.filterwarnings('ignore')


class SymbolicRegression(SymbolicRegression):
    def __init__(self, num_exp, n = 0 , has_test_set=False, invalid_fitness=np.inf):
        super().__init__(invalid_fitness)
        self.num_exp = num_exp
        self.n = n
        self.readpolynomial()
    
    def get_equation(self, num_exp=None):
        if num_exp==None:
            num_exp = self.num_exp
        nguyen_eq = ['x[0]**3 + x[0]**2 + x[0]', 
        'x[0]**4 + x[0]**3 + x[0]**2 + x[0]', 
        'x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]',
        'x[0]**6 + x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]', 
        'sin(x[0]**2)*cos(x[0])-1', 'sin(x[0]) + sin(x[0] + x[0]**2)', '_log_(1 + x[0]) + _log_(1+ x[0]**2)',
        '_sqrt_(x[0])', 'sin(x[0]) + sin(x[1]**2)', '2*sin(x[0])*cos(x[1])', 'x[0]**x[1]', 'x[0]**4 - x[0]**3 + 0.5*(x[1]**2) - x[1]']
        return nguyen_eq[num_exp-1]


    def readpolynomial(self):
        self.equation = self.get_equation()
        n_datos = 20 + 2*self.n
        if self.num_exp<9:
            self.X_train = np.zeros((n_datos, 1))
            if self.num_exp==7:
                self.X_train[:,0] = np.linspace(0-self.n, 2+self.n, n_datos)
            elif self.num_exp==8:
                self.X_train[:,0] = np.linspace(0-self.n, 4+self.n, n_datos)
            else:
                self.X_train[:,0] = np.linspace(-1-self.n, 1+self.n, n_datos)   
        else:
            array_1 = np.linspace(0-self.n, 1+self.n, n_datos)
            array_2 = np.linspace(0-self.n, 1+self.n, n_datos)
            self.X_train = np.array(np.meshgrid(array_1, array_2)).T.reshape(-1, 2)
        
        self.Y_train = list(map(lambda x: eval(self.equation), self.X_train))

if __name__ == "__main__":
    import sge 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_exp', type=int, required=True)
    parser.add_argument('--n', type=int, default=0)
    args, _ = parser.parse_known_args(sys.argv[1:])
    eval_func = SymbolicRegression(**vars(args))
    sge.evolutionary_algorithm(evaluation_function=eval_func)