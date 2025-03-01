# Importación de módulos o clases
from .algorithm import Algorithm
from .epsilon_greedy import EpsilonGreedy
from .softMax import Softmax
from .gradientePreferencias import GradientPreference
from .ucb2 import UCB2
from .ucb1 import UCB1

# Lista de módulos o clases públicas
__all__ = ['Algorithm', 'EpsilonGreedy','Softmax', 'GradientPreference','UCB2', 'UCB1']