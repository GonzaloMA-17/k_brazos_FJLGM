"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax para el problema de los k-brazos.

Authors: Gonzalo Marcos Andres and Francisco José López Fernández
Email: gonzalo.marcosa@um.es and franciscojose.lopezf@um.es
Date: 2025/02/25

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from src_algorithms.algorithm import Algorithm
class Softmax(Algorithm):
    
    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo softmax.

        :param k: Número de brazos.
        :param tau: Parámetro de temperatura que controla el grado de exploración.
                    Valores bajos (tau cercano a 0) hacen que la selección sea casi greedy,
                    mientras que valores altos favorecen la exploración.
        :raises ValueError: Si tau no es mayor que 0.
        """
        if tau <= 0:
            raise ValueError("El parámetro tau debe ser mayor que 0.")
        
        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política softmax.
        
        La función calcula la probabilidad de seleccionar cada brazo utilizando la
        siguiente fórmula:
        
            P(a) = exp( Q(a) / tau ) / sum( exp( Q(b) / tau ) for b in todos los brazos )
        
        donde Q(a) es la recompensa estimada para el brazo a y tau es el parámetro de temperatura.

        :return: índice del brazo seleccionado.
        """
        
        "Numerador: exponencial de la estimacion de la recompensa de cada brazo dividida por tau"
        expon = np.exp(self.values / self.tau)

        "Denominador: sumatorio de la exponencial de la estimacion de la recompensa de cada brazo dividida por tau"
        sum_expon = np.sum(expon)
        
        "Probabilidad de seleccionar cada brazo calculada con la formula softmax"
        probab = expon / sum_expon 
        
        chosen_arm = np.random.choice(self.k, p=probab)
        return chosen_arm
