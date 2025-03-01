"""
Module: src_algorithms/ucb1.py
Description: Implementación del algoritmo upper confidence bound en su primera version para el problema de los k-brazos.

Authors: Gonzalo Marcos Andres and Francisco José López Fernández
Email: gonzalo.marcosa@um.es and franciscojose.lopezf@um.es
Date: 2025/02/25

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""
import numpy as np
from src_algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de ajuste de exploración (por defecto c = 1).
        """
        assert c > 0, "El parámetro c debe ser mayor que 0."
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        
        :return: Índice del brazo seleccionado.
        """
        # Si algún brazo no ha sido seleccionado, lo seleccionamos primero (exploración inicial)
        if 0 in self.counts:
            return int(np.argmin(self.counts))

        # Calcular t como el número total de iteraciones
        t = np.sum(self.counts)
        # Calculamos UCB1 para cada brazo
        ucb_values = self.values + self.c * np.sqrt((2 * np.log(t)) / self.counts)
        # Seleccionamos el brazo con el mayor valor de UCB1
        return int(np.argmax(ucb_values))

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
