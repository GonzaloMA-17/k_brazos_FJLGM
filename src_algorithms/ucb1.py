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

class UCB1_v1(Algorithm):
    
    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.
        
        :param k: Número de brazos.
        :param c: Parámetro que controla el equilibrio entre exploración y explotación.
                  Valores más altos aumentan la exploración.
        """
        assert c >= 0, "El parámetro c debe ser mayor o igual a 0."
        
        super().__init__(k)
        self.c = c
        self.total_steps = 0  # Contador global de pasos para logaritmo en UCB1

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1:
        
            Q(a) + c * sqrt( ln(t) / N(a) )
        
        donde:
        - Q(a) es la recompensa promedio estimada del brazo a.
        - c es un parámetro que controla la exploración.
        - t es el número total de selecciones realizadas.
        - N(a) es el número de veces que el brazo a ha sido seleccionado.
        
        Si un brazo no ha sido seleccionado, se prioriza su selección.
        
        :return: Índice del brazo seleccionado.
        """
        self.total_steps += 1  # Aumentar el contador de pasos

        # Para evitar divisiones por cero, seleccionamos primero los brazos no explorados
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        # Calcular el valor UCB1 para cada brazo
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total_steps) / self.counts)

        # Seleccionar el brazo con el valor UCB más alto
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Actualiza las recompensas promedio y el conteo de selecciones.
        
        Se usa el método incremental de actualización:
        
            Q(a) <- Q(a) + (reward - Q(a)) / N(a)
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        super().update(chosen_arm, reward)  # Usa la actualización definida en Algorithm

    def reset(self):
        """
        Reinicia el estado del algoritmo UCB1.
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
        self.total_steps = 0

class UCB1_v2(Algorithm):

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de ajuste de exploración (por defecto c = 1).
        """
        assert c > 0, "El parámetro c debe ser mayor que 0."

        super().__init__(k)
        self.c = c

    def select_arm(self, t: int) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        
        :param t: Número total de iteraciones (t ≥ 1).
        :return: Índice del brazo seleccionado.
        """
        # Si algún brazo no ha sido seleccionado, lo seleccionamos primero (exploración inicial)
        if 0 in self.counts:
            return np.argmin(self.counts)

        # Calculamos UCB1 para cada brazo
        ucb_values = self.values + self.c * np.sqrt((2 * np.log(t)) / self.counts)

        # Seleccionamos el brazo con el mayor valor de UCB1
        return np.argmax(ucb_values)

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)