"""
Module: src_algorithms/ucb2.py
Description: Implementación del algoritmo upper confidence bound en su segunda version para el problema de los k-brazos.

Authors: Gonzalo Marcos Andres and Francisco José López Fernández
Email: gonzalo.marcosa@um.es and franciscojose.lopezf@um.es
Date: 2025/02/25

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""
import numpy as np
from src_algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para el balance entre exploración y explotación (0 < alpha < 1).
        """
        assert 0 < alpha < 1, "El parámetro alpha debe estar en (0,1)."

        super().__init__(k)
        self.alpha = alpha
        self.kas = np.zeros(k, dtype=int)  # Número de épocas para cada brazo
        self.taus = np.ones(k, dtype=int)  # Duración de cada época para cada brazo
        self.remaining_pulls = np.zeros(k, dtype=int)  # Veces que falta ejecutar cada brazo en la época

    # def tau(self, ka: int) -> int:
    def tau(self, ka: int) -> float:
        """
        Calcula τ(ka) según la fórmula ⌈(1 + α)^(ka)⌉.
        :param ka: Número de épocas del brazo.
        :return: Número de veces que el brazo será seleccionado en esta época.
        """
        return (1 + self.alpha) ** ka
        # return int(np.ceil((1 + self.alpha) ** ka))

    def select_arm(self, t: int) -> int:
        """
        Selecciona un brazo basado en la política UCB2.
        :param t: Instante de tiempo en el que nos encontramos.
        :return: Índice del brazo seleccionado.
        """
        # Si quedan repeticiones pendientes en una época, seguimos con el mismo brazo
        for i in range(self.k):
            if self.remaining_pulls[i] > 0:
                self.remaining_pulls[i] -= 1
                return i

        # Si no hay repeticiones pendientes, seleccionamos el brazo con el mayor índice UCB2
        ucbs = self.values + np.sqrt(((1 + self.alpha) * np.log(t + 1)) / (2 * self.taus))
        chosen_arm = np.argmax(ucbs)

        # Calculamos cuántas veces repetiremos la acción en la época actual
        next_tau = self.tau(self.kas[chosen_arm] + 1)
        self.remaining_pulls[chosen_arm] = next_tau - self.taus[chosen_arm]
        self.taus[chosen_arm] = next_tau

        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los valores después de ejecutar una acción.
        :param chosen_arm: Brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

        # Cuando se termina una época, pasamos a la siguiente
        if self.remaining_pulls[chosen_arm] == 0:
            self.kas[chosen_arm] += 1

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
        self.kas = np.zeros(self.k, dtype=int)
        self.taus = np.ones(self.k, dtype=int)
        self.remaining_pulls = np.zeros(self.k, dtype=int)
