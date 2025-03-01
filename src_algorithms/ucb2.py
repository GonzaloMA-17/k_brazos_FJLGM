import math
import random
import numpy as np
from src_algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, k: int, alpha_param: float):
        """
        Inicializa el algoritmo UCB2.

        Parámetros:
        -----------
        k : int
            Número de brazos.
        alpha_param : float
            Parámetro de exploración que afecta la duración de las épocas (0 < alpha_param < 1).
        """
        assert 0 < alpha_param < 1, "El parámetro alpha_param debe estar en (0,1)."
        super().__init__(k)
        self.alpha_param = alpha_param
        # Contador de épocas para cada brazo (inicializado a 0 para todos)
        self.r = np.zeros(k, dtype=int)
        self.__current_arm = None      # Brazo que se está jugando actualmente
        self.__next_update = 0         # Instante en el que se cambiará el brazo actual

    def reset(self):
        """
        Restablece todas las variables a su estado inicial.
        """
        super().reset()
        self.r = np.zeros(self.k, dtype=int)
        self.__current_arm = None
        self.__next_update = 0

    def __tau(self, r_val: int) -> int:
        """
        Calcula la duración de la época para un contador de época r_val,
        usando la fórmula ceil((1 + alpha_param)^r_val).

        :param r_val: Número de épocas acumuladas para un brazo.
        :return: Duración de la época.
        """
        return int(math.ceil((1 + self.alpha_param) ** r_val))
    
    def __bonus(self, total_count: int, r_val: int) -> float:
        """
        Calcula el término de bonificación (Upper Confidence Bound) para el brazo,
        en función del total de jugadas y el contador de épocas r_val.

        :param total_count: Número total de jugadas.
        :param r_val: Número de épocas acumuladas para el brazo.
        :return: Valor del bonus.
        """
        tau_val = self.__tau(r_val)
        bonus = math.sqrt((1. + self.alpha_param) * math.log((math.e * total_count) / tau_val) / (2 * tau_val))
        return bonus

    def __set_arm(self, arm: int):
        """
        Asigna el brazo actual a 'arm' y actualiza el instante en el que se
        cambiará el brazo, según la diferencia entre épocas.

        :param arm: Índice del brazo a asignar.
        """
        self.__current_arm = arm
        # Actualiza el instante del próximo cambio sumando la diferencia entre épocas
        self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
        self.r[arm] += 1

    def select_arm(self) -> int:
        """
        Selecciona el brazo a jugar en función de la estrategia UCB2.

        :return: Índice del brazo seleccionado.
        """
        # Asegurarse de que cada brazo se juegue al menos una vez
        for arm in range(self.k):
            if self.counts[arm] == 0:
                self.__set_arm(arm)
                return arm

        total_counts = int(np.sum(self.counts))
        # Si aún no se ha completado la "época" del brazo actual, continuar con él
        if self.__next_update > total_counts:
            return self.__current_arm

        # Calcular el valor UCB2 para cada brazo
        ucb_values = np.zeros(self.k)
        for arm in range(self.k):
            bonus = self.__bonus(total_counts, self.r[arm])
            ucb_values[arm] = self.values[arm] + bonus

        # Escoger el brazo con el valor UCB2 máximo
        max_ucb = np.max(ucb_values)
        # En caso de empate, se elige uno al azar
        candidates = np.where(ucb_values == max_ucb)[0]
        chosen_arm = int(random.choice(candidates))
        self.__set_arm(chosen_arm)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las estadísticas tras jugar un brazo y observar la recompensa.

        :param chosen_arm: Índice del brazo jugado.
        :param reward: Recompensa obtenida.
        """
        # Se utiliza el método update de la clase base para actualizar counts y values
        super().update(chosen_arm, reward)


# """
# Module: src_algorithms/ucb2.py
# Description: Implementación del algoritmo upper confidence bound en su segunda version para el problema de los k-brazos.

# Authors: Gonzalo Marcos Andres and Francisco José López Fernández
# Email: gonzalo.marcosa@um.es and franciscojose.lopezf@um.es
# Date: 2025/02/25

# This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
# with the additional restriction that it may not be used for commercial purposes.

# For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
# """
# import numpy as np
# from src_algorithms.algorithm import Algorithm

# # class UCB2(Algorithm):
# #     def __init__(self, k: int, alpha: float = 0.1):
# #         """
# #         Inicializa el algoritmo UCB2.

# #         :param k: Número de brazos.
# #         :param alpha: Parámetro de ajuste para el balance entre exploración y explotación (0 < alpha < 1).
# #         """
# #         assert 0 < alpha < 1, "El parámetro alpha debe estar en (0,1)."

# #         super().__init__(k)
# #         self.alpha = alpha
# #         self.kas = np.zeros(k, dtype=int)  # Número de épocas para cada brazo
# #         self.taus = np.ones(k, dtype=int)  # Duración de cada época para cada brazo
# #         self.remaining_pulls = np.zeros(k, dtype=int)  # Veces que falta ejecutar cada brazo en la época

# #     # def tau(self, ka: int) -> int:
# #     def tau(self, ka: int) -> float:
# #         """
# #         Calcula τ(ka) según la fórmula ⌈(1 + α)^(ka)⌉.
# #         :param ka: Número de épocas del brazo.
# #         :return: Número de veces que el brazo será seleccionado en esta época.
# #         """
# #         return (1 + self.alpha) ** ka
# #         # return int(np.ceil((1 + self.alpha) ** ka))

# #     def select_arm(self, t: int) -> int:
# #         """
# #         Selecciona un brazo basado en la política UCB2.
# #         :param t: Instante de tiempo en el que nos encontramos.
# #         :return: Índice del brazo seleccionado.
# #         """
# #         # Si quedan repeticiones pendientes en una época, seguimos con el mismo brazo
# #         for i in range(self.k):
# #             if self.remaining_pulls[i] > 0:
# #                 self.remaining_pulls[i] -= 1
# #                 return i

# #         # Si no hay repeticiones pendientes, seleccionamos el brazo con el mayor índice UCB2
# #         ucbs = self.values + np.sqrt(((1 + self.alpha) * np.log(t + 1)) / (2 * self.taus))
# #         chosen_arm = np.argmax(ucbs)

# #         # Calculamos cuántas veces repetiremos la acción en la época actual
# #         next_tau = self.tau(self.kas[chosen_arm] + 1)
# #         self.remaining_pulls[chosen_arm] = next_tau - self.taus[chosen_arm]
# #         self.taus[chosen_arm] = next_tau

# #         return chosen_arm

# #     def update(self, chosen_arm: int, reward: float):
# #         """
# #         Actualiza los valores después de ejecutar una acción.
# #         :param chosen_arm: Brazo seleccionado.
# #         :param reward: Recompensa obtenida.
# #         """
# #         self.counts[chosen_arm] += 1
# #         self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

# #         # Cuando se termina una época, pasamos a la siguiente
# #         if self.remaining_pulls[chosen_arm] == 0:
# #             self.kas[chosen_arm] += 1

# #     def reset(self):
# #         """
# #         Reinicia el estado del algoritmo.
# #         """
# #         self.counts = np.zeros(self.k, dtype=int)
# #         self.values = np.zeros(self.k, dtype=float)
# #         self.kas = np.zeros(self.k, dtype=int)
# #         self.taus = np.ones(self.k, dtype=int)
# #         self.remaining_pulls = np.zeros(self.k, dtype=int)

# import math
# import numpy as np
# import random
# import numpy as np
# from src_algorithms.algorithm import Algorithm

# class UCB2(Algorithm):
#     def __init__(self, alpha_param, n_arms):
#         """
#         Inicializa el algoritmo UCB2.

#         Parámetros:
#         -----------
#         alpha_param : float
#             Parámetro de exploración que afecta la duración de las épocas.
#         n_arms : int
#             Número de brazos (acciones) disponibles.
#         """
#         self.alpha_param = alpha_param
#         self.n_arms = n_arms
#         self.counts = [0] * n_arms      # Número de veces que se ha jugado cada brazo
#         self.values = [0.0] * n_arms    # Recompensa media estimada para cada brazo
#         self.r = [0] * n_arms           # Contador de épocas para cada brazo
#         self.__current_arm = 0          # Brazo que se está jugando actualmente
#         self.__next_update = 0          # Cuándo se producirá el próximo cambio de brazo

#         # Parámetros para actualización tipo Beta (opcional, si las recompensas son binarias)
#         self.alpha = [1] * n_arms
#         self.beta = [1] * n_arms

#     def reset(self):
#         """
#         Restablece todas las variables a su estado inicial.
#         """
#         self.counts = [0] * self.n_arms
#         self.values = [0.0] * self.n_arms
#         self.r = [0] * self.n_arms
#         self.__current_arm = 0
#         self.__next_update = 0
#         self.alpha = [1] * self.n_arms
#         self.beta = [1] * self.n_arms
        
#     def __tau(self, r):
#         """
#         Calcula la longitud de la época para un contador de época r,
#         usando la fórmula (1 + alpha_param)^r redondeada hacia arriba.
#         """
#         return int(math.ceil((1 + self.alpha_param) ** r))
    
#     def __bonus(self, n, r):
#         """
#         Calcula el término de bonificación (Upper Confidence Bound)
#         para el brazo en su época r, dado el total de jugadas n.
#         """
#         tau = self.__tau(r)
#         bonus = math.sqrt((1. + self.alpha_param) * math.log(math.e * float(n) / tau) / (2 * tau))
#         return bonus
  
#     def __set_arm(self, arm):
#         """
#         Asigna el brazo actual a 'arm' y determina el número de jugadas
#         que se mantendrá ese brazo antes de volver a calcular índices.
#         """
#         self.__current_arm = arm
#         # Incrementa el contador de la siguiente actualización en la diferencia de épocas
#         self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
#         # Aumenta el contador de época para este brazo
#         self.r[arm] += 1

#     def select_arm(self):
#         """
#         Selecciona qué brazo jugar en la siguiente jugada siguiendo la estrategia UCB2.
#         """
#         # Asegurarse de que cada brazo se juegue al menos una vez
#         for arm in range(self.n_arms):
#             if self.counts[arm] == 0:
#                 self.__set_arm(arm)
#                 return arm
    
#         # Si aún no se ha terminado la "época" del brazo actual, seguir con él
#         if self.__next_update > sum(self.counts):
#             return self.__current_arm
    
#         # Calcular el valor UCB2 para cada brazo
#         ucb_values = [0.0 for _ in range(self.n_arms)]
#         total_counts = sum(self.counts)
#         for arm in range(self.n_arms):
#             bonus = self.__bonus(total_counts, self.r[arm])
#             ucb_values[arm] = self.values[arm] + bonus
        
#         # Escoger el brazo con el índice UCB2 más alto
#         max_ucb = max(ucb_values)
#         # Si hay empate, elige uno al azar entre los brazos con valor máximo
#         candidates = [i for i, val in enumerate(ucb_values) if val == max_ucb]
#         chosen_arm = random.choice(candidates)
        
#         # Actualizar variables internas para el brazo elegido
#         self.__set_arm(chosen_arm)
#         return chosen_arm

#     def update(self, chosen_arm, reward):
#         """
#         Actualiza las estadísticas tras jugar un brazo y observar la recompensa.
        
#         Parámetros:
#         -----------
#         chosen_arm : int
#             Índice del brazo que se jugó.
#         reward : float
#             Recompensa obtenida (0 o 1 si es binaria, o valor real en general).
#         """
#         # Incrementar el contador de jugadas para ese brazo
#         # self.counts[chosen_arm] += 1
        
#         # # Actualizar parámetros de distribución Beta (opcional)
#         # self.alpha[chosen_arm] += reward
#         # self.beta[chosen_arm] += (1 - reward)
        
#         # # Actualizar la recompensa media estimada (fórmula incremental)
#         # n = float(self.counts[chosen_arm])
#         # current_value = self.values[chosen_arm]
#         # new_value = ((n - 1) / n) * current_value + (1 / n) * reward
#         # self.values[chosen_arm] = new_value
#         super().update(chosen_arm, reward)  # Usa la actualización definida en Algorithm

