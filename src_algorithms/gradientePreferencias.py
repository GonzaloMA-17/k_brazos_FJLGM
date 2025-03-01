import numpy as np
from src_algorithms.algorithm import Algorithm

class GradientPreference(Algorithm):

    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo de gradiente de preferencias.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias (H).

        """
        assert 0 < alpha, "El parámetro alpha debe ser mayor que 0."
        
        super().__init__(k)
        
        self.preferences = np.zeros(k) # Inicializa las preferencias Ht(a)
        self.alpha = alpha # Tasa de aprendizaje
        self.probabilities = np.ones(k) / k # Probabilidad de seleccionar cada brazo
        """Las probabilidades deben comenzar con una distribución uniforme."""

    def select_arm(self) -> int:
        """
        Selecciona un brazo basándose en la distribución softmax de las preferencias.
        
        La probabilidad de seleccionar cada brazo se calcula mediante:
        
            P(a) = exp(H(a)) / sum(exp(H(b)) para todos los brazos b)
        
        :return: índice del brazo seleccionado.
        """
        exp_preferences = np.exp(self.preferences)
        self.probabilities = exp_preferences / np.sum(exp_preferences)  # Cálculo de πt(a)
        
        return np.random.choice(self.k, p=self.probabilities) # Devuelve el brazo basado en πt(a)

    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Actualiza las preferencias de los brazos en función de la recompensa recibida.
        
        La fórmula de actualización es:
        
            H(a) <- H(a) + alpha * (reward - recompensa promedio) * (I(a == chosen_arm) - P(a))
        
        donde:
          - H(a) es la preferencia del brazo a.
          - alpha es la tasa de aprendizaje.
          - reward es la recompensa obtenida.
          - recompensa promedio.
          - I(a == chosen_arm) es 1 si a es el brazo seleccionado y 0 en caso contrario.
          - P(a) es la probabilidad de seleccionar el brazo a calculada en select_arm().
        
        :param chosen_arm: El índice del brazo que fue seleccionado.
        :param reward: La recompensa obtenida al seleccionar ese brazo.

        """
        average_reward = np.mean(self.values)  # R̄t (recompensa promedio estimada)

        # Actualización de las preferencias usando el Gradiente de Preferencias
        for a in range(self.k):
            if a == chosen_arm:
                self.preferences[a] += self.alpha * (reward - average_reward) * (1 - self.probabilities[a])
            else:
                self.preferences[a] -= self.alpha * (reward - average_reward) * self.probabilities[a]

        """En Gradiente de Preferencias, no debemos actualizar las recompensas de la forma estándar, 
        ya que el algoritmo solo trabaja con preferencias. Elimina la línea de super().update(chosen_arm, reward)."""

    def reset(self):
        """
        Reinicia el estado del algoritmo, incluidos los parámetros Ht(a) y las probabilidades.
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
        self.preferences = np.zeros(self.k)
        self.probabilities = np.ones(self.k) / self.k
