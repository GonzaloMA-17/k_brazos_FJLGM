import numpy as np
from src_algorithms.algorithm import Algorithm

class GradientPreference(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1, use_baseline: bool = True):
        """
        Inicializa el algoritmo de gradiente de preferencias.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias.
        :param use_baseline: Indica si se utiliza un baseline (recompensa promedio) en la actualización.
        :raises ValueError: Si alpha no es mayor que 0.
        """
        if alpha <= 0:
            raise ValueError("El parámetro alpha debe ser mayor que 0.")

        super().__init__(k)
        # Inicializa las preferencias a cero para todos los brazos
        self.preferences = np.zeros(k)
        self.alpha = alpha
        self.use_baseline = use_baseline
        # Inicializa el baseline (recompensa promedio) y el contador de pasos
        self.avg_reward = 0.0
        self.time = 0

    def select_arm(self) -> int:
        """
        Selecciona un brazo basándose en la distribución softmax de las preferencias.
        
        La probabilidad de seleccionar cada brazo se calcula mediante:
        
            P(a) = exp(H(a)) / sum(exp(H(b)) para todos los brazos b)
        
        :return: índice del brazo seleccionado.
        """
        exp_preferences = np.exp(self.preferences)
        self.probabilities = exp_preferences / np.sum(exp_preferences)
        chosen_arm = np.random.choice(self.k, p=self.probabilities)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Actualiza las preferencias de los brazos en función de la recompensa recibida.
        
        La fórmula de actualización es:
        
            H(a) <- H(a) + alpha * (reward - baseline) * (I(a == chosen_arm) - P(a))
        
        donde:
          - H(a) es la preferencia del brazo a.
          - alpha es la tasa de aprendizaje.
          - reward es la recompensa obtenida.
          - baseline es la recompensa promedio (si se usa).
          - I(a == chosen_arm) es 1 si a es el brazo seleccionado y 0 en caso contrario.
          - P(a) es la probabilidad de seleccionar el brazo a calculada en select_arm().
        
        :param chosen_arm: El índice del brazo que fue seleccionado.
        :param reward: La recompensa obtenida al seleccionar ese brazo.
        """
        self.time += 1

        # Actualiza el baseline (recompensa promedio) si se está usando
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.time
            baseline = self.avg_reward
        else:
            baseline = 0

        # Actualiza las preferencias para cada brazo
        for a in range(self.k):
            indicator = 1 if a == chosen_arm else 0
            self.preferences[a] += self.alpha * (reward - baseline) * (indicator - self.probabilities[a])