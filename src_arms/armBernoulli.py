import numpy as np

from src_arms.arm import Arm

class ArmBernoulli(Arm):

    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli.

        :param p: Probabilidad de éxito del brazo.
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."

        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Bernoulli.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(1, self.p)
        return reward
    
    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.

        :return: Valor esperado de la distribución.
        """

        return self.p
    
    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"
    
    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la probabilidad.
        :param p_max: Valor máximo de la probabilidad.
        :return: Lista de brazos generados.
        """

        "Normas:"
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "El valor de p_min debe ser menor que p_max."

        # Generar k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = round(np.random.uniform(p_min, p_max), 2)
            p_values.add(p)

        # Crear brazos con las probabilidades generadas
        arms = [ArmBernoulli(p) for p in p_values]
        
        return arms