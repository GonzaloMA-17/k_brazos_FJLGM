import numpy as np

from src_arms.arm import Arm

class ArmBinomial(Arm):

    def __init__(self, n:int, p:float):
        """
        Inicializa el brazo con distribución Binomial.

        :param n: Número de ensayos.
        :param p: Probabilidad de éxito del brazo.
        """
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."

        self.n = n
        self.p = p

    def pull(self):

        """
        Genera una recompensa siguiendo una distribución Binomial.
        
        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward 
    
    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.

        El cáculo de la esperanza de la binomial es n * p.

        :return: Valor esperado de la distribución.
        """
        return self.n * self.p
    
    def __str__(self):
        """
        Representación en cadena del brazo Binomial.

        :return: Descripción detallada del brazo Binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"
    
    @classmethod
    def generate_arms(cls, k: int, n_min: int = 2, n_max: int = 20, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con parámetros únicos dentro de los rangos especificados.

        :param k: Número de brazos a generar.
        :param n_min: Valor mínimo de n (ensayos).
        :param n_max: Valor máximo de n (ensayos).
        :param p_min: Valor mínimo de la probabilidad de éxito.
        :param p_max: Valor máximo de la probabilidad de éxito.
        :return: Lista de brazos generados.
        """
        "Normas:"
        
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n_min < n_max, "n_min debe ser menor que n_max."
        assert 0 <= p_min < p_max <= 1, "p_min y p_max deben estar en el rango [0, 1] con p_min < p_max."

        p_values = set()
        n_values = set()

        while len(p_values) < k:
            p = round(np.random.uniform(p_min, p_max), 2)
            n = np.random.randint(n_min, n_max + 1)
            p_values.add(p)
            n_values.add(n)

        arms = [ArmBinomial(n, p) for n, p in zip(n_values, p_values)]

        # arms = []
        # for _ in range(k):
        #     n = np.random.randint(n_min, n_max + 1)  # Selecciona un número de ensayos aleatorio
        #     p = round(np.random.uniform(p_min, p_max), 2)  # Selecciona una probabilidad aleatoria
        #     arms.append(ArmBinomial(n, p))

        return arms