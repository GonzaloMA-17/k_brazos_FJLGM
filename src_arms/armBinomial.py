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