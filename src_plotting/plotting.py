from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src_algorithms import Algorithm, EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, 
                            optimal_selections: np.ndarray, 
                            algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Porcentaje de Selección del Brazo Óptimo', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

# def plot_arm_statistics(arm_stats: LoQueConsideres,
#                             algorithms: List[Algorithm], *args):
#     """
#     Genera gráficas separadas de Selección de Arms:
#     Ganancias vs Pérdidas para cada algoritmo.
#     - :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
#     - :param algorithms: Lista de instancias de algoritmos comparados.
#     - :param args: Opcional. Parámetros que consideres
#     """

def plot_arm_statistics_dep(arm_stats: List[dict], algorithms: List[Algorithm], optimal_arm: int):
    """
    Genera gráficas separadas de Selección de Arms:
    Ganancias vs Pérdidas para cada algoritmo.

    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
                      Cada diccionario debe contener 'average_rewards' y 'selection_counts'.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param optimal_arm: Índice del brazo óptimo.

    ADICIONAL:
    Para generar las estadísticas de cada brazo (arm_stats), puedes considerar varias métricas que te ayudarán a entender el rendimiento de cada brazo. Aquí hay algunos ejemplos de estadísticas que puedes usar:

    - Promedio de Ganancias (average_rewards): El promedio de las recompensas obtenidas por cada brazo.
    - Número de Selecciones (selection_counts): El número de veces que cada brazo fue seleccionado.
    - Varianza de las Ganancias (variance_rewards): La varianza de las recompensas obtenidas por cada brazo (opcional).
    - Recompensa Total (total_rewards): La suma de todas las recompensas obtenidas por cada brazo (opcional).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    num_algorithms = len(algorithms)
    fig, axes = plt.subplots(num_algorithms, 1, figsize=(14, 7 * num_algorithms), sharex=True)

    if num_algorithms == 1:
        axes = [axes]

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        stats = arm_stats[idx]
        average_rewards = stats['average_rewards']
        selection_counts = stats['selection_counts']

        bars = ax.bar(range(len(average_rewards)), average_rewards, tick_label=[f"{i}\n({count})" for i, count in enumerate(selection_counts)])
        for i, bar in enumerate(bars):
            if i == optimal_arm:
                bar.set_color('g')
            else:
                bar.set_color('b')

        ax.set_xlabel('Brazo (Número de Selecciones)', fontsize=14)
        ax.set_ylabel('Promedio de Ganancias', fontsize=14)
        ax.set_title(f'Estadísticas de Selección de Brazos para {get_algorithm_label(algo)}', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], optimal_arm: int):
    """
    Genera gráficas de estadísticas de cada brazo.

    :param arm_stats: Lista de diccionarios con estadísticas de cada brazo por algoritmo.
                      Cada diccionario debe contener 'average_rewards' y 'selection_counts'.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param optimal_arm: Índice del brazo óptimo.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    num_algorithms = len(algorithms)
    fig, axes = plt.subplots(num_algorithms, 1, figsize=(14, 7 * num_algorithms), sharex=True)

    if num_algorithms == 1:
        axes = [axes]

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        stats = arm_stats[idx]
        average_rewards = stats['average_rewards']
        selection_counts = stats['selection_counts']

        bars = ax.bar(range(len(average_rewards)), average_rewards, tick_label=[f"{i}\n({count})" for i, count in enumerate(selection_counts)])
        for i, bar in enumerate(bars):
            if i == optimal_arm:
                bar.set_color('g')
            else:
                bar.set_color('b')

        ax.set_xlabel('Brazo (Número de Selecciones)', fontsize=14)
        ax.set_ylabel('Promedio de Ganancias', fontsize=14)
        ax.set_title(f'Estadísticas de Selección de Brazos para {get_algorithm_label(algo)}', fontsize=16)

        # Añadir etiquetas de texto para clarificar el significado de la gráfica
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], expected_regret: np.ndarray = None):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param expected_regret: (Opcional) Arreglo con el arrepentimiento esperado para cada paso de tiempo.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    if expected_regret is not None:
        plt.plot(range(steps), expected_regret, label='Arrepentimiento Esperado', linestyle='--', color='r', linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Rechazo Acumulado', fontsize=14)
    plt.title('Rechazo Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def calculate_expected_regret(steps: int, constant: float) -> np.ndarray:
    """
    Calcula el arrepentimiento esperado utilizando la fórmula C * ln(T).

    :param steps: Número de pasos de tiempo.
    :param constant: Constante C utilizada en la fórmula.
    :return: Arreglo con el arrepentimiento esperado para cada paso de tiempo.
    """
    return constant * np.log(np.arange(1, steps + 1))