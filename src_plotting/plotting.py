"""
Module: scr_plotting/plotting.py
Description: Desarrollo de gráficas y estadísticas para el problema de los k-brazos.

Authors: Gonzalo Marcos Andres and Francisco José López Fernández
Email: gonzalo.marcosa@um.es and franciscojose.lopezf@um.es
Date: 2025/02/25

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src_algorithms import Algorithm, EpsilonGreedy, Softmax, GradientPreference, UCB2, UCB1


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
    elif isinstance(algo, Softmax):
        label += f" (tau={algo.tau})"
    elif isinstance(algo, GradientPreference):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, UCB2):
        label += f" (alfa={algo.alpha_param})"
    elif isinstance(algo, UCB1):
        label += f" (parametroAjusteExploracion={algo.c})"



    # # elif isinstance(algo, OtroAlgoritmo):
    # #     label += f" (parametro={algo.parametro})"
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
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

# def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], optimal_arm: int):
#     """
#     Genera gráficas de estadísticas de cada brazo.

#     :param arm_stats: Lista de diccionarios con estadísticas de cada brazo por algoritmo.
#                       Cada diccionario debe contener 'average_rewards' y 'selection_counts'.
#     :param algorithms: Lista de instancias de algoritmos comparados.
#     :param optimal_arm: Índice del brazo óptimo.
#     """
#     sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

#     num_algorithms = len(algorithms)
#     fig, axes = plt.subplots(num_algorithms, 1, figsize=(10, 5 * num_algorithms), sharex=True)

#     if num_algorithms == 1:
#         axes = [axes]

#     for idx, algo in enumerate(algorithms):
#         ax = axes[idx]
#         stats = arm_stats[idx]
#         average_rewards = stats['average_rewards']
#         selection_counts = stats['selection_counts']

#         bars = ax.bar(range(len(average_rewards)), average_rewards, tick_label=[f"{i}\n({count})" for i, count in enumerate(selection_counts)])
#         for i, bar in enumerate(bars):
#             if i == optimal_arm:
#                 bar.set_color('g')
#             else:
#                 bar.set_color('b')

#         ax.set_xlabel('Brazo (Número de Selecciones)', fontsize=14)
#         ax.set_ylabel('Promedio de Ganancias', fontsize=14)
#         ax.set_title(f'Estadísticas de Selección de Brazos para {get_algorithm_label(algo)}', fontsize=16)

#         # Añadir etiquetas de texto para clarificar el significado de la gráfica
#         for i, bar in enumerate(bars):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

#     plt.tight_layout()
#     plt.show()

def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], optimal_arm: int):
    """
    Genera gráficas de estadísticas de cada brazo.

    :param arm_stats: Lista de diccionarios con estadísticas de cada brazo por algoritmo.
                      Cada diccionario debe contener 'average_rewards' y 'selection_counts'.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param optimal_arm: Índice del brazo óptimo (0-based).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    num_algorithms = len(algorithms)
    num_rows = (num_algorithms + 1) // 2

    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for idx, algo in enumerate(algorithms):
        stats = arm_stats[idx]
        average_rewards = stats['average_rewards']
        selection_counts = stats['selection_counts']

        # Ajuste para mostrar los brazos empezando en 1 (i+1)
        bars = axes[idx].bar(
            range(len(average_rewards)), 
            average_rewards,
            tick_label=[f"{i+1}\n({count})" for i, count in enumerate(selection_counts)]
        )

        # Colorear de verde el brazo óptimo (si optimal_arm es 0-based, comparar con i directamente)
        for i, bar in enumerate(bars):
            if i == optimal_arm:
                bar.set_color('g')
            else:
                bar.set_color('b')

        axes[idx].set_xlabel('Brazo (Número de Selecciones)', fontsize=14)
        axes[idx].set_ylabel('Promedio de Ganancias', fontsize=14)
        axes[idx].set_title(f'Estadísticas de Selección de Brazos para {get_algorithm_label(algo)}', fontsize=16)

        # Mostrar la ganancia promedio encima de cada barra
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

    # Ocultar subplots extra si no se utilizan
    for j in range(len(algorithms), len(axes)):
        fig.delaxes(axes[j])

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

# def plot_arm_statistics_three(arm_stats: List[dict], algorithms: List, optimal_arm: int):
#     """
#     Genera gráficas de estadísticas de cada brazo, organizadas en una sola fila.

#     :param arm_stats: Lista de diccionarios con estadísticas de cada brazo por algoritmo.
#                       Cada diccionario debe contener 'average_rewards' y 'selection_counts'.
#     :param algorithms: Lista de instancias o nombres de los algoritmos comparados.
#     :param optimal_arm: Índice del brazo óptimo.
#     """
#     sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

#     num_algorithms = len(algorithms)
#     fig, axes = plt.subplots(1, num_algorithms, figsize=(20 * num_algorithms, 15), sharey=True)

#     # Si solo hay un algoritmo, convertimos axes en lista para iterar igual que en el bucle
#     if num_algorithms == 1:
#         axes = [axes]

#     for idx, algo in enumerate(algorithms):
#         ax = axes[idx]
#         stats = arm_stats[idx]
#         average_rewards = stats['average_rewards']
#         selection_counts = stats['selection_counts']

#         bars = ax.bar(
#             range(len(average_rewards)),
#             average_rewards,
#             tick_label=[f"{i}\n({count})" for i, count in enumerate(selection_counts)]
#         )
#         for i, bar in enumerate(bars):
#             if i == optimal_arm:
#                 bar.set_color('g')
#             else:
#                 bar.set_color('b')

#         ax.set_xlabel('Brazo (Número de Selecciones)', fontsize=14)
#         ax.set_ylabel('Promedio de Ganancias', fontsize=14)
#         ax.set_title(f'Estadísticas para {get_algorithm_label(algo)}', fontsize=16)

#         # Añadir etiquetas de texto sobre cada barra
#         for bar in bars:
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}',
#                     ha='center', va='bottom', fontsize=16)

#     plt.tight_layout()
#     plt.show()

