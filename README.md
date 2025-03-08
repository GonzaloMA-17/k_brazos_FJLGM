# Título del Trabajo 
## Información
- **Alumnos:** López, Francisco José; Marcos, Gonzalo; 
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** FJLGM

## Descripción 
Este trabajo de investigación consiste en el estudio e implementación de distintas familias de algoritmos para la resolución del problema *Multi-armed bandit*, con el fin de medir su rendimiento y capacidad de adaptación en diferentes escenarios.

## Estructura 
El repositorio está organizado de la siguiente manera:

```plaintext
|-- 📂 src_algorithms              # Carpeta que contiene los algoritmos desarrollados
|   |-- 📄 __init__.py             
|   |-- 📄 algorithm.py                
|   |-- 📄 epsilon-greedy.py   
|   |-- 📄 gradientePreferencias.py  
|   |-- 📄 softMax.py            
|   |-- 📄 ucb1.py                
|   |-- 📄 ucb2.py
|-- 📂 src_arms              # Carpeta que contiene los brazos de distintas distribuciones
|   |-- 📄 __init__.py             
|   |-- 📄 arm.py                
|   |-- 📄 armBernoulli.py                
|   |-- 📄 armBinomial.py                
|   |-- 📄 armNormal.py               
|   |-- 📄 bandit.py               
|-- 📂 src_plotting              # Carpeta que contiene las herramientas para visualización
|   |-- 📄 __init__.py             
|   |-- 📄 plotting.py
|--📄 main.ipynb # Notebook principal con introducción al problema
|--📄 notebook1.ipynb # Notebook con el primer experimento
|--📄... # Resto de notebooks con los demás experimentos
```

## Instalación y Uso 
Para instalar y utilizar este proyecto, sigue los siguientes pasos:
1. Clona el repositorio:
    ```bash
    git clone git clone https://github.com/GonzaloMA-17/k_brazos_FJLGM.git
    ```
2. Navega al directorio del proyecto:
    ```bash
    cd RL-Bandido
    ```
3. Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```
4. Ejecuta los scripts o notebooks según sea necesario.

## Tecnologías Utilizadas 
Este proyecto utiliza las siguientes tecnologías:
- **Lenguajes:** Python
- **Herramientas:** Jupyter Notebook, NumPy, Pandas, Matplotlib
- **Entornos:** VSCode (entornos virtuales locales), Google Colab.
