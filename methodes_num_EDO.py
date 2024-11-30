# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.

"""

import numpy as np
from scipy.optimize import fsolve

def methodes_num_EDO(methode, f, t0, y0, h, N):
    """
    Fonction principale pour résoudre des EDOs avec différentes méthodes numériques.
    - methode : spécifie la méthode ('Euler', 'Trapèze', 'RK4', 'AB3', 'Pred-Cor')
    - f : fonction f(t, y)
    - t0, y0 : conditions initiales
    - h : pas de temps
    - N : nombre d'itérations
    """
    if methode == 'Euler':
        return euler_explicite(f, t0, y0, h, N)
    elif methode == 'Trapèze':
        return trapeze_implicite(f, t0, y0, h, N)
    elif methode == 'RK4':
        return rungekutta_4(f, t0, y0, h, N)
    elif methode == 'AB3':
        return AB_3(f, t0, y0, h, N)
    elif methode == 'Pred-Cor':
        return predcor_4(f, t0, y0, h, N)
    else:
        raise ValueError("Méthode inconnue. Choisissez parmi : Euler, Trapèze, RK4, AB3, Pred-Cor.")

# Méthode d'Euler explicite
def euler_explicite(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y

# Méthode du trapèze implicite
def trapeze_implicite(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    def g(yi, ti, yi_prev):
        return yi - yi_prev - 0.5 * h * (f(ti, yi_prev) + f(ti + h, yi))
    
    for i in range(N):
        y[i + 1] = fsolve(g, y[i], args=(t[i], y[i]))
    return t, y

# Méthode de Runge-Kutta d'ordre 4
def rungekutta_4(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y

# Méthode d'Adams-Bashforth d'ordre 3
def AB_3(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    if N >= 2:
        _, y_temp = rungekutta_4(f, t0, y0, h, 2)
        y[1:3] = y_temp[1:3]
    for i in range(2, N):
        y[i + 1] = y[i] + h * (23 * f(t[i], y[i]) - 16 * f(t[i - 1], y[i - 1]) + 5 * f(t[i - 2], y[i - 2])) / 12
    return t, y

# Méthode Prédicteur-Correcteur d'ordre 4
def predcor_4(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    if N >= 3:
        _, y_temp = rungekutta_4(f, t0, y0, h, 3)
        y[1:4] = y_temp[1:4]
    for i in range(3, N):
        y_pred = y[i] + h * (55 * f(t[i], y[i]) - 59 * f(t[i - 1], y[i - 1]) + 37 * f(t[i - 2], y[i - 2]) - 9 * f(t[i - 3], y[i - 3])) / 24
        y[i + 1] = y[i] + h * (9 * f(t[i + 1], y_pred) + 19 * f(t[i], y[i]) - 5 * f(t[i - 1], y[i - 1]) + f(t[i - 2], y[i - 2])) / 24
    return t, y
