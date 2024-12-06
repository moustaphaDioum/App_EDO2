import numpy as np
from scipy.optimize import fsolve

def Methodes_num_EDOS(methode, f, t0, y0, h, N):
    """
    Résout une EDO ou un système d'EDOs avec la méthode spécifiée.
    - methode : spécifie la méthode ('Euler', 'Trapèze', 'RK4', 'AB3', 'Pred-Cor').
    - f : fonction f(t, y), où y peut être un scalaire ou un vecteur.
    - t0, y0 : conditions initiales.
    - h : pas de temps.
    - N : nombre d'itérations.
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
    y = np.zeros((N + 1, len(y0))) if isinstance(y0, (list, np.ndarray)) else np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        y[i + 1] = y[i] + h * np.array(f(t[i], y[i]))
    return t, y

# Méthode du trapèze implicite
def trapeze_implicite(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros((N + 1, len(y0))) if isinstance(y0, (list, np.ndarray)) else np.zeros(N + 1)
    y[0] = y0

    def g(yi, ti, yi_prev):
        return yi - yi_prev - 0.5 * h * (np.array(f(ti, yi_prev)) + np.array(f(ti + h, yi)))

    for i in range(N):
        y[i + 1] = fsolve(g, y[i], args=(t[i], y[i]))
    return t, y

# Méthode de Runge-Kutta d'ordre 4
def rungekutta_4(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros((N + 1, len(y0))) if isinstance(y0, (list, np.ndarray)) else np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        k1 = np.array(f(t[i], y[i]))
        k2 = np.array(f(t[i] + h / 2, y[i] + h * k1 / 2))
        k3 = np.array(f(t[i] + h / 2, y[i] + h * k2 / 2))
        k4 = np.array(f(t[i] + h, y[i] + h * k3))
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y

# Méthode d'Adams-Bashforth d'ordre 3
def AB_3(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros((N + 1, len(y0))) if isinstance(y0, (list, np.ndarray)) else np.zeros(N + 1)
    y[0] = y0
    if N >= 2:
        _, y_temp = rungekutta_4(f, t0, y0, h, 2)
        y[1:3] = y_temp[1:3]
    for i in range(2, N):
        y[i + 1] = y[i] + h * (23 * np.array(f(t[i], y[i])) - 16 * np.array(f(t[i - 1], y[i - 1])) + 5 * np.array(f(t[i - 2], y[i - 2]))) / 12
    return t, y

# Méthode Prédicteur-Correcteur d'ordre 4
def predcor_4(f, t0, y0, h, N):
    t = np.linspace(t0, t0 + N * h, N + 1)
    y = np.zeros((N + 1, len(y0))) if isinstance(y0, (list, np.ndarray)) else np.zeros(N + 1)
    y[0] = y0
    if N >= 3:
        _, y_temp = rungekutta_4(f, t0, y0, h, 3)
        y[1:4] = y_temp[1:4]
    for i in range(3, N):
        y_pred = y[i] + h * (55 * np.array(f(t[i], y[i])) - 59 * np.array(f(t[i - 1], y[i - 1])) + 37 * np.array(f(t[i - 2], y[i - 2])) - 9 * np.array(f(t[i - 3], y[i - 3]))) / 24
        y[i + 1] = y[i] + h * (9 * np.array(f(t[i + 1], y_pred)) + 19 * np.array(f(t[i], y[i])) - 5 * np.array(f(t[i - 1], y[i - 1])) + np.array(f(t[i - 2], y[i - 2]))) / 24
    return t, y

