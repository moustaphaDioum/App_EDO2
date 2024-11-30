import numpy as np
import matplotlib.pyplot as plt
from methodes_num_EDO import methodes_num_EDO

# Définition du problème
f = lambda t, y: -y * np.sin(t)
y_exact = lambda t: np.exp(np.cos(t))  # Solution analytique
t0 = 0
y0 = np.exp(1)
T = 10
h_values = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
schemes = ['Euler', 'Trapèze', 'RK4', 'AB3', 'Pred-Cor']

# Calcul des erreurs pour chaque schéma et chaque valeur de h
plt.figure(figsize=(10, 6))
for scheme in schemes:
    errors = []  # Réinitialiser la liste des erreurs pour chaque schéma
    for h in h_values:
        N = int(T / h)
        t, y = methodes_num_EDO(scheme, f, t0, y0, h, N)
        # Calcul de l'erreur maximale
        errors.append(np.max(np.abs(y_exact(t) - y)))
    # Tracé des erreurs en échelle log-log
    plt.loglog(h_values, errors, '-o', label=scheme)

# Configuration du graphique
plt.xlabel('log(h)')
plt.ylabel('log(||e||_∞)')
plt.title('Ordre de convergence des schémas')
plt.legend()
plt.grid(True, which="both", linestyle='--')
plt.show()
