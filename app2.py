# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from methodes_num_EDO import methodes_num_EDO  # Importez vos fonctions depuis le fichier externe

# Configurer Streamlit
st.set_page_config(page_title="Simulateur EDO", layout="wide")

# Titre de l'application
st.title("Simulateur de Résolution Numérique des EDOs")

# Description
st.markdown("""
Bienvenue dans le simulateur de résolution d'équations différentielles ordinaires (EDOs) !  
Choisissez les paramètres, la méthode numérique et visualisez les solutions.
""")

# Demander à l'utilisateur de saisir l'EDO
edo_input = st.text_area("Entrez l'EDO sous la forme f(t, y) (ex: -y * np.sin(t))", value="-y * np.sin(t)")

# Demander à l'utilisateur de saisir la solution exacte
solution_exacte_input = st.text_area("Entrez la solution exacte sous la forme y_exact(t) (si disponible, ex: np.exp(np.cos(t)))")

# Définir une fonction à partir de l'EDO donnée par l'utilisateur
try:
    f = eval(f"lambda t, y: {edo_input}")
except Exception as e:
    st.error(f"Erreur dans l'expression de l'EDO : {e}")
    f = None

# Définir la solution exacte si elle est donnée
if solution_exacte_input:
    try:
        y_exact = eval(f"lambda t: {solution_exacte_input}")
    except Exception as e:
        st.error(f"Erreur dans l'expression de la solution exacte : {e}")
        y_exact = None
else:
    y_exact = None  # Si l'utilisateur n'a pas fourni de solution exacte, utiliser None

# Entrée des paramètres pour la simulation
if f is not None:
    st.sidebar.header("Paramètres")
    T = st.sidebar.number_input("Temps de simulation (T)", min_value=10.0, max_value=1000.0, value=10.0, step=1.0)
    h = st.sidebar.number_input("Pas de temps (h)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    N = int(T / h)  # Nombre d'itérations basé sur T et h

    # Entrée des conditions initiales
    t0 = st.sidebar.number_input("Condition initiale t0", value=0.0)
    #y0 = st.sidebar.number_input("Condition initiale y0", value=1.0)
    
    
    # Permettre à l'utilisateur de saisir y0 sous forme d'expression mathématique
    y0_input = st.sidebar.text_input("Condition initiale y0", value="np.exp(1)")  # Expression mathématique par défaut

   # Convertir l'entrée de y0 en une valeur numérique
    try:
        y0 = eval(y0_input)  # Évaluer l'expression mathématique
    except Exception as e:
        st.error(f"Erreur dans l'expression de la condition initiale y0 : {e}")
        y0 = None

    # Sélection des méthodes
    methodes = ['Euler', 'Trapèze', 'RK4', 'AB3', 'Pred-Cor']
    nb_methodes = st.sidebar.number_input("Combien de méthodes comparer ?", min_value=1, max_value=5, value=2, step=1)
    methodes_choisies = [st.sidebar.selectbox(f"Méthode {i+1}", options=methodes) for i in range(nb_methodes)]

    # Résolution et affichage
    if st.sidebar.button("Simuler"):
        h_values = [0.25, 0.125, 0.0625, 0.03125, 0.015625]  # Différentes valeurs de h pour estimer l'ordre
        errors_all_methods = {}  # Dictionnaire pour stocker les erreurs pour chaque méthode

        # Créer des subplots avec un nombre de colonnes égal au nombre de méthodes sélectionnées
        fig, axes = plt.subplots(nrows=1, ncols=nb_methodes, figsize=(15, 5))  # Organise les subplots horizontalement
        if nb_methodes == 1:
            axes = [axes]  # Assurer que 'axes' soit toujours une liste

        # Créer des subplots pour les erreurs également
        fig_error, axes_error = plt.subplots(nrows=1, ncols=nb_methodes, figsize=(15, 5))  # Subplots pour erreurs
        if nb_methodes == 1:
            axes_error = [axes_error]  # Assurer que 'axes_error' soit toujours une liste

        # Pour chaque méthode sélectionnée, résoudre l'EDO et afficher les résultats
        for i, methode in enumerate(methodes_choisies):
            st.subheader(f"Résolution avec la méthode : {methode}")
            
            errors = []  # Liste des erreurs pour les différentes valeurs de h
            for h in h_values:
                N = int(T / h)
                t, y = methodes_num_EDO(methode, f, t0, y0, h, N)
                
                # Calcul de l'erreur maximale si une solution exacte est donnée
                if y_exact:
                    errors.append(np.max(np.abs(y_exact(t) - y)))
                else:
                    errors.append(np.nan)  # Pas d'erreur si la solution exacte n'est pas disponible
            
            errors_all_methods[methode] = errors  # Stocker les erreurs de cette méthode pour calculer l'ordre de convergence

            # Tracé de la solution numérique et exacte sur chaque subplot
            axes[i].plot(t, y, label=f"Solution numérique ({methode})", linewidth=2)
            if y_exact:  # Si une solution exacte est donnée, la tracer
                axes[i].plot(t, y_exact(t), '--', label="Solution exacte", color='k')
            axes[i].set_xlabel("Temps t")
            axes[i].set_ylabel("y(t)")
            axes[i].set_title(f"Comparaison pour {methode}")
            axes[i].legend()
            axes[i].grid()

            # Tracé des erreurs sur un autre subplot
            if y_exact:  # Afficher les erreurs uniquement si une solution exacte est donnée
                axes_error[i].plot(h_values, errors, '-o', label=f"Erreur {methode}", color='r')
            axes_error[i].set_xlabel("h")
            axes_error[i].set_ylabel("Erreur")
            axes_error[i].set_title(f"Erreurs pour {methode}")
            axes_error[i].legend()
            axes_error[i].grid()

        # Afficher les graphiques
        st.pyplot(fig)
        st.pyplot(fig_error)

        # Tracé des erreurs en échelle log-log et calcul de l'ordre de convergence
        fig_loglog, ax_loglog = plt.subplots(figsize=(10, 6))
        for i, methode in enumerate(methodes_choisies):
            errors = errors_all_methods[methode]
            # Tracé des erreurs pour chaque méthode
            ax_loglog.loglog(h_values, errors, '-o', label=f"Erreur {methode}")
            
            # Calcul de l'ordre de convergence basé sur les deux dernières erreurs
            if len(errors) > 1:
                p = np.log(errors[-2] / errors[-1]) / np.log(h_values[-2] / h_values[-1])
                st.write(f"Ordre de convergence pour la méthode {methode} : {round(p)}")
        
        ax_loglog.set_xlabel('log(h)')
        ax_loglog.set_ylabel('log(||e||_∞)')
        ax_loglog.set_title("Ordre de convergence des schémas")
        ax_loglog.legend()
        ax_loglog.grid(True, which="both", linestyle='--')
        st.pyplot(fig_loglog)
