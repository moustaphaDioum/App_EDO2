# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from Methodes_num_EDOS import Methodes_num_EDOS
from methodes_num_EDO import methodes_num_EDO

# Configuration générale
st.set_page_config(page_title="Simulateur EDOs et Systèmes", layout="wide", page_icon="🧮")

# Titre principal
st.title("Simulateur de Résolution Numérique d'EDOs et de Systèmes d'EDOs")
st.markdown("---")

# Sélection du type de problème
choix = st.radio("Sélectionnez le type de problème à résoudre :", 
                 ("Une seule EDO", "Un système d'EDOs"), index=0)

if choix == "Une seule EDO":
    # Interface pour une EDO unique
    st.markdown("### 📘 Résolution d'une EDO simple")
    
    # Entrée de l'EDO
    edo_input = st.text_area("Entrez l'EDO sous la forme `f(t, y)` en langage python (ex: `-y * np.sin(t)`)", value="-y * np.sin(t)")

    try:
        f = eval(f"lambda t, y: {edo_input}")
        f(0, 1)  # Vérification rapide
    except Exception as e:
        st.error("Erreur : L'expression de l'EDO est invalide.")
        f = None

    # Entrée de la solution exacte
    solution_exacte_input = st.text_area("Entrez la solution exacte si connue en langage python (ex: `np.exp(np.cos(t))`)", value="np.exp(np.cos(t))")
    if solution_exacte_input:
        try:
            y_exact = eval(f"lambda t: {solution_exacte_input}")
            y_exact(0)  # Vérification rapide
        except Exception as e:
            st.error("Erreur : L'expression de la solution exacte est invalide.")
            y_exact = None
    else:
        y_exact = None

    # Paramètres dans la barre latérale
    if f:
        st.sidebar.header("🔧 Paramètres")
        T = st.sidebar.number_input("Temps de simulation (T)", min_value=10.0, max_value=1000.0, value=10.0)
        h = st.sidebar.number_input("Pas de temps (h)", min_value=0.01, max_value=1.0, value=0.1)
        t0 = st.sidebar.number_input("Temps initial t0", value=0.0)
        y0_input = st.sidebar.text_input("Condition initiale y0", value="np.exp(1)")

        try:
            y0 = eval(y0_input)
        except Exception as e:
            st.sidebar.error("Condition initiale invalide.")
            y0 = None

        # Sélection des méthodes numériques
        st.sidebar.markdown("### Méthodes numériques")
        methodes = ['Euler', 'Trapèze', 'RK4', 'AB3', 'Pred-Cor']
        nb_methodes = st.sidebar.number_input("Nombre de méthodes à comparer", min_value=1, max_value=5, value=1)
        methodes_choisies = [st.sidebar.selectbox(f"Méthode {i+1}", options=methodes) for i in range(nb_methodes)]

        # Bouton Simuler
        if y0 is not None and st.sidebar.button("Simuler"):
            try:
                h_values = [0.25, 0.125, 0.0625, 0.03125, 0.015625]  # Différents pas pour la convergence
                errors_all_methods = {}
                orders_all_methods = {}

                # Résolution et tracé pour chaque méthode
                for methode in methodes_choisies:
                    st.markdown(f"### Résultats pour la méthode : **{methode}**")
                    
                    # Résolution de l'EDO
                    t, y = methodes_num_EDO(methode, f, t0, y0, h, int(T/h))

                    # Graphique des solutions
                    fig_sol, ax_sol = plt.subplots(figsize=(10, 6))
                    ax_sol.plot(t, y, label="Solution numérique", linewidth=2)
                    if y_exact:
                        ax_sol.plot(t, y_exact(t), "--", label="Solution exacte", color="orange")
                    ax_sol.set_title(f"Solutions numériques avec la méthode {methode}")
                    ax_sol.set_xlabel("Temps \( t \)")
                    ax_sol.set_ylabel("Valeurs de \( y(t) \)")
                    ax_sol.legend()
                    ax_sol.grid()
                    st.pyplot(fig_sol)

                    # Calcul des erreurs pour les différents \( h \)
                    errors = []
                    for h_i in h_values:
                        N = int(T / h_i)
                        t_temp, y_temp = methodes_num_EDO(methode, f, t0, y0, h_i, N)
                        if y_exact:
                            errors.append(np.max(np.abs(y_exact(t_temp) - y_temp)))
                        else:
                            errors.append(np.nan)
                    errors_all_methods[methode] = errors

                    # Calcul de l'ordre de convergence
                    if len(errors) > 1:
                        orders = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(np.array(h_values[:-1]) / np.array(h_values[1:]))
                        orders_all_methods[methode] = orders[-1]
                    else:
                        orders_all_methods[methode] = np.nan

                    # Affichage de l'ordre de convergence
                    st.write(f"**Ordre de convergence pour {methode}** : {round(orders_all_methods[methode])}")

                    # Graphique des erreurs
                    fig_err, ax_err = plt.subplots(figsize=(10, 6))
                    ax_err.plot(h_values, errors, "-o", label="Erreur maximale")
                    ax_err.set_title(f"Erreurs maximales pour la méthode {methode}")
                    ax_err.set_xlabel("Pas \( h \)")
                    ax_err.set_ylabel("Erreur \( ||e||_\infty \)")
                    ax_err.legend()
                    ax_err.grid()
                    st.pyplot(fig_err)

                    # Graphique de convergence
                    fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
                    ax_conv.loglog(h_values, errors, "-o", label="Convergence log-log")
                    ax_conv.set_title(f"Convergence log-log pour la méthode {methode}")
                    ax_conv.set_xlabel("log(h)")
                    ax_conv.set_ylabel("log(||e||_∞)")
                    ax_conv.legend()
                    ax_conv.grid()
                    st.pyplot(fig_conv)

                    # Téléchargement des graphiques et données
                    def download_plot(fig, filename, label):
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.download_button(label=label, data=buf, file_name=filename, mime="image/png")
                    
                    def download_values(t, y, filename):
                        df = pd.DataFrame({"t": t, "y": y})
                        csv = df.to_csv(index=False)
                        st.download_button(label="Télécharger les valeurs numériques", data=csv, file_name=filename, mime="text/csv")

                    # Options de téléchargement
                    download_plot(fig_sol, f"solution_{methode}.png", f"Télécharger le graphique des solutions ({methode})")
                    download_plot(fig_err, f"erreurs_{methode}.png", f"Télécharger le graphique des erreurs ({methode})")
                    download_plot(fig_conv, f"convergence_{methode}.png", f"Télécharger le graphique de convergence ({methode})")
                    download_values(t, y, f"valeurs_solution_{methode}.csv")

            except Exception as e:
                st.error(f"Erreur lors de la simulation : {e}")












elif choix == "Un système d'EDOs":
    st.markdown("### 📗 Résolution d'un système d'EDOs")
    
    # Entrée du système
    st.markdown("#### Définissez votre système d'EDOs")
    nb_eqs = st.number_input("Nombre d'équations dans le système", min_value=1, max_value=10, value=2)
    edos = []
    for i in range(nb_eqs):
        eq = st.text_area(f"EDO {i+1}", placeholder=f"Exemple : -y[{i}] + t", value=f"-y[{i}] * np.sin(t)")
        edos.append(eq)

    try:
        def system(t, y):
            return [eval(eq) for eq in edos]
        system(0, [1.0] * nb_eqs)  # Vérification rapide
    except Exception as e:
        st.error("❌ Erreur : Une ou plusieurs EDOs sont invalides.")
        system = None

    if system:
        st.sidebar.header("🔧 Paramètres")
        T = st.sidebar.number_input("Temps de simulation (T)", min_value=1.0, max_value=1000.0, value=10.0)
        h = st.sidebar.number_input("Pas de temps (h)", min_value=0.01, max_value=1.0, value=0.1)
        y0 = [st.sidebar.number_input(f"Condition initiale y0[{i+1}]", value=1.0) for i in range(nb_eqs)]

        # Sélection des méthodes numériques
        methodes = ['Euler', 'Trapèze', 'RK4', 'AB3', 'Pred-Cor']
        nb_methodes = st.sidebar.number_input("Nombre de méthodes à comparer", min_value=1, max_value=5, value=1)
        methodes_choisies = [st.sidebar.selectbox(f"Méthode {i+1}", options=methodes) for i in range(nb_methodes)]

        # Lancer la simulation
        if st.sidebar.button("Simuler"):
            try:
                # Préparer les figures pour chaque méthode
                for methode in methodes_choisies:
                    st.markdown(f"### Résultats pour la méthode : **{methode}**")

                    # Résolution
                    t, y = Methodes_num_EDOS(methode, system, 0, y0, h, int(T / h))

                    # Tracé des solutions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i in range(nb_eqs):
                        ax.plot(t, y[:, i], label=f"y[{i+1}](t)", linewidth=2)
                    ax.set_title(f"Solutions numériques avec la méthode {methode}")
                    ax.set_xlabel("Temps \( t \)")
                    ax.set_ylabel("Valeurs de \( y(t) \)")
                    ax.legend()
                    ax.grid()
                    st.pyplot(fig)

                    # Téléchargement des graphiques
                    def download_plot(fig, filename, label):
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.download_button(label=label, data=buf, file_name=filename, mime="image/png")

                    download_plot(fig, f"solutions_{methode}.png", f"Télécharger le graphique des solutions ({methode})")

                    # Téléchargement des données numériques
                    df = pd.DataFrame(y, columns=[f"y[{i+1}](t)" for i in range(nb_eqs)], index=t)
                    csv = df.to_csv(index=True)
                    st.download_button(
                        label=f"Télécharger les données numériques ({methode})",
                        data=csv,
                        file_name=f"donnees_{methode}.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Erreur lors de la simulation : {e}")
