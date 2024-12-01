import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
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

# **Validation de l'entrée EDO**
edo_input = st.text_area("Entrez l'EDO sous la forme f(t, y) (ex: -y * np.sin(t))", value="-y * np.sin(t)")

try:
    f = eval(f"lambda t, y: {edo_input}")
    f(0, 1)  # Vérification rapide
except Exception as e:
    st.error("Erreur : L'expression de l'EDO est invalide. Assurez-vous d'utiliser une syntaxe Python correcte, ex. : `-y * np.sin(t)`.")
    f = None

# **Validation de l'entrée de la solution exacte**
solution_exacte_input = st.text_area("Entrez la solution exacte ",value = "np.exp(np.cos(t))")

if solution_exacte_input:
    try:
        y_exact = eval(f"lambda t: {solution_exacte_input}")
        y_exact(0)  # Vérification rapide
    except Exception as e:
        st.error("Erreur : L'expression de la solution exacte est invalide. Assurez-vous d'utiliser une syntaxe Python correcte.")
        y_exact = None
else:
    y_exact = None  # Si aucune solution exacte n'est fournie

# Paramètres pour la simulation (si l'EDO est valide)
if f is not None:
    st.sidebar.header("Paramètres")
    T = st.sidebar.number_input("Temps de simulation (T)", min_value=10.0, max_value=1000.0, value=10.0, step=1.0)
    h = st.sidebar.number_input("Pas de temps (h)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    N = int(T / h)

    t0 = st.sidebar.number_input("Condition initiale t0", value=0.0)

    # Saisie et validation de y0
    y0_input = st.sidebar.text_input("Condition initiale y0", value="np.exp(1)")  # Expression par défaut
    try:
        y0 = eval(y0_input)
    except Exception as e:
        st.error("Erreur : L'expression de la condition initiale y0 est invalide. Assurez-vous d'utiliser une valeur numérique ou une expression valide.")
        y0 = None

    # Sélection des méthodes numériques
    methodes = ['Euler', 'Trapèze', 'RK4', 'AB3', 'Pred-Cor']
    nb_methodes = st.sidebar.number_input("Combien de méthodes à comparer ?", min_value=1, max_value=5, value=1, step=1)
    methodes_choisies = [st.sidebar.selectbox(f"Méthode {i+1}", options=methodes) for i in range(nb_methodes)]

    # Résolution et affichage des résultats
    if y0 is not None and st.sidebar.button("Simuler"):
        h_values = [0.25, 0.125, 0.0625, 0.03125, 0.015625]  # Différentes valeurs de h pour l'ordre de convergence
        errors_all_methods = {}  # Pour stocker les erreurs de chaque méthode

        # Préparation des subplots
        fig, axes = plt.subplots(nrows=1, ncols=nb_methodes, figsize=(15, 5))
        if nb_methodes == 1:
            axes = [axes]

        if y_exact:
            fig_error, axes_error = plt.subplots(nrows=1, ncols=nb_methodes, figsize=(15, 5))
            if nb_methodes == 1:
                axes_error = [axes_error]

        # Résolution et tracé pour chaque méthode
        for i, methode in enumerate(methodes_choisies):
            st.subheader(f"Résolution avec la méthode : {methode}")

            errors = []
            for h in h_values:
                N = int(T / h)
                try:
                    t, y = methodes_num_EDO(methode, f, t0, y0, h, N)
                    if y_exact:
                        errors.append(np.max(np.abs(y_exact(t) - y)))
                    else:
                        errors.append(np.nan)
                except Exception as e:
                    st.error(f"Erreur lors de la simulation avec la méthode {methode} : {e}")
                    continue

            errors_all_methods[methode] = errors

            # Tracé des solutions numériques
            axes[i].plot(t, y, label=f"Solution numérique ({methode})", linewidth=2)
            if y_exact:
                axes[i].plot(t, y_exact(t), '--', label="Solution exacte", color='k')
            axes[i].set_xlabel("Temps t")
            axes[i].set_ylabel("y(t)")
            axes[i].set_title(f"Solutions pour {methode}")
            axes[i].legend()
            axes[i].grid()

            # Tracé des erreurs uniquement si la solution exacte est fournie
            if y_exact:
                axes_error[i].plot(h_values, errors, '-o', label=f"Erreur {methode}", color='r')
                axes_error[i].set_xlabel("h")
                axes_error[i].set_ylabel("Erreur")
                axes_error[i].set_title(f"Erreurs pour {methode}")
                axes_error[i].legend()
                axes_error[i].grid()

        st.pyplot(fig)
        if y_exact:
            st.pyplot(fig_error)

        # Tracé log-log et calcul des ordres de convergence si la solution exacte est connue
        if y_exact:
            fig_loglog, ax_loglog = plt.subplots(figsize=(10, 6))
            for methode in methodes_choisies:
                errors = errors_all_methods[methode]
                ax_loglog.loglog(h_values, errors, '-o', label=f"Erreur {methode}")
                if len(errors) > 1:
                    p = np.log(errors[-2] / errors[-1]) / np.log(h_values[-2] / h_values[-1])
                    st.write(f"Ordre de convergence pour la méthode {methode} : {round(p)}")

            ax_loglog.set_xlabel('log(h)')
            ax_loglog.set_ylabel('log(||e||_∞)')
            ax_loglog.set_title("Ordre de convergence des schémas")
            ax_loglog.legend()
            ax_loglog.grid(True, which="both", linestyle='--')
            st.pyplot(fig_loglog)

        # **Option de téléchargement des graphiques**
        def download_plot(fig, filename, label):
            # Sauvegarde le graphique dans un buffer et le renvoie sous forme de fichier téléchargeable
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(label=label, data=buf, file_name=filename, mime="image/png")
        
        # Téléchargement des graphiques des solutions et des erreurs
        download_plot(fig, "solutions.png", "Télécharger le graphique des solutions")
        if y_exact:
            download_plot(fig_error, "erreurs.png", "Télécharger le graphique des erreurs")
        
        # **Option de téléchargement des valeurs**
        def download_values(t, y, filename):
            # Sauvegarde des valeurs dans un fichier CSV
            df = pd.DataFrame({"t": t, "y": y})
            csv = df.to_csv(index=False)
            st.download_button(label="Télécharger les valeurs", data=csv, file_name=filename, mime="text/csv")
        
        # Téléchargement des valeurs des solutions numériques
        download_values(t, y, "valeurs_solution.csv")

