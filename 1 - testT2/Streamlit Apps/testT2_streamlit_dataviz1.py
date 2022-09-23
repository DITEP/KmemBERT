'''
    Application Streamlit : Patients DashBoard
    Autheur : Théo Di Piazza pour Centre Léon Bérard, Lyon

    Informations sur le jeu de données.

    Merci d'avoir un dataframe dans le même repo que ce script pour run.

    commande pour run : streamlit run testT2_streamlit_dataviz1.py
'''
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from PIL import Image
from datetime import datetime
from utils import add_dateCreation, strDate_to_days_bis, strDate_to_days_cr
from lifelines import KaplanMeierFitter

#######################################################################################
# App config
st.set_page_config(
    page_title="Patients Dashboard",
    page_icon="🚑",
    layout="wide",
)
clb_logo = Image.open("clb.jpg")
st.image(clb_logo, width = 150)
st.markdown("### Exploration du jeu de données")

################################################################################
c1, c2 = st.columns(2)
# Lecture et filtre des donnees
# Slider sur les donnees temporelles
with c1:
    values = st.slider(
        "Selectionner un intervalle de temps pour l'année de prise en compte.",
        2000, 2022, (2000, 2022), step=1)
    cohort = st.radio(
     "Sélectionner la cohorte à étudier",
     ('Aucune', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
with c2:
    res_button = st.button('Appliquer mes choix')
    if(cohort=="Aucune"):
        res_button = False
        st.write("Veuillez sélectionner une cohorte avant d'appliquer vos choix.")
    else:
        if(cohort=="Aucune"):
            st.write("Aucune cohorte sélectionée.")
        else:
            st.write("Cohorte n°" + str(cohort))

if res_button:

    # Lecture des donnees et modifications
    #df = pd.read_csv("doc_toTest\\doc_dataviz1.csv")
    df = pd.read_csv("doc_toTest\\test_rs" + str(cohort) + ".csv")
    df = add_dateCreation(df)
    df['TIME_SURVIVAL'] = df.apply(strDate_to_days_bis, axis=1)
    df['TIME_SURVIVAL_CR'] = df.apply(strDate_to_days_cr, axis=1)
    # Filtre sur annee au besoin
    df['year_crea'] = df['Date creation'].apply(lambda x: int(x[:4]))
    df = df[(df.year_crea>=values[0]) & (df.year_crea<=values[1])]
    df = df.drop('year_crea', axis=1)
    # Ajout de colonnes
    df["Nb_doc"] = df.groupby(["Noigr"])["Noigr"].transform("count")
    df['Nb_letter'] = df.Texte.apply(lambda x: len(x)-x.count(" "))
    df['Nb_word'] = df.Texte.apply(lambda x: len(x.split()))

    df_unique = df.groupby('Noigr').first()

    ################################################################################
    # First CONTAINER
    placeholder = st.empty()
    with placeholder.container():
        
        ####################################################################
        # METRICS
        # create 2 columns
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label = "Nombre de patients 🆔",
            value = len(df.Noigr.unique())
        )
        
        kpi2.metric(
            label = "Nombre CR disponibles 📄",
            value = len(df)
        )

        kpi3.metric(
            label = "Survie moyenne ⌛​",
            value = str(math.floor(np.mean(np.array(df_unique['TIME_SURVIVAL'])))) + " jours"
        )

        kpi4.metric(
            label = "Survie moyenne par CR ⌛​",
            value = str(math.floor(np.mean(df['TIME_SURVIVAL_CR']))) + " jours"
        )

        ####################################################################
        # 2 Plots : Violin Plot, KM

        # Creation des 2 colonnes
        p11, p12 = st.columns(2)

        # Premiere colonne - Temps de survie
        with p11:
            st.markdown("### Violin Plot : Temps de survie par patient")
            fig = plt.figure(figsize=(8, 4))
            sns.set_theme(style="whitegrid")
            ax = sns.violinplot(data=df_unique, y='TIME_SURVIVAL', color="dodgerblue", inner=None, linewidth=0, saturation=0.5)
            sns.boxplot(data=df_unique, y='TIME_SURVIVAL', saturation=0.5, width=0.4,
                        color="skyblue", boxprops={'zorder': 2}, ax=ax)
            plt.axhline(y=365, c='red', linestyle='dashed', label="365 days")
            ax.set_ylabel("Survival Time (days)", fontsize = 12)
            ax.set_xlabel("Status", fontsize = 12)
            ax.set_xticklabels(['Patients'])
            plt.legend(loc='upper left')
            plt.ylim([0, 7000])
            st.write(fig)
            plt.title("Violin plot : Survival Time per patient")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\violinPlot_survivalTime.png', bbox_inches='tight')


        # Deuxieme colonne : KM
        with p12:
            st.markdown("### Courbe de survie par CR: Kaplan Meier")
            kmf = KaplanMeierFitter()
            kmf.fit(df['TIME_SURVIVAL_CR'])
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
            kmf.plot_survival_function()
            ax.set_ylabel('Probability of survival', fontdict = {'fontsize' : 12})
            ax.set_xlabel('Time (days)', fontdict = {'fontsize' : 12})
            ax.set_ylim([-0.05, 1.05])
            st.write(fig)
            plt.title("Survival Curve : Survival Time at each consultation [Kaplan Meier]")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\KaplanMeier_survivalCurve.png', bbox_inches='tight')

        st.markdown("""---""")

        # Creation des 2 colonnes
        p21, p22 = st.columns(2)
        # Premiere colonne
        with p21:
            st.markdown("### Histogramme : Année de prise en compte")
            fig = plt.figure(figsize=(8,4))
            ax = sns.distplot(df_unique['Date creation'].apply(lambda x: int(x[:4])), 
                        hist=True, kde=True, 
                        bins=20, color = 'lightgreen', 
                        hist_kws={'edgecolor':'black'},
                        kde_kws={'linewidth': 4})
            plt.xlim([2000, 2020])
            plt.xticks(list(range(2000,2021, 2)), rotation=45)
            ax.set_ylabel("Density", fontsize = 12)
            ax.set_xlabel("Year of admission", fontsize = 12)
            st.write(fig)
            plt.title("Histogram : Year the patient was admitted to the Centre Léon Bérard")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\Histogramme_AnneePriseEnCompte.png', bbox_inches='tight')

        # Deuxieme colonne
        with p22:
            st.markdown("### Histogramme : Année de décès")
            fig = plt.figure(figsize=(8,4))
            ax = sns.distplot(df_unique['Date deces'].apply(lambda x: int(x[:4])), 
                        hist=True, kde=True, 
                        bins=20, color = 'lightcoral', 
                        hist_kws={'edgecolor':'black'},
                        kde_kws={'linewidth': 4})
            plt.xlim([2000, 2020])
            plt.xticks(list(range(2000,2021, 2)), rotation=45)
            ax.set_ylabel("Density", fontsize = 12)
            ax.set_xlabel("Year of death", fontsize = 12)
            st.write(fig)
            plt.title("Histogram : Year of death")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\Histogramme_AnneeDeDeces.png', bbox_inches='tight')

        st.markdown("""---""")

        # Creation des 2 colonnes
        p31, p32 = st.columns(2)

        with p31:
            st.markdown("### Histogramme : Nombre de documents par patient")
            fig = plt.figure(figsize=(8, 4))
            sns.set_theme(style="whitegrid")
            ax = sns.histplot(data=df_unique, y="Nb_doc", color="sandybrown")

            fq, med, tq = df_unique.Nb_doc.quantile([0.25,0.5,0.75])
            plt.axhline(y=np.mean(np.array(df_unique['Nb_doc'])), c='red', linestyle='dashed', label="Moyenne")
            plt.axhline(y=med, c='brown', linestyle='dashed', label="Mediane")
            plt.axhline(y=fq, c='springgreen', linestyle='dashed', label="Premier quartile")
            plt.axhline(y=tq, c='springgreen', linestyle='dashed', label="Deuxieme quartile")

            ax.set_ylabel("Nombre de documents", fontsize = 14)
            ax.set_xlabel("Nombre de patients", fontsize = 14)
            plt.legend(loc='upper right')
            st.write(fig)
            plt.title("Histogram : Number of documents per patient")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\Histogramme_NbDocsParPatient.png', bbox_inches='tight')

        with p32:
            st.markdown("### PieChart : Répartition des patients")
            total_patients = len(df_unique)
            less4_patients = len(df_unique[df_unique['Nb_doc']<=4])
            more4_patients = total_patients - less4_patients
            data = [less4_patients, more4_patients]
            labels = ['< 4 documents.\n ['+str(less4_patients)+' patients]', 
                    '>= 4 documents.\n ['+str(more4_patients)+' patients]']
            #define sns color palette to use
            colors = sns.color_palette('pastel')[0:5]
            #create pie chart
            fig = plt.figure(figsize=(7, 3))
            plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
            st.write(fig)
            plt.title("PieChart : Distribution of patients by number of documents")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\PieChart_RepartitionPatientsDocs.png', bbox_inches='tight')

        st.markdown("""---""")

        # Creation des 2 colonnes
        p41, p42 = st.columns(2)

        with p41:
            st.markdown("### Histogramme : Nombre de lettres par document")
            fig = plt.figure(figsize=(8, 4))
            sns.set_theme(style="whitegrid")
            ax = sns.histplot(data=df, x="Nb_letter", color="orchid")

            fq, med, tq = df.Nb_letter.quantile([0.25,0.5,0.75])
            plt.axvline(x=np.mean(np.array(df.Nb_letter)), c='red', linestyle='dashed', label="Moyenne")
            plt.axvline(x=med, c='brown', linestyle='dashed', label="Mediane")
            plt.axvline(x=fq, c='springgreen', linestyle='dashed', label="Premier quartile")
            plt.axvline(x=tq, c='springgreen', linestyle='dashed', label="Deuxieme quartile")

            ax.set_xlabel("Nombre de lettres", fontsize = 14)
            ax.set_ylabel("Nombre de documents", fontsize = 14)
            plt.legend(loc='upper right')
            st.write(fig)
            plt.title("Histogram : Number of letters per document")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\Histogramme_NbLettresDocs.png', bbox_inches='tight')

        with p42:
            st.markdown("### Histogramme : Nombre de mots par document")
            fig = plt.figure(figsize=(8, 4))
            sns.set_theme(style="whitegrid")
            ax = sns.histplot(data=df, x="Nb_word", color="orchid")

            fq, med, tq = df.Nb_word.quantile([0.25,0.5,0.75])
            plt.axvline(x=np.mean(np.array(df.Nb_word)), c='red', linestyle='dashed', label="Moyenne")
            plt.axvline(x=med, c='brown', linestyle='dashed', label="Mediane")
            plt.axvline(x=fq, c='springgreen', linestyle='dashed', label="Premier quartile")
            plt.axvline(x=tq, c='springgreen', linestyle='dashed', label="Deuxieme quartile")

            ax.set_xlabel("Nombre de mots", fontsize = 14)
            ax.set_ylabel("Nombre de documents", fontsize = 14)
            plt.legend(loc='upper right')
            st.write(fig)
            plt.title("Number of words per document")
            plt.savefig('saveFigures\\cohort'+str(cohort)+'\\Histogramme_NbMotsParDocuments.png', bbox_inches='tight')

#####################################################################################
st.markdown("""---""")
st.write('Si besoin, veuillez contactez Théo Di Piazza (theo.dipiazza@lyon-unicancer.fr)')


