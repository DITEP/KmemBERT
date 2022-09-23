'''
    Application Streamlit : Dashboard Patient
    Autheur : Théo Di Piazza pour Centre Léon Bérard, Lyon

    Informations de bases sur le patient, courbe de survie, suivi et prédiction.

    Merci d'avoir un dataframe dans le même repo que ce script pour run.

    commande pour exécuter l'application : streamlit run testT2_streamlit2.py
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
import datetime
import math
from PIL import Image
from utils import strDate_to_days
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter
from datetime import date, datetime
from matplotlib.ticker import MaxNLocator

#######################################################################################
# App config
st.set_page_config(
    page_title="Patients Dashboard",
    page_icon="🚑",
    layout="wide",
)

#######################################################################################
# UI Interface : CLB image
clb_logo = Image.open("clb.jpg")
st.image(clb_logo, width = 150)
st.markdown("<h1 style='text-align: center; color: black;'>Application de visualisation patient</h1>", unsafe_allow_html=True)

#######################################################################################
# Read dataframe
df = pd.read_csv("doc_streamlit_Little_02052022.csv").drop('Unnamed: 0', axis=1)
# Ajout des '-' pour faciliter la lecture des dates
# Date creation
df['Date creation'] = df['Date creation'].apply(lambda x: str(x))
df['Date creation'] = df['Date creation'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
# Date deces
df['Date deces'] = df['Date deces'].apply(lambda x: str(x))
df['Date deces'] = df['Date deces'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
# Date derniere maj
df['Date derniere maj'] = df['Date derniere maj'].apply(lambda x: str(x))
df['Date derniere maj'] = df['Date derniere maj'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
# Date cr
df['Date cr'] = df['Date cr'].apply(lambda x: str(x))
df['Date cr'] = df['Date cr'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
# Compute and add Survival Time
df['TIME_SURVIVAL'] = df.apply(strDate_to_days, axis=1)
#######################################################################################
# Box to select ID patient and filter on it
id_patient = st.selectbox("Quel patient souhaitez-vous étudier ?", df.Noigr.unique())
df_option = df[df.Noigr == id_patient]

#######################################################################################
# creating a single-element container
placeholder = st.empty()

with placeholder.container():

    # create three columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="ID Patient 🆔",
        value=id_patient
    )
    
    kpi2.metric(
        label="Nombre CR disponibles 📄",
        value=len(df_option)
    )
    
    kpi3.metric(
        label="Date prise en compte 📆",
        value=df_option.iloc[0]['Date creation']
    )

    if(df_option.iloc[0]['FLAG_DECES']==1):
        kpi4.metric(
            label="Date décès 📆",
            value=str(df_option.iloc[0]['Date deces'])
        )
    else:
        kpi4.metric(
            label="Date MàJ 📆",
            value=str(df_option.iloc[0]['Date derniere maj'])
        )

    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)

    # Premiere courbe : Courbe de survie
    with fig_col1:
        st.markdown("### Courbe de survie")
        kmf = KaplanMeierFitter()
        kmf.fit(df['TIME_SURVIVAL'], df['FLAG_DECES'])
        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        kmf.plot(ax=ax, ci_show=True, show_censors=True)
        # Ajout ligne verticale (time to event)
        if(df_option.iloc[0]['FLAG_DECES']==1):
            ax.axvline(x=df_option.iloc[0]['TIME_SURVIVAL'], color="red", ls="--", lw=1, label="Time to death")
        else:
            ax.axvline(x=df_option.iloc[0]['TIME_SURVIVAL'], color="purple", ls="--", lw=1, label="Time to censore")
        ax.legend(loc="topright", prop={'size': 6})
        ax.set_ylabel('Survival probability', fontdict = {'fontsize' : 7})
        ax.set_xlabel('Time in days', fontdict = {'fontsize' : 7})
        ax.set_ylim([-0.05, 1.05])
        plt.xticks(fontsize=5); plt.yticks(fontsize=5)
        st.write(fig)
    
    # Seconde courbe : Regularite de suivi patient
    with fig_col2:
        st.markdown("### Régularité de suivi patient")
        
        # Recuperer la frequence de consultation du patient
        date_and_occurences = df_option.groupby(pd.to_datetime(df_option['Date cr']).dt.strftime('%m-%Y'), sort=False).size()
        freq_consult = pd.DataFrame({"month": date_and_occurences.index.tolist(), 
                                "occ": date_and_occurences.values.tolist()})
        x = freq_consult.month.apply(lambda u: datetime.strptime("01-"+u, '%d-%m-%Y'))
        y = freq_consult.occ
        # Creation de la figure et gestion des axes
        fig = plt.figure(figsize=(3, 2))
        ax = plt.subplot(111)
        ax.bar(x, y, width=10)
        ax.xaxis_date()
        ax.set_ylabel('Nbrs de consultation', fontdict = {'fontsize' : 7})
        ax.set_xlabel('Temps', fontdict = {'fontsize' : 7})
        # Gestion des valeurs sur les axes
        yint = range(int(min(y)), math.ceil(max(y))+1)
        plt.xticks(rotation=45, fontsize=5); plt.yticks(yint, fontsize=5)
        st.write(fig)


    ### Prediction pour un CR
    # Contenu du CR
    st.markdown("### Contenu et date du CR pour une prédiction")
    fig_col3, fig_col4, fig_col5 = st.columns(3)
    with fig_col3:
        txt = st.text_area('Contenu consultation', '''
        Le patient va bien. Il a été vue pour [...].
        ''')
        st.write('Compte-Rendu:', txt)

    # Date du CR
    with fig_col4:
        d = st.date_input(
        "Date de consultation",
        date.today())
        st.write('Date de consultation:', d)

    # Clique pour recuperer la prediction
    with fig_col5:
        if st.button('Prédiction du modèle'):
            st.metric(
            label="Probabilité de survie dans l'année",
            value=0.72,
            delta = -0.04
            )



    # Vue detaillee du jeu de donnees
    st.markdown("### Vue détaillée du jeu de données")
    st.dataframe(df)
#####################################################################################

# UI Interface : Horizontal Line
st.markdown("""---""")

st.write('Si besoin, veuillez contactez Théo Di Piazza (theo.dipiazza@lyon-unicancer.fr)')

