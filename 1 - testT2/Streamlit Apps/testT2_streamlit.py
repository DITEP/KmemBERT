import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils import strDate_to_days
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter

# App config
st.set_page_config(
    page_title="Patients Dashboard",
    page_icon="🚑",
    layout="wide",
)

# UI Interface : CLB image
clb_logo = Image.open("clb.jpg")
st.image(clb_logo, width = 150)
st.markdown("<h1 style='text-align: center; color: black;'>Application de visualisation patient</h1>", unsafe_allow_html=True)

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

# UI Interface : Title
st.subheader('Aperçu du jeu de données :')
st.dataframe(df.head(3))

# UI Interface : Horizontal Line
st.markdown("""---""")
# Selection du patient à etudier
# Using object notation
option = st.sidebar.selectbox(
    'Quel patient souhaitez-vous étudier ?',
     df.Noigr.unique()
)
df_option = df[df.Noigr == option]

# Affichage informations patient (ID, dates, nb CR)
# UI Interface
st.subheader('Informations sur le patient :')
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col1.metric("ID patient", option)
col2.metric("Nombre CR disponibles", len(df_option))
col3.metric("Date prise en compte", df_option.iloc[0]['Date creation'])
if(df_option.iloc[0]['FLAG_DECES']==1):
    col4.metric("Date décès", str(df_option.iloc[0]['Date deces']))
else:
    col4.metric("Date dernière MàJ", df_option.iloc[0]['Date derniere maj'])

#UI Interface
st.subheader('Courbe de survie (Kaplan-Meier) :')

# Survival Plot - For all patients - Kaplan Meier
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
st.pyplot(fig)

# UI Interface : Horizontal Line
st.markdown("""---""")

st.write('Si besoin, veuillez contactez Théo Di Piazza (theo.dipiazza@lyon-unicancer.fr) :')

