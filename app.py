import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import openai

from gpt_explanation import *
from model_evaluation import * 
from data_exploration import *

# Charge les données du NYC Taxi Dataset
data = pd.read_csv('data/default_nyc_taxi.csv')
df = data.copy()
data['class'] = data['class'].astype(str)
data['class'].replace("1","anomaly",inplace=True) 
data['class'].replace("0","normal",inplace=True) 

# Liste des modèles à choisir
model_list = ["LSTM","AutoEncoder","IsolationForest","Prophet","LocalOutlierFactor","DBScan"]

# OpenAI API Key (remplace "ta_clé_api_openai" par ta clé API)
openai.api_key = "ta_clé_api_openai"

#################
### SQUELETTE ###
#################

st.set_page_config(layout="wide")

def presentation():
    st.header('Présentation')
    st.markdown(f"""    
        ### Contexte
            
        Ce Projet de Fin d'Études (PFE) intitulé "Détection d'anomalies sur le NY Taxi Dataset" est un travail complexe qui illustre l'importance de l'analyse de données pour la détection d'anomalies dans le cadre urbain. En étudiant les motifs de mobilité dans l'une des villes les plus trépidantes du monde, New York, ce projet explore les nombreux facteurs qui peuvent influencer ces motifs et comment nous pouvons utiliser les données pour anticiper les changements inattendus. Ainsi, l'information que nous recueillons et analysons n'est pas seulement académiquement intéressante, mais elle a également des implications pratiques significatives, par exemple en aidant les services de transport urbain à mieux comprendre et répondre aux variations de la demande.
                """,unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1 :
        st.image("./data/img/taxi3.jpg",use_column_width=True)
    with col2 :
        st.image("./data/img/taxi4.jpg",use_column_width=True)

    st.markdown(f"""
        ### Objectif

        L'objectif de ce PFE est de mettre au point une méthode efficace pour détecter les anomalies dans les données, ce qui nous permettrait d'identifier les écarts significatifs par rapport aux tendances et aux motifs habituels. Ces anomalies peuvent être dues à une multitude d'événements, tels que des pannes de métro, des événements sportifs majeurs ou des catastrophes naturelles. En reconnaissant ces anomalies rapidement et précisément, nous sommes en mesure de contribuer à la résilience urbaine, permettant aux décideurs de réagir en temps réel aux situations imprévues.

        C'est ce qui rend ce projet si passionnant et pertinent : l'idée que grâce à l'analyse des données, nous pouvons transformer un flux apparemment chaotique d'informations en des aperçus précieux, qui peuvent ensuite être utilisés pour améliorer la vie urbaine. A travers ce travail, j'ai eu l'occasion d'apporter une contribution à ce domaine passionnant et en évolution constante.
    """, unsafe_allow_html=True)

    st.image("./data/img/anomaly.png",width=600)

    st.markdown(f"""
        ### Dataset
        Le projet repose sur l'utilisation du nyc_taxi.csv, une base de données fournies par le Numenta Anomaly Benchmark (NAB). Cette base de données renferme un grand nombre de données détaillées recueillies sur une longue période concernant le nombre de courses de taxi effectuées à New York à chaque heure. L'ampleur et la densité de ces informations font du nyc_taxi.csv un outil de recherche précieux, permettant de saisir les tendances, les motifs et les exceptions dans le paysage des transports urbains. Chaque enregistrement représente un instantané de la vie urbaine, rendant le dataset incroyablement riche et fascinant à explorer.
    """, unsafe_allow_html=True)

    st.image("./data/img/nyc.png",width=1600)

def data_exploration():
    st.header('Data Exploration')

    # Visualisation du dataframe
    col1, col2 = st.columns(2)
    with col1 :
        st.subheader('Visualisation du dataframe')
        n, m = data.shape
        st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)
        st.write(df)
    with col2 :
        st.subheader('Visualisation des informations du dataframe')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.write(df_info(df))
        st.subheader('Feature Engineering')
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['weekend'] = (df['dayofweek'] > 5).astype(int)
        df['daylight'] = ((df['hour']>=7) & (df['hour']<=20)).astype(int)
        st.write(df.head())

    # Visualise les données de la time series
    st.subheader('Visualisation des données')
    fig = px.scatter(data, x='timestamp', y='value',color="class",title='NYC Taxi Dataset',color_discrete_map={"anomaly": "green", "normal": "blue"})
    fig.update_layout(autosize=False, width=1000, height=1000)
    st.plotly_chart(fig, use_container_width=True)

    res = seasonal_decompose(df["value"], model='additive', period=48)
    df["trend"],df["seasonal"],df["residuals"] = res.trend,res.seasonal,res.resid
    fig1 = px.line(df,title='Décomposition de la série temporelle',x="timestamp",y=["value","trend","seasonal","residuals"],markers=True)
    fig1.update_layout(autosize=False, width=1000, height=1000)
    st.plotly_chart(fig1, use_container_width=True)

    col3, col4, = st.columns(2)
    with col3 :
        fig2 = px.histogram(df,title='Distribution de la demande', x="value", color="class")
        st.plotly_chart(fig2, use_container_width=True)
    with col4 :
        fig3 = px.histogram(df,title="Distribution de la variable cible", x="class",color="class")
        st.plotly_chart(fig3,use_container_width=True)
        
    col6, col7, col8 = st.columns(3)
    with col6 :
        fig4 = px.bar(df.groupby(['class', 'year']).mean().reset_index(),title='Exploration par années de la série', x="year",y="value", color="class")
        st.plotly_chart(fig4, use_container_width=True)
    with col7 :
        fig5 = px.bar(df.groupby(['class', 'month']).mean().reset_index(),title='Exploration par mois de la série', x="month",y="value", color="class")
        st.plotly_chart(fig5, use_container_width=True)
    with col8 :
        fig6 = px.bar(df.groupby(['class', 'hour']).mean().reset_index(),title='Exploration par heure de la série', x="hour",y="value", color="class")
        st.plotly_chart(fig6, use_container_width=True)

    col9, col10, col11 = st.columns(3)
    with col9 :
        fig9 = px.bar(df.groupby(['class', 'dayofweek']).mean().reset_index(),title='Exploration par jour de la semaine de la série', x="dayofweek",y="value", color="class")
        st.plotly_chart(fig9, use_container_width=True)
    with col10 :
        fig10 = px.bar(df.groupby(['daylight', 'dayofweek']).mean().reset_index(),title='Exploration par luminosité (jour/nuit) de la série', x="dayofweek",y="value", color="daylight")
        st.plotly_chart(fig10, use_container_width=True)
    with col11 :
        fig11 = px.bar(df.groupby(['hour', 'dayofweek']).mean().reset_index(),title='Exploration par jour & heure de la série', x="dayofweek",y="value", color="hour")
        st.plotly_chart(fig11, use_container_width=True)

def anomaly_detection():
    st.header('Détection d\'anomalies')
    st.subheader("Modèles de détection d'anomalies")
    selected_model = st.selectbox("Choisissez un modèle", model_list)

    def load_anomalies_data(selected_model):
        anomalies_data = pd.read_csv(f'data/{selected_model.lower()}_nyc_taxi.csv')
        anomalies_data = anomalies_data.sort_values('timestamp')  # Assure-toi que les données sont triées par temps
        anomalies_data['pred'] = anomalies_data['pred'].astype(str)
        anomalies_data['pred'].replace("1","anomaly",inplace=True) 
        anomalies_data['pred'].replace("0","normal",inplace=True)
        daily_anomalies = anomalies_data.copy()
        daily_anomalies['timestamp'] = pd.to_datetime(daily_anomalies['timestamp'])
        daily_anomalies["date"] = daily_anomalies['timestamp'].dt.date
        return anomalies_data,daily_anomalies

    if "detect_anomalies" not in st.session_state:
        st.session_state["detect_anomalies"] = False

    if st.button('Détecter les anomalies'):
        st.session_state["detect_anomalies"] = True

    if st.session_state["detect_anomalies"]:

        # Affiche les scores
        st.markdown('**Métriques**', unsafe_allow_html=True)
        total_df = pd.read_csv('data/total_nyc_taxi.csv') 
        silhouette,auprc,f1,precision,recall = evaluate(total_df,selected_model.lower()+"_pred")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(label="Silhouette Score", value=silhouette)
        with col2:
            st.metric(label="AUPRC", value=auprc)
        with col3:
            st.metric(label="F1", value=f1)
        with col4 :
            st.metric(label="Precision", value=precision)
        with col5 :
            st.metric(label="Recall", value=recall)

        # Charge l'ensemble de données des anomalies pour le modèle sélectionné
        anomalies_data,daily_anomalies = load_anomalies_data(selected_model)

        # Crée un graphique avec les anomalies en couleurs et l'affiche
        fig_anomalies = px.scatter(anomalies_data, x='timestamp', y='value',color="pred",title='Anomalies détectées',color_discrete_map={"anomaly": "red", "normal": "blue"})
        fig_anomalies.update_layout(autosize=False, width=1000, height=800)
        st.plotly_chart(fig_anomalies, use_container_width=True)

        # Crée une liste des anomalies
        anomalies_list = [""] + daily_anomalies[daily_anomalies['pred'] == "anomaly"]['date'].unique().tolist()
        
        # Affiche les vraies anomalies
        correct_data = pd.read_csv('data/default_nyc_taxi.csv')
        correct_data['class'] = correct_data['class'].astype(str)
        correct_data['class'].replace("1","anomaly",inplace=True) 
        correct_data['class'].replace("0","normal",inplace=True) 
        fig2 = px.scatter(correct_data, x='timestamp', y='value',color="class",title='Anomalies attendues',color_discrete_map={"anomaly": "green", "normal": "blue"})
        fig2.update_layout(autosize=False, width=1000, height=800)
        st.plotly_chart(fig2, use_container_width=True)   

def anomaly_explainer():
    st.header('Explication d\'anomalies')
    st.subheader("Modèles de détection d'anomalies")
    selected_model = st.selectbox("Choisissez un modèle", model_list)

    def load_anomalies_data(selected_model):
        anomalies_data = pd.read_csv(f'data/{selected_model.lower()}_nyc_taxi.csv')
        anomalies_data = anomalies_data.sort_values('timestamp')  # Assure-toi que les données sont triées par temps
        anomalies_data['pred'] = anomalies_data['pred'].astype(str)
        anomalies_data['pred'].replace("1","anomaly",inplace=True) 
        anomalies_data['pred'].replace("0","normal",inplace=True)
        daily_anomalies = anomalies_data.copy()
        daily_anomalies['timestamp'] = pd.to_datetime(daily_anomalies['timestamp'])
        daily_anomalies["date"] = daily_anomalies['timestamp'].dt.date
        return anomalies_data,daily_anomalies

    if "detect_anomalies" not in st.session_state:
        st.session_state["detect_anomalies"] = False

    if st.button('Détecter les anomalies'):
        st.session_state["detect_anomalies"] = True

    if st.session_state["detect_anomalies"]:
        # Charge l'ensemble de données des anomalies pour le modèle sélectionné
        anomalies_data,daily_anomalies = load_anomalies_data(selected_model)

        # Crée un graphique avec les anomalies en couleurs et l'affiche
        fig_anomalies = px.scatter(anomalies_data, x='timestamp', y='value',color="pred",title='Anomalies détectées',color_discrete_map={"anomaly": "red", "normal": "blue"})
        fig_anomalies.update_layout(autosize=False, width=1000, height=800)
        st.plotly_chart(fig_anomalies, use_container_width=True)

        # Crée une liste des anomalies
        anomalies_list = [""] + daily_anomalies[daily_anomalies['pred'] == "anomaly"]['date'].unique().tolist()

        # Affiche les explications potentielles
        st.image('./data/img/nyc.png',width=1500)

        # Affiche un menu déroulant pour sélectionner une anomalie à expliquer
        st.markdown('**Sélectionnez une anomalie à expliquer**', unsafe_allow_html=True)
        selected_anomaly = st.selectbox('', anomalies_list)
        thinking_image = st.empty()
        thinking_message = st.empty()
        
        if selected_anomaly:
            thinking_image.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTV1endwNHZnMXRoMGQ1c2trNGhoZTN2NHQ4bDVqbzd3MXRnNjMydiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6x6uJ04HWKQmHUypFD/giphy.gif)")
            thinking_message.markdown("<div>L'assistant réfléchit...</div>", unsafe_allow_html=True)

            # Obtient une explication de l'API OpenAI
            explanation_prompt = build_prompt(str(selected_anomaly))
            explanation = get_completion(openai,explanation_prompt)

            # Met à jour le message de l'assistant
            thinking_image.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDY1NHE2Zmc2ZGttZDJnbDNtb201emE0ZDZwN2cxaDEyamwwaGNmdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2Gy39kd4oJqoWSk8XL/giphy.gif)")
            thinking_message.markdown("<div>"+explanation+"</div>", unsafe_allow_html=True)
        else : 
            thinking_image.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDRjeDh3ZXN0eWF2a25taDFycm1kb2NyNDRudHFqbXdsdjBpbmdmaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Lo0ciD9Hgc7AP58ci0/giphy.gif)")
            thinking_message.markdown("<div>L'assistant attend une demande...</div>", unsafe_allow_html=True)

# Création d'un menu à onglets
menu = ["Présentation","Data Exploration", "Détection d'anomalies","Explication d'anomalies"]
choice = st.sidebar.radio("Menu", menu)

if choice == "Présentation":
    presentation()
elif choice == "Data Exploration":
    data_exploration()
elif choice == "Détection d'anomalies":
    anomaly_detection()
elif choice == "Explication d'anomalies":
    anomaly_explainer()


