#conda create -n projet python pandas numpy matplotlib jupyterlab kagglehub seaborn streamlit plotly 
#conda activate projet
#jupyter lab --notebook-dir="c:/"
# se situer dans le répertoir avec le fichier py
# streamlit run application_LECOQ_ROESCH.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
#df = pd.read_csv("........ds_salaries.csv")
df = pd.read_csv("ds_salaries.csv")

### 2. Exploration visuelle des données
st.title("📊 Visualisation des Salaires en Data Science")
st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")

# Aperçu des données
if st.checkbox("Afficher un aperçu des données"):
    st.write(df.head())


#Statistique générales avec describe pandas 
# Statistiques générales
st.subheader("📌 Statistiques générales")
st.write(df.describe()) # discribe pour voir les statistique de base liée a la df
st.markdown("On constate que le salaire(USD) moyen est de 137 000$. Le taux de télétravail moyen est de 46%.")

### 3. Distribution des salaires en France par rôle et niveau d'expérience, uilisant px.box et st.plotly_chart
st.subheader("📈 Distribution des salaires en France")


df_france = df.query("company_location == 'FR'") # permet de filtré sur la france uniquement
fig_box = px.box(df_france, x="experience_level", y="salary_in_usd", color="experience_level")
st.plotly_chart(fig_box)

st.markdown("On constate ici que le salaire moyen des seniors est le plus élevé avec pratiquement 80 000$ usd, contre 60 000$ pour les moyennement experimentés et 40 000$ pour les peu expertimentés. Il n'y a pas d'experts en France. Le salaire minimum d'un serior est supérieur au salaire moyen d'un junior et pratiquement supérieur à son salaire maximum. On constate que les salaires les plus élevés de notre jeu de données sont pratiquement tous gagnés par des seniors, ce uqi paraît plutôt légitime.")



### 4. Analyse des tendances de salaires :
#### Salaire moyen par catégorie : 
#### en choisisant une des : ['experience_level', 'employment_type', 'job_title', 'company_location'], utilisant px.bar et st.selectbox 
st.subheader("Salaire moyen par catégorie")

categorie = st.selectbox("Sélectionnez une catégorie", ['experience_level', 'employment_type', 'job_title', 'company_location'])
# création d'un datafarme qui group by la catégorie selectionner, qui fait une moyenne des salaire, et qui rénisialise les index (sans index, les graphiques ne peuvent pas se générer)
df_mean_salary = df.groupby(categorie)["salary_in_usd"].mean().reset_index()
fig_bar = px.bar(df_mean_salary, x=categorie, y='salary_in_usd', color=categorie)
st.plotly_chart(fig_bar)
st.markdown("Ici nous pouvons voir le salaire moyen selon plusieurs axes d'analyse : le niveau d'expérience, la catégorie d'emploi, le métier, et le pays de l'entreprise. Pour ce qui est de la catégorie d'emploi, il y a un gros écart entre les contrats en plein temps et les 'contractors' et les autres. Le métier qui ressort bien aju dessus des autres est Applied Machine Learning Engineer. Et le pays qui paye le mieux ses employés dans la data est Israel.")



### 5. Corrélation entre variables
# Sélectionner uniquement les colonnes numériques pour la corrélation*
st.subheader("🔗 Corrélations entre variables numériques")





# Calcul de la matrice de corrélation
#votre code


# Affichage du heatmap avec sns.heatmap
#votre code 
#st.subheader("🔗 Corrélations entre variables numériques")




### 6. Analyse interactive des variations de salaire
# Une évolution des salaires pour les 10 postes les plus courants
# count of job titles pour selectionner les postes
# calcule du salaire moyen par an
#utilisez px.line
#votre code 





### 7. Salaire médian par expérience et taille d'entreprise
# utilisez median(), px.bar
#votre code 




### 8. Ajout de filtres dynamiques
#Filtrer les données par salaire utilisant st.slider pour selectionner les plages 
#votre code 




### 9.  Impact du télétravail sur le salaire selon le pays




### 10. Filtrage avancé des données avec deux st.multiselect, un qui indique "Sélectionnez le niveau d'expérience" et l'autre "Sélectionnez la taille d'entreprise"
#votre code 

