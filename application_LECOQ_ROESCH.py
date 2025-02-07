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

st.markdown("https://github.com/LECOQLomann/SAE-6-01-VCOD")
st.markdown("https://sae-6-01-vcod-fwjhhbnth43gkohq6twrhv.streamlit.app/")

# Chargement des données
#df = pd.read_csv("........ds_salaries.csv")
df = pd.read_csv("ds_salaries.csv")

### 2. Exploration visuelle des données
st.title("📊 Visualisation des Salaires en Data Science")
st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")

# Aperçu des données
# affiche un checkbox qui quand il est cocher, afficher le résumer de notre jeu de données (les 5 premières lignes)
if st.checkbox("Afficher un aperçu des données"):
    st.write(df.head())


# Statistique générales avec describe pandas 
# Statistiques générales
st.subheader("📌 Statistiques générales")
st.write(df.describe(include='all')) # discribe pour voir les statistiques sur les variable quantitative 
st.markdown("On constate que le salaire(USD) moyen est de 137 000$. Le taux de télétravail moyen est de 46%.")
st.markdown("Le type d'emploi le plus représenté est FT ou Full time, le niveau d'expérience majoritaire est Senior, et le job le plus présent dans les données est Data Engineer. Les employés proviennent de 78 pays différents et majoritairemnt des US. Il y a 72 pays pour les entreprises, majoritairement en US.Et enfin, la taille la plus représentée en M, entreprise moyenne.")


### 3. Distribution des salaires en France par rôle et niveau d'expérience, uilisant px.box et st.plotly_chart
st.subheader("📈 Distribution des salaires en France")

df_france = df.query("company_location == 'FR'") # permet de filtré sur la france uniquement
fig_box = px.box(df_france, x="experience_level", y="salary_in_usd", color="experience_level",title="Boxplot des salaires par niveau d'expérience en France")#On colore selon la valeur de la variable expéreince
st.plotly_chart(fig_box)

st.markdown("On constate ici que le salaire moyen des seniors est le plus élevé avec pratiquement 80 000$ usd, contre 60 000$ pour les moyennement experimentés et 40 000$ pour les peu expertimentés. Il n'y a pas d'experts en France. Le salaire minimum d'un serior est supérieur au salaire moyen d'un junior et pratiquement supérieur à son salaire maximum. On constate que les salaires les plus élevés de notre jeu de données sont pratiquement tous gagnés par des seniors, ce uqi paraît plutôt légitime.")



### 4. Analyse des tendances de salaires :
#### Salaire moyen par catégorie : 
#### en choisisant une des : ['experience_level', 'employment_type', 'job_title', 'company_location'], utilisant px.bar et st.selectbox 
st.subheader("Salaire moyen par catégorie")

categorie = st.selectbox("Sélectionnez une catégorie", ['experience_level', 'employment_type', 'job_title', 'company_location'])#Grâce à cette selection on choisit note axe d'analyse entre 4 variables. Le graphique sera actualisé en fontion du choix réalisé
# création d'un datafarme qui group by la catégorie selectionner, qui fait une moyenne des salaire, et qui rénisialise les index (sans index, les graphiques ne peuvent pas se générer)
df_mean_salary = df.groupby(categorie)["salary_in_usd"].mean().reset_index()
fig_bar = px.bar(df_mean_salary, x=categorie, y='salary_in_usd', color=categorie, title="Moyenne des salires en fonction de la catégorie sélectionnée")
st.plotly_chart(fig_bar)

st.markdown("Ici nous pouvons voir le salaire moyen selon plusieurs axes d'analyse : le niveau d'expérience, la catégorie d'emploi, le métier, et le pays de l'entreprise. Pour ce qui est de la catégorie d'emploi, il y a un gros écart entre les contrats en plein temps et les 'contractors' et les autres. Le métier qui ressort bien aju dessus des autres est Applied Machine Learning Engineer. Et le pays qui paye le mieux ses employés dans la data est Israel.")



### 5. Corrélation entre variables
# Sélectionner uniquement les colonnes numériques pour la corrélation*
st.subheader("🔗 Corrélations entre variables numériques")

# Calcul de la matrice de corrélation
numeric_cols = df.select_dtypes(include=np.number).columns # séléction des colonnes uniquement numérique de notre datafarm
df_corr = df[numeric_cols].corr() # on calcul la matrix de corélation des données numérics créé précédement


# Affichage du heatmap avec sns.heatmap
fig, ax = plt.subplots() #création d'un figure et d'un axe pour la réprénstation du heatmap
sns.heatmap(df_corr, cmap="coolwarm", annot=True, ax=ax) # création du heatmap grâce a la corrélation calculer plus tôt, annot=True permet d'afficher sur le Heatmap les valeur corréspondent, et spécifie l'axe que le souhaite utiliser, a savoir ax créé également plus haut. cmap="coolwarm" permet d'avoir des couleur varillant du bleu au rouge 
st.pyplot(fig)  

st.markdown("Ici on constate qu'il n'y a aucune corrélation forte entre nos variables. La corrélation maximale est de 0.23 ou -0.24, et c'est entre l'année et le ration de télétravail(négatif) ou bien le salaire (positif).Mais il est clair que ce n'est pas pertinent, car l'année correspond à l'année d'emploi du salairié (entre 2020 et 2023)...")


### 6. Analyse interactive des variations de salaire
# Une évolution des salaires pour les 10 postes les plus courants
# count of job titles pour selectionner les postes
# calcule du salaire moyen par an
# utilisez px.line

st.subheader("📈 Évolution des salaires")


top = df['job_title'].value_counts().nlargest(10).index # permet de compter les occurences de chaque intitulé de poste  et de garde les 10 plus courant
df_top = df[df['job_title'].isin(top)] # permet de ne conserve que les 10 intituler séléctionner
salaire_an = df_top.groupby(['job_title', 'work_year'])['salary_in_usd'].mean().reset_index() # calcule de la moyenne par poste et par année
    
salaires_10 = px.line(salaire_an,x='work_year',y='salary_in_usd', color='job_title', title='Évolution des salaires pour les 10 postes les plus courants',labels={'salary_in_usd': 'Salaire moyen (USD)', 'work_year': 'Année'},template="plotly_white") #création d'un graphique en ligne pour visualisé l'évolution des salaire des 10 postes les plus courant

st.plotly_chart(salaires_10)
st.write('on observe une chute generale des salaires en 2021, probablement a cause de developpement des IA, soit a cause du confinement, mais ils se retablissent rapidements.' )


### 7. Salaire médian par expérience et taille d'entreprise
# utilisez median(), px.bar
#votre code 

st.subheader("Salaire médian par expérience et taille d'entreprise")

df_exp_taille = df.groupby(['company_size', 'experience_level'])['salary_in_usd'].median().reset_index() # Calcul du salaire median selon la taille de la companies par le niveau d'expérience
    
experience_taille = px.bar(df_exp_taille, title="Somme des salaires médianes par expérience et taille d'entreprise", x='company_size', y='salary_in_usd',color='experience_level') # création d'un barplot de la df df_exp_taille de la taille de la companie par le salaire median

st.plotly_chart(experience_taille)
st.write('la lecture de ce graphique n est pas evidante, il faut comprendre que chaque barre est une somme mediannes des types de salaries (leurs experience), et il faut utiliser le fait que graphque est interactif pour observer les mediannes exactes pour tel ou tel classe.' )

### 8. Ajout de filtres dynamiques
#Filtrer les données par salaire utilisant st.slider pour selectionner les plages 

st.subheader("Ajout de filtres dynamiques")

min_ = 0 # création d'une valeur minimum
max_ = 500000 # création d'une valeur maximum
selection_salaire = st.slider('Sélection de la plage de salaire', min_value= min_, max_value=max_, value=(min_,max_)) # création d'un slider avec les iformation des minimums et maximums créé plus tôt
    
salaire = df[(df['salary_in_usd'] >= selection_salaire[0]) & (df['salary_in_usd'] <= selection_salaire[1])] #filtrage du dataFrame en fonction de la plage de salaires sélectionner
st.write(salaire)


### 9.  Impact du télétravail sur le salaire selon le pays




### 10. Filtrage avancé des données avec deux st.multiselect, un qui indique "Sélectionnez le niveau d'expérience" et l'autre "Sélectionnez la taille d'entreprise"
#votre code 

