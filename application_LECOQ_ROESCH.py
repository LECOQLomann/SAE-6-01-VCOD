import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
def load_data():
    df = pd.read_csv("ds_salaries.csv")  # Mets à jour le chemin si nécessaire
    return df

df = load_data()

st.title("📊 Visualisation des Salaires en Data Science")
st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")

# Aperçu des données
if st.checkbox("Afficher un aperçu des données"):
    st.write(df.head())

# Statistiques générales
st.subheader("📌 Statistiques générales")
st.write(df.describe())

# Distribution des salaires en France
st.subheader("📈 Distribution des salaires en France")
df_france = df[df["company_location"] == "FR"]
fig_box = px.box(df_france, x="experience_level", y="salary_in_usd", color="job_title")
st.plotly_chart(fig_box)

# Salaire moyen par catégorie
st.subheader("📊 Salaire moyen par catégorie")
categorie = st.selectbox("Sélectionnez une catégorie", ['experience_level', 'employment_type', 'job_title', 'company_location'])
fig_bar = px.bar(df.groupby(categorie)["salary_in_usd"].mean().reset_index(), x=categorie, y="salary_in_usd")
st.plotly_chart(fig_bar)

# Corrélation entre variables
st.subheader("🔗 Corrélations entre variables numériques")
numeric_cols = df.select_dtypes(include=np.number).columns
df_corr = df[numeric_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(df_corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Évolution des salaires pour les 10 postes les plus courants
st.subheader("📈 Évolution des salaires")
top_jobs = df["job_title"].value_counts().head(10).index
df_top_jobs = df[df["job_title"].isin(top_jobs)]
fig_line = px.line(df_top_jobs.groupby(["work_year", "job_title"])["salary_in_usd"].mean().reset_index(), x="work_year", y="salary_in_usd", color="job_title")
st.plotly_chart(fig_line)

# Filtres dynamiques
st.subheader("🔍 Filtrer par salaire")
min_salary, max_salary = st.slider("Sélectionnez une plage de salaire", int(df["salary_in_usd"].min()), int(df["salary_in_usd"].max()), (50000, 150000))
st.write(df[(df["salary_in_usd"] >= min_salary) & (df["salary_in_usd"] <= max_salary)])

# Impact du télétravail
st.subheader("🏠 Impact du télétravail sur le salaire")
fig_remote = px.box(df, x="remote_ratio", y="salary_in_usd", color="company_location")
st.plotly_chart(fig_remote)

# Filtrage avancé
st.subheader("🔎 Filtrage avancé")
exp_levels = st.multiselect("Sélectionnez le niveau d'expérience", df["experience_level"].unique())
company_sizes = st.multiselect("Sélectionnez la taille d'entreprise", df["company_size"].unique())

filtered_df = df.copy()
if exp_levels:
    filtered_df = filtered_df[filtered_df["experience_level"].isin(exp_levels)]
if company_sizes:
    filtered_df = filtered_df[filtered_df["company_size"].isin(company_sizes)]

st.write(filtered_df)
