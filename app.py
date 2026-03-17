import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------
# Load the cleaned dataset
# ------------------------------
df = pd.read_csv("cleaned_suicide_data.csv")

# ------------------------------
# App Title
# ------------------------------
st.set_page_config(page_title="Global Suicide Trends Explorer", layout="wide")
st.title("🌍 Global Suicide Trends Explorer")
st.write("Explore suicide trends across countries, genders, and years.")

# ------------------------------
# Sidebar Filters
# ------------------------------
year = st.sidebar.selectbox("Select Year", sorted(df['year'].unique()))
country = st.sidebar.selectbox("Select Country", sorted(df['country'].unique()))
sex_filter = st.sidebar.selectbox("Select Gender", ["Both", "male", "female"])
age_filter = st.sidebar.selectbox("Select Age Group", ["All"] + df['age'].unique().tolist())

# ------------------------------
# Filter Data Based on User Input
# ------------------------------
filtered_df = df[df['year'] == year]

if country != "All":
    filtered_df = filtered_df[filtered_df['country'] == country]
if sex_filter != "Both":
    filtered_df = filtered_df[filtered_df['sex'] == sex_filter]
if age_filter != "All":
    filtered_df = filtered_df[filtered_df['age'] == age_filter]

# ------------------------------
# Handle Missing Data Case
# ------------------------------
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected country, year, or filters.")
else:
    # ------------------------------
    # Summary Metrics
    # ------------------------------
    st.header("📊 Summary Metrics")
    total_suicides = int(filtered_df['suicides_no'].sum())
    avg_rate = round(filtered_df['suicides/100k pop'].mean(), 2)
    st.metric("Total Suicides", total_suicides)
    st.metric("Average Suicide Rate per 100k", avg_rate)

    # ------------------------------
    # Global Trend Over Years
    # ------------------------------
    st.header("1️⃣ Global Suicide Rate Trend Over Years")
    trend_df = df.groupby('year')['suicides/100k pop'].mean().reset_index()
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=trend_df, x='year', y='suicides/100k pop', marker="o", ax=ax1)
    ax1.set_ylabel("Suicides per 100k")
    st.pyplot(fig1)

    # ------------------------------
    # Gender Comparison
    # ------------------------------
    st.header("2️⃣ Suicide Rate by Gender")
    gender_df = df.groupby(['year', 'sex'])['suicides/100k pop'].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=gender_df, x='year', y='suicides/100k pop', hue='sex', marker="o", ax=ax2)
    ax2.set_ylabel("Suicides per 100k")
    st.pyplot(fig2)

    # ------------------------------
    # Age Group Analysis
    # ------------------------------
    st.header("3️⃣ Suicide Rate by Age Group")
    age_df = df.groupby('age')['suicides/100k pop'].mean().reset_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(data=age_df, x='age', y='suicides/100k pop', ax=ax3)
    ax3.set_ylabel("Suicides per 100k")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # ------------------------------
    # Country-wise Analysis
    # ------------------------------
    st.header("4️⃣ Top 10 Countries by Average Suicide Rate")
    top_countries = df.groupby('country')['suicides/100k pop'].mean().sort_values(ascending=False).head(10).reset_index()
    fig4, ax4 = plt.subplots()
    sns.barplot(data=top_countries, x='country', y='suicides/100k pop', palette="magma", ax=ax4)
    ax4.set_ylabel("Suicides per 100k")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    # ------------------------------
    # GDP vs Suicide Rate Regression
    # ------------------------------
    st.header("5️⃣ GDP per Capita vs Suicide Rate")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=df, x='gdp_per_capita ($)', y='suicides/100k pop', ax=ax5)
    # Regression line
    X = df[['gdp_per_capita ($)']]
    y = df['suicides/100k pop']
    model = LinearRegression()
    model.fit(X, y)
    ax5.plot(df['gdp_per_capita ($)'], model.predict(X), color='red')
    ax5.set_ylabel("Suicides per 100k")
    st.pyplot(fig5)

