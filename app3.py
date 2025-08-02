# -*- coding: utf-8 -*-
"""
Restaurant AI Toolkit - Streamlit Web Application (Enhanced Version)

This Streamlit app integrates four machine learning models and analyses for a restaurant dataset:
1.  Predicting restaurant aggregate ratings with model comparison.
2.  An advanced restaurant recommendation system.
3.  In-depth cuisine classification.
4.  Detailed geographical analysis and insights.
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- App Configuration and Styling ---
st.set_page_config(
    page_title="Restaurant AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Polished UI and Improved Design (Dark Theme)
def inject_custom_css():
    css = """
    <style>
        /* Main app background */
        body {
            color: #f1f1f1;
        }
        .main {
            background-color: #111111 !important;
            padding-top: 2rem;
            padding-bottom: 3rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #222222 !important;
            border-right: 2px solid #333333 !important;
            padding-top: 1.5rem;
        }

        /* Ensure all text in sidebar is readable */
        [data-testid="stSidebar"] * {
            color: #f1f1f1 !important;
        }

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-weight: 700 !important;
            color: #2ecc71 !important;
        }
        
        /* Buttons styling */
        .stButton>button {
            border: 2px solid #2ecc71 !important;
            background-color: #2ecc71 !important;
            color: white !important;
            padding: 0.55em 1.3em !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 8px rgba(46, 204, 113, 0.2);
            font-size: 15px !important;
        }

        .stButton>button:hover {
            background-color: #222222 !important;
            color: #2ecc71 !important;
            border-color: #2ecc71 !important;
            box-shadow: 0 0 10px #2ecc71 !important;
        }

        /* Metric styling */
        .stMetric {
            background-color: #222222 !important;
            border: 1.5px solid #333333 !important;
            border-radius: 14px !important;
            padding: 18px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
            text-align: center;
        }
        
        .stMetric > div > div > div {
             color: #f1f1f1 !important;
        }
        
        .stMetric > label {
            color: #a0a0b0 !important;
        }

        /* Headers and subtitles */
        h1 {
            font-weight: 900 !important;
            color: #2ecc71 !important;
        }

        h2 {
            color: #f1f1f1 !important;
            font-weight: 700 !important;
        }
        
        h3 {
             color: #e0e0e0 !important;
        }

        /* Dataframe styling */
        .stDataFrame {
            background-color: #222222;
        }
        .dataframe tbody tr:hover {
            background-color: #333333 !important;
        }

        /* Expander styling */
        .st-expander, .st-expander header {
            background-color: #222222 !important;
            color: #f1f1f1 !important;
        }
        button[aria-expanded="true"] {
            font-weight: 700 !important;
            color: #2ecc71 !important;
        }
        
        /* Info box styling */
        .stAlert {
            background-color: #263238 !important;
            border-radius: 12px !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(filepath):
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv(filepath)
        df['Cuisines'].fillna('Not Specified', inplace=True)
        # Convert boolean 'Yes'/'No' to 1/0 for easier processing
        le = LabelEncoder()
        for col in ['Has Table booking', 'Has Online delivery']:
            df[col] = le.fit_transform(df[col].astype(str))
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please ensure it's in the same directory as the script.")
        return None

@st.cache_resource
def train_rating_models(data):
    """Trains multiple regression models for rating prediction."""
    df_task1 = data.copy()
    features = ['Average Cost for two', 'Price range', 'Votes', 'Has Table booking', 'Has Online delivery']
    target = 'Aggregate rating'

    X = df_task1[features]
    y = df_task1[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models, X.columns, X_test, y_test

@st.cache_resource
def get_recommendation_system(data):
    """Builds the TF-IDF matrix and cosine similarity for recommendations."""
    df_task2 = data.copy()
    tfidf = TfidfVectorizer(stop_words='english')
    # Combine key features for a richer recommendation profile
    df_task2['Recommendation_Profile'] = df_task2['Cuisines'] + ' ' + df_task2['Locality']
    tfidf_matrix = tfidf.fit_transform(df_task2['Recommendation_Profile'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df_task2.index, index=df_task2['Restaurant Name']).drop_duplicates()
    return cosine_sim, indices, df_task2

@st.cache_resource
def train_cuisine_model(data):
    """Trains the cuisine classification model."""
    df_task3 = data.copy()
    df_task3['Primary Cuisine'] = df_task3['Cuisines'].apply(lambda x: x.split(',')[0])
    
    top_cuisines_series = df_task3['Primary Cuisine'].value_counts().nlargest(10)
    top_cuisines_index = top_cuisines_series.index
    df_filtered = df_task3[df_task3['Primary Cuisine'].isin(top_cuisines_index)]

    features = ['Average Cost for two', 'Price range', 'Votes', 'Aggregate rating']
    target = 'Primary Cuisine'
    
    X = df_filtered[features]
    y = df_filtered[target]

    le_cuisine = LabelEncoder()
    y_encoded = le_cuisine.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    
    rf_classifier = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)
    
    return rf_classifier, le_cuisine, X.columns, X_test, y_test, top_cuisines_series

# --- Main Application Logic ---
def main():
    inject_custom_css()
    df = load_data('Dataset .csv')

    if df is not None:
        st.sidebar.title("🍽️ Restaurant AI")
        st.sidebar.markdown("---")
        app_mode = st.sidebar.radio(
            "Select a Task",
            [
                "🏠 Introduction",
                "⭐ 1. Rating Prediction",
                "🧑‍🍳 2. Restaurant Recommender",
                "🍜 3. Cuisine Classifier",
                "🗺️ 4. Geographical Analysis"
            ]
        )
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This app demonstrates various ML models for restaurant data analysis.",
            icon="🤖"
        )

        if app_mode == "🏠 Introduction":
            page_introduction(df)
        elif app_mode == "⭐ 1. Rating Prediction":
            page_rating_prediction(df)
        elif app_mode == "🧑‍🍳 2. Restaurant Recommender":
            page_recommendation(df)
        elif app_mode == "🍜 3. Cuisine Classifier":
            page_cuisine_classification(df)
        elif app_mode == "🗺️ 4. Geographical Analysis":
            page_geo_analysis(df)

def page_introduction(df):
    st.title("Welcome to the Restaurant AI! ✨")
    st.markdown("""
    This interactive application provides a suite of machine learning tools to explore a rich restaurant dataset. 
    Each task below offers unique insights, from predicting ratings to discovering geographical patterns.

    **Navigate through the different models using the sidebar on the left.**
    """)
    
    with st.expander("📂 Click here to see a preview of the dataset"):
        st.dataframe(df.head(), use_container_width=True)
        st.markdown(f"The dataset contains **{df.shape[0]:,}** records and **{df.shape[1]}** features.")

    st.subheader("What can you do here?")
    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.info("**⭐ Predict Ratings:** Use various features to predict a restaurant's aggregate rating.")
        st.info("**🧑‍🍳 Get Recommendations:** Find new restaurants based on your favorite places.")
    with col2:
        st.info("**🍜 Classify Cuisines:** Discover which features are key in identifying a restaurant's cuisine.")
        st.info("**🗺️ Analyze Locations:** Visualize the global and local distribution of restaurants.")

def page_rating_prediction(df):
    st.title("⭐ 1. Aggregate Rating Prediction")
    st.markdown("Predict a restaurant's rating and understand which features are most influential.")
    
    models, feature_names, X_test, y_test = train_rating_models(df)

    st.sidebar.header("Prediction Inputs")
    model_choice = st.sidebar.selectbox("Choose a Prediction Model", list(models.keys()))
    
    avg_cost = st.sidebar.slider("Average Cost for Two", int(df['Average Cost for two'].min()), 2000, int(df['Average Cost for two'].median()), step=50)
    price_range = st.sidebar.select_slider("Price Range", sorted(df['Price range'].unique()), value=2)
    votes = st.sidebar.slider("Number of Votes", int(df['Votes'].min()), int(df['Votes'].max()), 500, step=10)
    has_booking = st.sidebar.radio("Has Table Booking?", ('Yes', 'No'), horizontal=True)
    has_delivery = st.sidebar.radio("Has Online Delivery?", ('Yes', 'No'), horizontal=True)

    input_data = pd.DataFrame({
        'Average Cost for two': [avg_cost], 'Price range': [price_range], 'Votes': [votes],
        'Has Table booking': [1 if has_booking == 'Yes' else 0],
        'Has Online delivery': [1 if has_delivery == 'Yes' else 0]
    })

    model = models[model_choice]
    predicted_rating = model.predict(input_data)[0]
    predicted_rating = max(0, min(predicted_rating, 5.0))

    st.subheader(f"📈 Predicted Rating using {model_choice}")
    st.metric(label="The model predicts a rating of:", value=f"{predicted_rating:.2f} / 5.0")
    
    st.markdown("---")
    
    with st.expander("🔬 Click to see Model Performance & Details"):
        st.subheader(f"Performance of {model_choice}")
        y_pred = model.predict(X_test)
        col1, col2 = st.columns(2, gap='medium')
        col1.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.4f}")
        col2.metric("R-squared (R2 Score)", f"{r2_score(y_test, y_pred):.4f}")

        if model_choice == "Decision Tree":
            st.subheader("Feature Importances")
            importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
            importances = importances.sort_values('importance', ascending=True)
            fig = px.bar(importances, x='importance', y='feature', orientation='h',
                         title="Feature Importance for Rating Prediction",
                         labels={'importance':'Importance', 'feature':'Feature'},
                         color='importance', color_continuous_scale='Emrld', template='plotly_dark')
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Actual vs. Predicted Ratings")
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.regplot(x=y_test, y=y_pred, ax=ax, scatter_kws={'alpha':0.4}, line_kws={'color':'#2ecc71'})
            ax.set_xlabel("Actual Ratings", fontsize=12, fontweight='bold')
            ax.set_ylabel("Predicted Ratings", fontsize=12, fontweight='bold')
            ax.set_title("Regression Plot", fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            st.pyplot(fig, clear_figure=True)

def page_recommendation(df):
    st.title("🧑‍🍳 2. Restaurant Recommendation System")
    st.markdown("Discover new dining experiences! Select a restaurant to get recommendations for similar places based on **cuisine and locality**.")
    
    cosine_sim, indices, df_task2 = get_recommendation_system(df)
    
    restaurant_list = sorted(df['Restaurant Name'].unique())
    default_restaurant = 'Le Petit Souffle' if 'Le Petit Souffle' in restaurant_list else restaurant_list[0]
    selected_restaurant = st.selectbox("Choose a Restaurant", restaurant_list, index=restaurant_list.index(default_restaurant))

    if st.button("Get Recommendations!", type="primary"):
        try:
            selected_info = df_task2[df_task2['Restaurant Name'] == selected_restaurant].iloc[0]
            st.markdown(f"### You selected: {selected_info['Restaurant Name']}")
            st.markdown(f"**Cuisines:** {selected_info['Cuisines']}")
            st.markdown(f"**Location:** {selected_info['Locality']}, {selected_info['City']}")
            st.markdown("---")

            idx = indices[selected_restaurant]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            
            restaurant_indices = [i[0] for i in sim_scores]
            
            recommendations = df_task2.iloc[restaurant_indices][['Restaurant Name', 'Cuisines', 'City', 'Locality', 'Aggregate rating']]
            
            st.subheader(f"Top 10 Recommendations for You:")
            st.dataframe(recommendations.style.background_gradient(cmap='BuGn', subset=['Aggregate rating']), use_container_width=True)

        except (KeyError, IndexError):
            st.error(f"Could not process '{selected_restaurant}'. This can happen with duplicate names. Please try another.")

def page_cuisine_classification(df):
    st.title("🍜 3. Cuisine Classifier")
    st.markdown("This model predicts a restaurant's primary cuisine from its cost, price, votes, and rating.")
    
    rf_classifier, le_cuisine, feature_names, X_test, y_test, top_cuisines = train_cuisine_model(df)

    col1, col2 = st.columns([1, 2], gap='large')
    with col1:
        st.subheader("Classifier Inputs")
        avg_cost_c = st.slider("Average Cost for Two", int(df['Average Cost for two'].min()), 2000, 500, step=25, key="c_cost")
        price_range_c = st.select_slider("Price Range", sorted(df['Price range'].unique()), value=2, key="c_price")
        votes_c = st.slider("Number of Votes", int(df['Votes'].min()), int(df['Votes'].max()), 500, step=10, key="c_votes")
        rating_c = st.slider("Aggregate Rating", 0.0, 5.0, 4.0, 0.1, key="c_rating")
        
        input_data_c = pd.DataFrame({
            'Average Cost for two': [avg_cost_c], 'Price range': [price_range_c],
            'Votes': [votes_c], 'Aggregate rating': [rating_c]
        })
        
        predicted_cuisine_encoded = rf_classifier.predict(input_data_c)[0]
        predicted_cuisine_proba = rf_classifier.predict_proba(input_data_c)
        predicted_cuisine = le_cuisine.inverse_transform([predicted_cuisine_encoded])[0]

        st.subheader("🔮 Predicted Cuisine")
        st.metric(label="The model predicts the primary cuisine is:", value=predicted_cuisine)

        proba_df = pd.DataFrame({
            'Cuisine': le_cuisine.classes_,
            'Probability': predicted_cuisine_proba[0]
        }).sort_values('Probability', ascending=False).head(3)
        st.markdown("**Top 3 predicted cuisines probabilities:**")
        for _, row in proba_df.iterrows():
            st.progress(row['Probability'], text=f"{row['Cuisine']}: {row['Probability']:.1%}")

    with col2:
        st.subheader("Top 10 Cuisines in Dataset")
        fig = px.bar(top_cuisines, x=top_cuisines.index, y=top_cuisines.values,
                     labels={'x':'Cuisine', 'y':'Number of Restaurants'},
                     color=top_cuisines.values, color_continuous_scale='Emrld',
                     title="Most Common Primary Cuisines", template='plotly_dark')
        fig.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Click to see Detailed Model Performance"):
        st.subheader("Model Accuracy")
        y_pred = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=le_cuisine.classes_, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
        
        st.subheader("Confusion Matrix")
        with plt.style.context('dark_background'):
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le_cuisine.classes_, yticklabels=le_cuisine.classes_, ax=ax)
            ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted Cuisine', fontsize=14)
            ax.set_ylabel('True Cuisine', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig, clear_figure=True)

def page_geo_analysis(df):
    st.title("🗺️ 4. Geographical Analysis of Restaurants")
    st.markdown("Explore the global footprint of restaurants in the dataset.")

    st.subheader("Interactive World Map of Restaurants")
    st.markdown("Zoom in and hover over points to see details. Points are colored by rating and sized by votes.")
    fig_map = px.scatter_geo(df, lat='Latitude', lon='Longitude', color='Aggregate rating',
                             hover_name='Restaurant Name', size='Votes', projection="natural earth",
                             title="Global Restaurant Distribution", color_continuous_scale=px.colors.sequential.Viridis,
                             size_max=15, opacity=0.8, template='plotly_dark')
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.subheader("Deep Dive into City-Level Analytics")

    city_counts = df['City'].value_counts()
    cities_to_analyze = city_counts[city_counts >= 20].index
    df_filtered = df[df['City'].isin(cities_to_analyze)]

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.markdown("**Top Cities by Restaurant Count**")
        top_cities = city_counts.nlargest(20)
        fig_city = px.bar(top_cities, y=top_cities.index, x=top_cities.values, orientation='h',
                          labels={'x':'Number of Restaurants', 'y':'City'}, color=top_cities.values,
                          color_continuous_scale='Viridis', title="Cities with Most Restaurants", template='plotly_dark')
        fig_city.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_city, use_container_width=True)

    with col2:
        st.markdown("**Average Rating by City**")
        avg_rating_by_city = df_filtered.groupby('City')['Aggregate rating'].mean().sort_values()
        fig_rating = px.bar(avg_rating_by_city, y=avg_rating_by_city.index, x=avg_rating_by_city.values,
                            orientation='h', labels={'x':'Average Rating', 'y':'City'},
                            color=avg_rating_by_city.values, color_continuous_scale='RdYlGn', title="Average Ratings per City", template='plotly_dark')
        fig_rating.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_rating, use_container_width=True)

if __name__ == "__main__":
    main()
