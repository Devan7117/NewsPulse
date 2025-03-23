import streamlit as st
import requests
import subprocess
import threading
import time
import plotly.express as px
import plotly.graph_objects as go  # Added for better visualizations
from utils import analyze_sentiment, extract_topics, generate_key_insights, text_to_speech
import os

# Function to start the FastAPI backend
def start_fastapi():
    subprocess.run(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])

# Start the FastAPI backend in a separate thread
threading.Thread(target=start_fastapi, daemon=True).start()

# Function to call the backend API with retries
def call_backend_with_retries(api_url, payload, max_retries=5, delay=2):
    """
    Calls the backend API with retries in case of connection errors.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                st.info(f"Connection error. Retrying in {delay} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                # Handle case where no articles were found
                st.error(f"No news articles found for '{payload['company_name']}'. Try a different company name.")
                return None
            else:
                # Re-raise other HTTP errors
                raise e

# Custom CSS for a more professional look
def load_css():
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stApp {
        background-color: black;  /* Set background to black */
        color: white;  /* Set default text color to white */
    }
    h1 {
        color: #1e3a8a;
        margin-bottom: 1.5rem;
    }
    h2, h3, h4 {
        color: #2563eb;
        margin-top: 1.5rem;
    }
    .sentiment-positive {
        color: #059669;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc2626;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6b7280;
        font-weight: bold;
    }
    .article-card {
        background-color: #333;  /* Darker background for cards */
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;  /* Set text color to white */
    }
    .topic-badge {
        display: inline-block;
        background-color: #444;  /* Darker background for badges */
        color: #fff;  /* Set text color to white */
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    .info-box {
        background-color: #222;  /* Darker background for info boxes */
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: white;  /* Set text color to white */
    }
    </style>
    """, unsafe_allow_html=True)

# Create a more visually appealing sentiment distribution chart
def create_sentiment_chart(sentiment_distribution):
    labels = list(sentiment_distribution.keys())
    values = list(sentiment_distribution.values())
    
    colors = {
        'Positive': '#059669',
        'Negative': '#dc2626',
        'Neutral': '#6b7280'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=[colors.get(label, '#3b82f6') for label in labels]
    )])
    
    fig.update_layout(
        title_text="Sentiment Distribution",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        width=500,
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

# Better loading spinner
def show_loading_animation():
    with st.spinner("Analyzing news articles..."):
        # Add a progress bar for better UX during loading
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.03)  # Simulate loading time
            progress_bar.progress(i + 1)

# Streamlit App
def main():
    # Set page title and description
    st.set_page_config(
        page_title="News Sentiment Analyzer",
        page_icon="📰",
        layout="wide"
    )
    
    # Load custom CSS
    load_css()
    
    # App header with better styling
    st.markdown("<h1>📰 News Summarization and Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This app analyzes news articles related to a given company, performs sentiment analysis, extracts key topics, 
    and generates a comprehensive report with a Hindi audio summary. Enter the company name below and click <b>Analyze</b> to get started.
    </div>
    """, unsafe_allow_html=True)

    # Create a cleaner layout with columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input field with improved styling
        company_name = st.text_input("Enter Company Name", placeholder="e.g., Tesla, Apple, Microsoft", help="Enter the name of the company you want to analyze")
    
    with col2:
        # Analysis button with better positioning
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)

    # Store the analysis results in session state to persist between reruns
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None

    if analyze_button:
        if company_name:
            # Show loading animation
            show_loading_animation()
            
            # Call the backend API
            api_url = "http://localhost:8000/analyze-news"
            payload = {
                "company_name": company_name,
                "api_key": st.secrets["GNEWS_API_KEY"] if "GNEWS_API_KEY" in st.secrets else os.environ.get("GNEWS_API_KEY", "your_api_key_here")
            }

            try:
                # Use the retry mechanism to call the backend API
                data = call_backend_with_retries(api_url, payload)
                if data:
                    # Store results in session state
                    st.session_state.analysis_data = data
                    # Show success message
                    st.success(f"✅ Successfully analyzed news for {company_name}!")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please check your internet connection and make sure your API key is correct.")
        else:
            st.warning("⚠️ Please enter a company name.")

    # Display analysis results if available
    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        
        # 1. Display Articles
        st.markdown("<h2>📑 Articles</h2>", unsafe_allow_html=True)
        
        # Create tabs for better organization
        tabs = st.tabs([f"Article {i+1}" for i in range(len(data['Articles']))])
        
        for i, (tab, article) in enumerate(zip(tabs, data['Articles'])):
            with tab:
                # Get sentiment class for styling
                sentiment_class = f"sentiment-{article['Sentiment'].lower()}"
                
                # Article card with improved styling
                st.markdown(f"""
                <div class="article-card">
                    <h3>{article['Title']}</h3>
                    <p><b>Summary:</b> {article['Summary']}</p>
                    <p><b>Sentiment:</b> <span class="{sentiment_class}">{article['Sentiment']}</span></p>
                    <p><b>Topics:</b> {''.join(f'<span class="topic-badge">{topic}</span>' for topic in article['Topics'])}</p>
                    <p><b>Read full article:</b> <a href="{article['URL']}" target="_blank">Click here</a></p>
                </div>
                """, unsafe_allow_html=True)
        
        # 2. Display Sentiment Overview
        st.markdown("<h2>📈 Sentiment Overview</h2>", unsafe_allow_html=True)
        
        # Use columns for better layout
        chart_col, info_col = st.columns([1, 2])
        
        with chart_col:
            sentiment_distribution = data['Comparative Sentiment Score']['Sentiment Distribution']
            fig = create_sentiment_chart(sentiment_distribution)
            st.plotly_chart(fig)
        
        with info_col:
            st.markdown("<h3>Coverage Differences</h3>", unsafe_allow_html=True)
            for difference in data['Comparative Sentiment Score']['Coverage Differences']:
                st.markdown(f"<b>Comparison:</b> {difference['Comparison']}", unsafe_allow_html=True)
                st.markdown(f"<b>Impact:</b> {difference['Impact']}", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
        
        # 3. Display Topic Analysis
        st.markdown("<h2>🔍 Topic Analysis</h2>", unsafe_allow_html=True)
        topic_overlap = data['Comparative Sentiment Score']['Topic Overlap']
        
        # Display common topics
        common_topics = topic_overlap['Common Topics']
        st.markdown("<h3>Common Topics Across Articles</h3>", unsafe_allow_html=True)
        st.markdown(''.join(f'<span class="topic-badge">{topic}</span>' for topic in common_topics), unsafe_allow_html=True)
        
        # Display unique topics for each article
        st.markdown("<h3>Unique Topics by Article</h3>", unsafe_allow_html=True)
        for i, unique_topics in enumerate(topic_overlap['Unique Topics']):
            topic_badges = ""
            for topic in unique_topics:
                topic_badges += f'<span class="topic-badge">{topic}</span>'
            st.markdown(f"<b>Article {i + 1}:</b> {topic_badges}", unsafe_allow_html=True)
        
        # 4. Display Final Sentiment Analysis
        st.markdown("<h2>📊 Final Sentiment Analysis</h2>", unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{data["Final Sentiment Analysis"]}</div>', unsafe_allow_html=True)
        
        # 5. Play Hindi TTS
        st.markdown("<h2>🔊 Hindi Summary</h2>", unsafe_allow_html=True)
        st.audio(data['Audio'])
        st.caption("Hindi audio summary of the sentiment analysis. Click to play.")

    # Footer
    st.markdown("---")
    st.markdown("Developed by [Devan](https://github.com/Devan7117) | [Copyrights © 2025 All Rights Reserved by Devan]")

if __name__ == "__main__":
    main()