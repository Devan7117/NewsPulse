from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
from utils import analyze_sentiment, extract_topics, generate_key_insights, text_to_speech
import logging
import time
from typing import List, Dict, Any, Optional
import os

# Set up logging to track what's happening in the app
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="News Sentiment Analysis API",
    description="An API for analyzing news sentiment and generating insights",
    version="1.0.0"
)

# Allow cross-origin requests (CORS) so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define data models using Pydantic for request and response validation

# Model for the request body when analyzing news
class CompanyRequest(BaseModel):
    company_name: str = Field(..., description="Name of the company to analyze")
    api_key: str = Field(..., description="API key for GNews")
    language: Optional[str] = Field("en", description="Language for news articles (default: English)")
    max_articles: Optional[int] = Field(10, description="Maximum number of articles to retrieve", ge=1, le=20)

# Model for an individual article
class Article(BaseModel):
    title: str
    summary: str
    sentiment: str
    topics: List[str]
    url: str

# Model for sentiment distribution (positive, negative, neutral counts)
class SentimentDistribution(BaseModel):
    Positive: int
    Negative: int
    Neutral: int

# Model for comparing coverage differences between articles
class ComparisonItem(BaseModel):
    Comparison: str
    Impact: str

# Model for topic overlap between articles
class TopicOverlap(BaseModel):
    Common_Topics: List[str]
    Unique_Topics: List[List[str]]

# Model for the comparative sentiment score
class ComparativeSentimentScore(BaseModel):
    Sentiment_Distribution: SentimentDistribution
    Coverage_Differences: List[ComparisonItem]
    Topic_Overlap: dict

# Model for the final analysis response
class AnalysisResponse(BaseModel):
    Company: str
    Articles: List[Article]
    Comparative_Sentiment_Score: ComparativeSentimentScore
    Final_Sentiment_Analysis: str
    Audio: str

# Simple rate limiting to prevent too many requests from the same IP
last_request_time = {}
RATE_LIMIT_SECONDS = 5  # Minimum seconds between requests from the same IP

# Middleware to enforce rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    # Check if the IP has made a request recently
    if client_ip in last_request_time:
        time_since_last_request = current_time - last_request_time[client_ip]
        if time_since_last_request < RATE_LIMIT_SECONDS:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Please wait {RATE_LIMIT_SECONDS} seconds between requests."}
            )
    
    # Update the last request time for this IP
    last_request_time[client_ip] = current_time
    response = await call_next(request)
    return response

# Function to fetch news articles using the GNews API
def fetch_news(company_name, api_key, language="en", max_articles=10):
    """
    Fetches news articles related to the given company name using the GNews API.
    """
    logger.info(f"Fetching news articles for company: {company_name}")
    url = f"https://gnews.io/api/v4/search?q={company_name}&lang={language}&max={max_articles}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)  # Add timeout to prevent hanging
        response.raise_for_status()
        data = response.json()
        logger.info(f"GNews API response received for {company_name}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch news articles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch news articles: {str(e)}")

    articles = []
    seen_titles = set()

    if 'articles' in data and data['articles']:
        for article_data in data['articles']:
            title = article_data.get('title', 'No Title')
            if title not in seen_titles:
                seen_titles.add(title)
                summary = article_data.get('description', 'No Summary')
                content = article_data.get('content', 'No Content')
                article_url = article_data.get('url', '')
                articles.append({
                    "title": title,
                    "summary": summary,
                    "content": content,
                    "url": article_url  # Add URL to each article
                })
        logger.info(f"Found {len(articles)} articles for {company_name}")
    else:
        logger.warning("No articles found for the given company name.")
        raise HTTPException(status_code=404, detail="No articles found for the given company name.")

    return articles

# Function to generate coverage differences between articles
def generate_coverage_differences(articles):
    """
    Generates meaningful coverage differences and their impact between articles.
    """
    differences = []
    
    # Mapping of sentiment to descriptive insights
    sentiment_insights = {
        "POSITIVE": ["positive outlook", "growth potential", "successful initiatives", "market optimism"],
        "NEGATIVE": ["challenges", "concerns", "regulatory issues", "market skepticism", "public backlash"],
        "NEUTRAL": ["mixed reviews", "balanced perspective", "ongoing developments", "market assessment"]
    }
    
    # Mapping of topics to business areas
    topic_to_area = {
        "Financial Performance": "financial results",
        "Stock Market": "stock performance",
        "Innovation": "innovation strategy",
        "Technology": "technological advancements",
        "Product Line": "product offerings",
        "Electric Vehicles": "EV market position",
        "Autonomous Technology": "autonomous capabilities",
        "Regulatory Compliance": "regulatory landscape",
        "Executive Leadership": "leadership decisions",
        "Political Impact": "political influences",
        "Corporate Controversy": "public controversies",
        "Social Media Impact": "social media presence",
        "Public Protests": "protests against the company",
        "Public Backlash": "public criticism",
        "Brand Perception": "brand image",
        "Corporate Image": "corporate reputation",
        "Business Challenges": "operational challenges",
        "Organizational Challenges": "internal difficulties"
    }
    
    # Generate meaningful differences between consecutive articles
    for i in range(len(articles) - 1):
        article1 = articles[i]
        article2 = articles[i + 1]
        
        # Get main topics or use appropriate fallbacks
        topic1 = article1['topics'][0] if article1['topics'] else "general news"
        topic2 = article2['topics'][0] if article2['topics'] else "general news"
        
        # Get sentiment descriptors
        sentiment1_desc = sentiment_insights[article1['sentiment']][0]
        sentiment2_desc = sentiment_insights[article2['sentiment']][0]
        
        # Get business area descriptors
        area1 = topic_to_area.get(topic1, topic1.lower())
        area2 = topic_to_area.get(topic2, topic2.lower())
        
        # Generate meaningful comparison
        comparison = (
            f"Article {i + 1} focuses on {topic1} with a {article1['sentiment'].lower()} tone, "
            f"while Article {i + 2} highlights {topic2} with a {article2['sentiment'].lower()} perspective."
        )
        
        # Generate meaningful impact analysis based on sentiment alignment
        if article1['sentiment'] == article2['sentiment']:
            if article1['sentiment'] == "POSITIVE":
                impact = (
                    f"Both articles reinforce the {sentiment1_desc} regarding different aspects "
                    f"of the company ({area1} and {area2}), which strengthens the overall positive narrative."
                )
            elif article1['sentiment'] == "NEGATIVE":
                impact = (
                    f"Both articles emphasize {sentiment1_desc} in different areas "
                    f"({area1} and {area2}), which compounds the negative perception of the company."
                )
            else:  # NEUTRAL
                impact = (
                    f"Both articles present a {sentiment1_desc} on different aspects "
                    f"of the company ({area1} and {area2}), suggesting a period of stability or transition."
                )
        else:
            # Different sentiments
            if article1['sentiment'] == "POSITIVE" and article2['sentiment'] == "NEGATIVE":
                impact = (
                    f"The {sentiment1_desc} regarding {area1} contrasts with "
                    f"the {sentiment2_desc} about {area2}, presenting a complex picture where positive developments "
                    f"are overshadowed by emerging problems."
                )
            elif article1['sentiment'] == "NEGATIVE" and article2['sentiment'] == "POSITIVE":
                impact = (
                    f"The {sentiment1_desc} regarding {area1} are somewhat offset by "
                    f"the {sentiment2_desc} related to {area2}, suggesting the company may be addressing its issues "
                    f"or experiencing growth in some areas despite problems in others."
                )
            elif article1['sentiment'] == "NEUTRAL" and article2['sentiment'] != "NEUTRAL":
                impact = (
                    f"The {sentiment1_desc} on {area1} shifts to a more definitive "
                    f"{sentiment2_desc} regarding {area2}, indicating evolving market perception."
                )
            else:  # article2['sentiment'] == "NEUTRAL"
                impact = (
                    f"The clear {sentiment1_desc} regarding {area1} gives way to a more "
                    f"{sentiment2_desc} on {area2}, suggesting some uncertainty in recent developments."
                )
        
        differences.append({"Comparison": comparison, "Impact": impact})
    
    return differences

# Function to generate topic overlap between articles
def generate_topic_overlap(articles):
    """
    Generates comprehensive analysis of common and unique topics across articles.
    """
    # Extract all topics from all articles
    all_topics_sets = [set(article['topics']) for article in articles]
    
    # Find common topics across all articles
    if all_topics_sets:
        common_topics = set.intersection(*all_topics_sets) if all_topics_sets else set()
    else:
        common_topics = set()
    
    # For each article, find unique topics
    unique_topics_by_article = []
    for i, topics_set in enumerate(all_topics_sets):
        unique = topics_set - common_topics
        unique_topics_by_article.append(list(unique))
    
    return {
        "Common Topics": list(common_topics) if common_topics else ["No common topics found"],
        "Unique Topics": [
            list(topics) if topics else ["No unique topics"] 
            for topics in unique_topics_by_article
        ]
    }

# Function to generate the final sentiment analysis
def generate_final_sentiment(company_name, articles, sentiment_distribution):
    """
    Generates a comprehensive final sentiment analysis based on all articles.
    """
    # Determine overall sentiment trend
    if sentiment_distribution["Positive"] > sentiment_distribution["Negative"]:
        primary_sentiment = "positive"
    elif sentiment_distribution["Negative"] > sentiment_distribution["Positive"]:
        primary_sentiment = "negative"
    else:
        primary_sentiment = "mixed"
    
    # Extract key topics across all articles
    all_topics = []
    for article in articles:
        all_topics.extend(article['topics'])
    
    # Count topic frequency
    topic_count = {}
    for topic in all_topics:
        topic_count[topic] = topic_count.get(topic, 0) + 1
    
    # Get top 3 most mentioned topics
    top_topics = sorted(topic_count.items(), key=lambda x: x[1], reverse=True)[:3]
    top_topic_names = [topic[0] for topic in top_topics]
    
    # Generate specific insights based on sentiment and topics
    insights = []
    if primary_sentiment == "positive":
        insights.append(f"strong performance in {top_topic_names[0] if top_topic_names else 'key areas'}")
        if len(top_topic_names) > 1:
            insights.append(f"favorable outlook for {top_topic_names[1]}")
    elif primary_sentiment == "negative":
        insights.append(f"challenges related to {top_topic_names[0] if top_topic_names else 'various factors'}")
        if len(top_topic_names) > 1:
            insights.append(f"concerns regarding {top_topic_names[1]}")
    else:
        insights.append("conflicting signals in the market")
        
    # Add investment implications
    if primary_sentiment == "positive":
        market_outlook = "which may lead to potential growth in stock value and increased investor confidence"
    elif primary_sentiment == "negative":
        market_outlook = "which could result in cautious investor sentiment and potential short-term market volatility"
    else:
        market_outlook = "suggesting investors should monitor developments closely before making significant decisions"
    
    # Construct the final analysis
    final_analysis = (
        f"{company_name}'s latest news coverage is predominantly {primary_sentiment}, "
        f"highlighting {', '.join(insights)}, {market_outlook}."
    )
    
    return final_analysis

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify API is running.
    """
    return {"status": "healthy", "version": "1.0.0"}

# Documentation endpoint
@app.get("/docs-info")
def docs_info():
    """
    Provides information about the API and how to use it.
    """
    return {
        "API_Name": "News Sentiment Analysis API",
        "Description": "Analyzes news sentiment for a given company",
        "Endpoints": {
            "/analyze-news": {
                "method": "POST",
                "description": "Analyze news for a specific company",
                "body_parameters": ["company_name", "api_key", "language (optional)", "max_articles (optional)"],
                "sample_request": {
                    "company_name": "Tesla",
                    "api_key": "your_gnews_api_key",
                    "language": "en",
                    "max_articles": 10
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    }

# Main endpoint to analyze news
@app.post("/analyze-news", response_model=None)
async def analyze_news(request: CompanyRequest):
    """
    Fetches news articles, performs sentiment analysis, and generates a Hindi TTS summary.
    """
    company_name = request.company_name
    api_key = request.api_key
    language = request.language
    max_articles = request.max_articles

    start_time = time.time()
    logger.info(f"Starting analysis for company: {company_name}")

    try:
        # Fetch news articles
        articles = fetch_news(company_name, api_key, language, max_articles)

        # Perform sentiment analysis and topic extraction
        for article in articles:
            article['sentiment'] = analyze_sentiment(article['summary'])
            article['topics'] = extract_topics(article['content'])

        # Generate key insights in Hindi
        key_insights = generate_key_insights(articles)

        # Generate Hindi TTS
        positive_count = sum(1 for a in articles if a['sentiment'] == 'POSITIVE')
        negative_count = sum(1 for a in articles if a['sentiment'] == 'NEGATIVE')
        neutral_count = sum(1 for a in articles if a['sentiment'] == 'NEUTRAL')
        
        dominant_sentiment = max(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], 
                                key=lambda x: sum(1 for a in articles if a['sentiment'] == x))
        
        hindi_summary = (
            f"{company_name} की खबरों का विश्लेषण पूरा हो गया है। "
            f"कुल {len(articles)} खबरों में से {positive_count} खबरें सकारात्मक, "
            f"{negative_count} खबरें नकारात्मक, और {neutral_count} खबरें तटस्थ हैं। "
            f"ज्यादातर खबरें {dominant_sentiment} हैं। "
            f"{' '.join(key_insights)}"
        )
        tts_file = text_to_speech(hindi_summary)

        # Generate comparative sentiment score
        sentiment_distribution = {
            "Positive": positive_count,
            "Negative": negative_count,
            "Neutral": neutral_count
        }

        # Generate coverage differences
        coverage_differences = generate_coverage_differences(articles)

        # Generate topic overlap
        topic_overlap = generate_topic_overlap(articles)

        # Generate final sentiment analysis
        final_sentiment_analysis = generate_final_sentiment(
            company_name, 
            articles, 
            sentiment_distribution
        )

        processing_time = time.time() - start_time
        logger.info(f"Analysis completed for {company_name} in {processing_time:.2f} seconds")

        return {
            "Company": company_name,
            "Articles": [{
                "Title": article['title'],
                "Summary": article['summary'],
                "Sentiment": article['sentiment'].capitalize(),
                "Topics": article['topics'],
                "URL": article['url']
            } for article in articles],
            "Comparative Sentiment Score": {
                "Sentiment Distribution": sentiment_distribution,
                "Coverage Differences": coverage_differences,
                "Topic Overlap": topic_overlap
            },
            "Final Sentiment Analysis": final_sentiment_analysis,
            "Audio": tts_file,
            "Processing_Time": f"{processing_time:.2f} seconds"
        }
    except Exception as e:
        logger.error(f"Error in analyze_news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)