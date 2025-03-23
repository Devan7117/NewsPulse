from keybert import KeyBERT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gtts import gTTS
import logging
import re
import nltk
from nltk.corpus import stopwords
import base64
import os
from io import BytesIO
import tempfile
import string
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Set up logging to track what's happening in the app
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define custom stopwords to ignore in text processing
CUSTOM_STOPWORDS = ["company", "said", "will", "also", "news", "has", "and", "the", "for", "with", "its", 
                    "had", "have", "been", "was", "were", "is", "are", "this", "that", "than",
                    "could", "would", "should", "may", "might", "must", "can", "cannot", "can't"]

# Cache the models for better performance
_keybert_model = None
_sentiment_analyzer = None
_summarizer = None

# Function to load the KeyBERT model
def load_keybert_model():
    """
    Loads and caches the KeyBERT model.
    Returns:
        KeyBERT: The cached KeyBERT model instance.
    """
    global _keybert_model
    if _keybert_model is None:
        logger.info("Loading KeyBERT model...")
        _keybert_model = KeyBERT()
    return _keybert_model

# Function to load the VADER sentiment analyzer
def get_sentiment_analyzer():
    """
    Loads and caches the VADER sentiment analyzer.
    Returns:
        SentimentIntensityAnalyzer: The cached sentiment analyzer instance.
    """
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        logger.info("Loading VADER sentiment analyzer...")
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer

# Function to load the summarization model
def get_summarizer():
    """
    Loads and caches the summarization model.
    Returns:
        pipeline: The cached summarization pipeline.
    """
    global _summarizer
    if _summarizer is None:
        try:
            logger.info("Loading summarization model...")
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            _summarizer = None
    return _summarizer

# Function to clean text by removing unwanted characters and formatting
def clean_text(text):
    """
    Cleans the input text by removing excess whitespace, special characters, etc.
    Args:
        text (str): The text to clean.
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    
    # Replace newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove trailing 'and'
    text = re.sub(r'\s+and\s*$', '', text)
    
    return text

# Function to analyze sentiment using VADER with improved context awareness
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using VADER with improved context awareness.
    Args:
        text (str): The text to analyze.
    Returns:
        str: The sentiment label (POSITIVE, NEGATIVE, or NEUTRAL).
    """
    if not text:
        return "NEUTRAL"
    
    # Clean the text
    text = clean_text(text)
    
    # Check for negative context keywords
    negative_contexts = [
        "backlash", "protest", "demonstration", "controversy", "criticism", "concern", 
        "issue", "problem", "challenge", "turmoil", "trouble", "crisis", "negative", 
        "fall", "drop", "decline", "lawsuit", "investigation", "recall", "fail",
        "struggling", "layoff", "downgrade", "loss", "bankrupt", "fraud", "scandal",
        "fine", "penalty", "warning", "crash", "accident", "delay", "cancel"
    ]
    
    # Check for positive context keywords
    positive_contexts = [
        "success", "growth", "increase", "improve", "innovation", "positive", "opportunity",
        "breakthrough", "achievement", "profit", "gain", "advance", "progress", "excel",
        "surpass", "exceed", "outperform", "award", "recognition", "partnership", "collaboration",
        "launch", "expansion", "milestone", "record", "recovery", "rebound", "upgrade"
    ]
    
    # Get VADER scores
    analyzer = get_sentiment_analyzer()
    sentiment_score = analyzer.polarity_scores(text)
    
    # Adjust sentiment based on context keywords
    text_lower = text.lower()
    negative_count = sum(1 for word in negative_contexts if word in text_lower)
    positive_count = sum(1 for word in positive_contexts if word in text_lower)
    
    # Adjust compound score based on context
    context_adjustment = (positive_count - negative_count) * 0.1
    adjusted_score = sentiment_score['compound'] + context_adjustment
    
    # Apply additional context rules
    if any(word in text_lower for word in ["protest", "backlash", "demonstration", "layoff", "scandal"]) and sentiment_score['compound'] > -0.1:
        # Force negative sentiment for protests/backlash regardless of other positive terms
        logger.info(f"Forcing negative sentiment due to presence of negative context terms: {text[:100]}...")
        return "NEGATIVE"
    
    # Handle financial context specifically
    if any(term in text_lower for term in ["revenue", "profit", "earnings", "stock"]):
        if any(term in text_lower for term in ["increase", "rise", "grew", "growth", "up", "higher"]):
            if adjusted_score > -0.15:  # Lower threshold for financial good news
                logger.info(f"Financial positive context detected: {text[:100]}...")
                return "POSITIVE"
        elif any(term in text_lower for term in ["decrease", "fall", "drop", "down", "lower", "loss"]):
            if adjusted_score < 0.15:  # Higher threshold for financial bad news
                logger.info(f"Financial negative context detected: {text[:100]}...")
                return "NEGATIVE"
    
    # Apply standard threshold on adjusted score
    if adjusted_score >= 0.05:
        return "POSITIVE"
    elif adjusted_score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# Function to extract topics from text using KeyBERT
def extract_topics(text):
    """
    Extracts and generalizes topics from the given text using an improved KeyBERT approach.
    Args:
        text (str): The text to extract topics from.
    Returns:
        list: A list of generalized topics.
    """
    # If text is too short or empty, return general topics
    if not text or len(text) < 50:
        return ["General News", "Company Update"]
    
    # Clean text
    text = clean_text(text)
    
    # Extract company name patterns to avoid them becoming topics
    common_companies = ["Tesla", "Apple", "Microsoft", "Google", "Amazon", "Facebook", "Meta", "Twitter", "X"]
    extraction_stopwords = CUSTOM_STOPWORDS.copy()
    
    for company in common_companies:
        if company.lower() in text.lower():
            extraction_stopwords.append(company.lower())
    
    # Try to extract keywords
    try:
        kw_model = load_keybert_model()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),  # Focus on 1-2 word phrases for cleaner topics
            stop_words=extraction_stopwords,
            top_n=7  # Extract more candidates for better filtering
        )
    except Exception as e:
        logger.error(f"KeyBERT extraction failed: {e}")
        keywords = []
    
    # Comprehensive mapping of keywords to business topics
    topic_mapping = {
        # Financial Terms
        'sales': 'Sales Performance',
        'revenue': 'Financial Performance',
        'profit': 'Financial Performance',
        'earnings': 'Financial Performance',
        'quarter': 'Quarterly Results',
        'fiscal': 'Financial Performance',
        'stock': 'Stock Market',
        'shares': 'Stock Market',
        'investor': 'Investor Relations',
        'shareholder': 'Investor Relations',
        'market': 'Market Position',
        'valuation': 'Company Valuation',
        'dividend': 'Dividend Policy',
        'investment': 'Investment Strategy',
        'funding': 'Funding & Capital',
        
        # Technology & Innovation
        'technology': 'Technology',
        'innovation': 'Innovation',
        'ai': 'Artificial Intelligence',
        'artificial intelligence': 'Artificial Intelligence',
        'machine learning': 'AI & Machine Learning',
        'software': 'Software Development',
        'hardware': 'Hardware Products',
        'algorithm': 'Technology Development',
        'cloud': 'Cloud Computing',
        'data': 'Data Analytics',
        'chip': 'Semiconductor',
        'semiconductor': 'Semiconductor',
        
        # Products
        'product': 'Product Development',
        'model': 'Product Line',
        'launch': 'Product Launch',
        'release': 'Product Release',
        'feature': 'Product Features',
        'design': 'Product Design',
        'update': 'Product Updates',
        
        # Automotive Specific
        'electric': 'Electric Vehicles',
        'ev': 'Electric Vehicles',
        'vehicle': 'Automotive Industry',
        'car': 'Automotive Industry',
        'autonomous': 'Autonomous Technology',
        'self-driving': 'Autonomous Technology',
        'battery': 'Battery Technology',
        'charging': 'EV Infrastructure',
        
        # Legal & Regulatory
        'legal': 'Legal Issues',
        'lawsuit': 'Legal Disputes',
        'regulation': 'Regulatory Compliance',
        'compliance': 'Regulatory Compliance',
        'policy': 'Policy Impact',
        'rule': 'Regulatory Rules',
        'legislation': 'Legislation Impact',
        'copyright': 'Intellectual Property',
        'patent': 'Intellectual Property',
        'trademark': 'Intellectual Property',
        
        # Government & Geopolitical
        'government': 'Government Relations',
        'trade': 'Trade Relations',
        'tariff': 'International Trade',
        'sanction': 'International Sanctions',
        'geopolitical': 'Geopolitical Factors',
        'global': 'Global Strategy',
        'international': 'International Business',
        
        # Environmental & Social
        'security': 'Security Concerns',
        'privacy': 'Privacy Concerns',
        'environment': 'Environmental Impact',
        'sustainable': 'Sustainability',
        'green': 'Green Initiatives',
        'carbon': 'Carbon Footprint',
        'emission': 'Emissions Reduction',
        'esg': 'ESG Initiatives',
        'diversity': 'Diversity & Inclusion',
        'social': 'Social Responsibility',
        
        # Leadership & Organization
        'leadership': 'Corporate Leadership',
        'ceo': 'Executive Leadership',
        'executive': 'Executive Leadership',
        'management': 'Management Changes',
        'board': 'Board Decisions',
        'restructuring': 'Corporate Restructuring',
        'layoff': 'Workforce Reduction',
        'hiring': 'Talent Acquisition',
        'talent': 'Talent Management',
        
        # Political & Public Relations
        'political': 'Political Relations',
        'politics': 'Political Impact',
        'policy': 'Policy Decisions',
        'rebate': 'Financial Incentives',
        'tax': 'Tax Policy',
        'consumer': 'Consumer Sentiment',
        'criticism': 'Public Relations',
        'controversy': 'Corporate Controversy',
        'social media': 'Social Media Impact',
        'media': 'Media Coverage',
        
        # Brand & Reputation
        'brand': 'Brand Perception',
        'image': 'Corporate Image',
        'reputation': 'Brand Reputation',
        'customer': 'Customer Relations',
        'satisfaction': 'Customer Satisfaction',
        'loyalty': 'Customer Loyalty',
        'feedback': 'Customer Feedback',
        
        # Opposition & Challenges
        'demonstration': 'Public Protests',
        'protest': 'Public Protests',
        'backlash': 'Public Backlash',
        'takedown': 'Public Backlash',
        'boycott': 'Consumer Boycott',
        'critic': 'Corporate Critics',
        
        # Operations
        'showroom': 'Retail Operations',
        'store': 'Retail Operations',
        'production': 'Manufacturing',
        'factory': 'Manufacturing',
        'supply': 'Supply Chain',
        'logistics': 'Supply Chain & Logistics',
        'inventory': 'Inventory Management',
        
        # Challenges & Risk
        'turmoil': 'Organizational Challenges',
        'challenge': 'Business Challenges',
        'hurdle': 'Market Hurdles',
        'obstacle': 'Business Obstacles',
        'risk': 'Business Risks',
        'threat': 'Market Threats',
        'disruption': 'Market Disruption',
        'competitive': 'Competitive Pressure',
        'competition': 'Market Competition',
        
        # Growth & Strategy
        'growth': 'Growth Strategy',
        'expansion': 'Business Expansion',
        'acquisition': 'Mergers & Acquisitions',
        'merger': 'Mergers & Acquisitions',
        'partnership': 'Strategic Partnerships',
        'alliance': 'Business Alliances',
        'collaboration': 'Industry Collaboration',
        'joint venture': 'Joint Ventures',
        'strategy': 'Corporate Strategy',
        'vision': 'Corporate Vision',
        'mission': 'Corporate Mission',
        
        # Research & Development
        'research': 'Research & Development',
        'development': 'R&D Initiatives',
        'lab': 'Research Labs',
        'scientist': 'Research Team',
        'engineer': 'Engineering Team',
        'patent': 'Patent Portfolio',
        'intellectual property': 'Intellectual Property'
    }
    
    generalized_topics = []
    
    # First try to match keywords to our mapping
    if keywords:
        for keyword, score in keywords:
            # Skip if keyword is too short
            if len(keyword) <= 3:
                continue
                
            matched = False
            for key_term, mapped_topic in topic_mapping.items():
                if key_term in keyword.lower():
                    if mapped_topic not in generalized_topics:  # Avoid duplicates
                        generalized_topics.append(mapped_topic)
                        matched = True
                        break
            
            # If no match found and keyword is meaningful, add it with proper capitalization
            if not matched and len(keyword) > 3:
                # Capitalize each word properly
                proper_topic = ' '.join(word.capitalize() for word in keyword.split())
                if proper_topic not in generalized_topics:
                    generalized_topics.append(proper_topic)
    
    # If we couldn't extract meaningful topics, check for key terms in text
    if not generalized_topics:
        # Extract potential topics based on common business terms in the text
        text_lower = text.lower()
        for key_term, mapped_topic in topic_mapping.items():
            if key_term in text_lower and mapped_topic not in generalized_topics:
                generalized_topics.append(mapped_topic)
    
    # Add context-specific topics based on article content
    text_lower = text.lower()
    # Check for protests/demonstrations context
    if any(term in text_lower for term in ["protest", "demonstration", "backlash", "takedown"]):
        if "Public Protests" not in generalized_topics and "Public Backlash" not in generalized_topics:
            generalized_topics.append("Public Backlash")
    
    # Check for brand reputation context
    if any(term in text_lower for term in ["brand", "image", "love", "hard to love"]):
        if "Brand Perception" not in generalized_topics:
            generalized_topics.append("Brand Perception")
    
    # If still no topics, add general ones
    if not generalized_topics:
        generalized_topics = ["Company News", "Industry Update"]
    
    # Return unique topics, limit to 3
    return list(dict.fromkeys(generalized_topics))[:3]  # This preserves order while removing duplicates

# Function to generate key insights in Hindi
def generate_key_insights(articles):
    """
    Generates key insights from the articles based on their sentiment and topics.
    Args:
        articles (list): A list of articles with sentiment labels and topics.
    Returns:
        list: A list of key insights in Hindi.
    """
    if not articles:
        return ["कोई समाचार नहीं मिला।"]
    
    insights = []
    
    # Map English sentiments to Hindi
    sentiment_map = {
        "POSITIVE": "सकारात्मक",
        "NEGATIVE": "नकारात्मक",
        "NEUTRAL": "तटस्थ"
    }
    
    # Map common topics to Hindi
    topic_map = {
        "Financial Performance": "वित्तीय प्रदर्शन",
        "Stock Market": "शेयर बाजार",
        "Product Development": "उत्पाद विकास",
        "Innovation": "नवाचार",
        "Technology": "प्रौद्योगिकी",
        "Corporate Leadership": "कॉर्पोरेट नेतृत्व",
        "Market Competition": "बाजार प्रतिस्पर्धा",
        "Regulatory Compliance": "नियामक अनुपालन",
        "Business Expansion": "व्यापार विस्तार",
        "Sales Performance": "बिक्री प्रदर्शन",
        "Electric Vehicles": "इलेक्ट्रिक वाहन",
        "Autonomous Technology": "स्वायत्त तकनीक",
        "Public Backlash": "सार्वजनिक विरोध",
        "Environmental Impact": "पर्यावरण प्रभाव"
    }
    
    # Group articles by sentiment
    positive_articles = [a for a in articles if a['sentiment'] == "POSITIVE"]
    negative_articles = [a for a in articles if a['sentiment'] == "NEGATIVE"]
    neutral_articles = [a for a in articles if a['sentiment'] == "NEUTRAL"]
    
    # Add insight about dominant sentiment
    if len(positive_articles) > len(negative_articles) and len(positive_articles) > len(neutral_articles):
        insights.append(f"अधिकतर समाचार लेख सकारात्मक हैं, जो कंपनी के लिए अच्छा संकेत है।")
    elif len(negative_articles) > len(positive_articles) and len(negative_articles) > len(neutral_articles):
        insights.append(f"अधिकतर समाचार लेख नकारात्मक हैं, जो कंपनी के लिए चिंता का विषय हो सकता है।")
    else:
        insights.append(f"समाचार लेखों में मिश्रित राय है, जो कंपनी की वर्तमान स्थिति को दर्शाता है।")
    
    # Find common topics across articles
    all_topics = []
    for article in articles:
        all_topics.extend(article.get('topics', []))
    
    # Count topic frequency
    topic_count = {}
    for topic in all_topics:
        topic_count[topic] = topic_count.get(topic, 0) + 1
    
    # Add insight about most discussed topics
    if topic_count:
        top_topics = sorted(topic_count.items(), key=lambda x: x[1], reverse=True)[:2]
        hindi_topics = []
        for topic, _ in top_topics:
            hindi_topics.append(topic_map.get(topic, topic))
        
        topics_str = " और ".join(hindi_topics)
        insights.append(f"समाचारों में सबसे अधिक चर्चित विषय हैं: {topics_str}।")
    
    # Add specific insights based on sentiment and topics
    if positive_articles:
        positive_topics = []
        for article in positive_articles:
            if article.get('topics'):
                positive_topics.extend(article['topics'])
        
        if positive_topics:
            top_positive = max(set(positive_topics), key=positive_topics.count)
            hindi_topic = topic_map.get(top_positive, top_positive)
            insights.append(f"{hindi_topic} के बारे में सकारात्मक खबरें हैं।")
    
    if negative_articles:
        negative_topics = []
        for article in negative_articles:
            if article.get('topics'):
                negative_topics.extend(article['topics'])
        
        if negative_topics:
            top_negative = max(set(negative_topics), key=negative_topics.count)
            hindi_topic = topic_map.get(top_negative, top_negative)
            insights.append(f"{hindi_topic} के बारे में कुछ चिंताएं व्यक्त की गई हैं।")
    
    # Add a general conclusion
    if len(positive_articles) > len(negative_articles):
        insights.append("समग्र रूप से, कंपनी के बारे में समाचार कवरेज अनुकूल है।")
    elif len(negative_articles) > len(positive_articles):
        insights.append("समग्र रूप से, कंपनी के बारे में समाचार कवरेज चुनौतीपूर्ण है।")
    else:
        insights.append("समग्र रूप से, कंपनी के बारे में मिश्रित समाचार कवरेज है।")
    
    # Limit to 5 insights maximum
    return insights[:5]

# Function to convert text to speech in Hindi
def text_to_speech(text, language='hi'):
    """
    Converts the given text to Hindi speech and saves it as an audio file.
    Args:
        text (str): The text to convert to speech.
        language (str): The language of the text (default: 'hi' for Hindi).
    Returns:
        str: The path to the saved audio file.
    """
    if not text:
        logger.warning("Empty text provided for TTS conversion")
        return "No text to convert"
    
    try:
        logger.info(f"Generating TTS for text ({len(text)} chars)")
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Create temp dir if it doesn't exist
        temp_dir = "temp_audio"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Use tempfile to create a unique filename
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir=temp_dir) as f:
            filename = f.name
        
        tts.save(filename)
        logger.info(f"TTS audio saved to: {filename}")
        
        return filename
    except Exception as e:
        logger.error(f"Failed to generate TTS: {e}")
        # For robustness, return a default audio file path if it exists
        default_audio = "default_audio.mp3"
        if os.path.exists(default_audio):
            return default_audio
        raise Exception(f"Failed to generate TTS: {str(e)}")

# Function to summarize text using a pre-trained model
def summarize_text(text, max_length=150, min_length=50):
    """
    Summarizes the input text using a pre-trained summarization model.
    Args:
        text (str): Text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
    Returns:
        str: Summarized text.
    """
    if not text or len(text) < 100:  # Don't summarize very short texts
        return text
    
    text = clean_text(text)
    
    summarizer = get_summarizer()
    if summarizer is None:
        # Fallback to extractive summarization if model loading failed
        return extractive_summarize(text, max_sentences=3)
    
    try:
        # Truncate text if it's too long for the model
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return extractive_summarize(text, max_sentences=3)

# Function to perform extractive summarization
def extractive_summarize(text, max_sentences=3):
    """
    Performs simple extractive summarization by selecting important sentences.
    
    Args:
        text (str): The text to summarize.
        max_sentences (int): Maximum number of sentences to include.
    
    Returns:
        str: Extractive summary of the text.
    """
    if not text:
        return ""
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= max_sentences:
        return text
    
    # Score sentences based on position and word importance
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        # Position score (first sentences usually more important)
        position_score = 1.0 if i < 2 else 0.5
        
        # Length score (avoid very short sentences)
        length_score = min(1.0, len(sentence) / 40)
        
        # Word importance score
        words = word_tokenize(sentence.lower())
        word_score = sum(1 for word in words if word not in CUSTOM_STOPWORDS and word not in string.punctuation) / max(1, len(words))
        
        # Calculate final score
        sentence_scores[i] = (position_score + length_score + word_score) / 3
    
    # Select top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Re-sort by position
    
    # Construct summary
    summary = " ".join(sentences[i] for i, _ in top_sentences)
    return summary