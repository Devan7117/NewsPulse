# News Sentiment Analysis and Summarization

This project analyzes news articles related to a given company, performs sentiment analysis, extracts key topics, and generates a comprehensive report with a Hindi audio summary.

## Project Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Devan7117/NewsPulse.git
   cd NewsPulse
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your GNews API key:
     ```plaintext
     GNEWS_API_KEY=your_api_key_here
     ```

### Running the Application

1. Start the FastAPI backend:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501` to access the application.

## Model Details

### Sentiment Analysis
- **Model**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Purpose**: Analyzes the sentiment of news articles (Positive, Negative, Neutral).
- **Integration**: The `analyze_sentiment` function in `utils.py` uses VADER to determine sentiment scores and adjusts them based on context keywords.

### Summarization
- **Model**: BART (Bidirectional and Auto-Regressive Transformers) from Hugging Face (`facebook/bart-large-cnn`).
- **Purpose**: Summarizes long news articles into concise summaries.
- **Integration**: The `summarize_text` function in `utils.py` uses the BART model for abstractive summarization. If the model fails, it falls back to extractive summarization.

### Text-to-Speech (TTS)
- **Model**: gTTS (Google Text-to-Speech)
- **Purpose**: Converts the generated Hindi summary into an audio file.
- **Integration**: The `text_to_speech` function in `utils.py` uses gTTS to generate Hindi audio summaries.

## API Development

### FastAPI Backend
- **Endpoints**:
  - `POST /analyze-news`: Fetches news articles, performs sentiment analysis, and generates a Hindi TTS summary.
  - `GET /health`: Health check endpoint to verify the API is running.
  - `GET /docs-info`: Provides information about the API and how to use it.

### Streamlit Frontend
- The frontend interacts with the FastAPI backend to display the analysis results, including sentiment distribution, topic analysis, and Hindi audio summaries.

### Accessing the API via Postman
1. Start the FastAPI backend.
2. Open Postman and create a new request.
3. Set the request type to `POST` and enter the URL `http://localhost:8000/analyze-news`.
4. In the body, select `raw` and `JSON`, then provide the following payload:
   ```json
   {
       "company_name": "Tesla",
       "api_key": "your_gnews_api_key",
       "language": "en",
       "max_articles": 10
   }
   ```
5. Send the request to get the analysis results.

## API Usage

### Third-Party APIs
- **GNews API**: Used to fetch news articles related to the given company name.
  - **Purpose**: Provides the latest news articles for sentiment analysis and summarization.
  - **Integration**: The `fetch_news` function in `api.py` makes a request to the GNews API to retrieve news articles.

## Assumptions & Limitations

### Assumptions
- The GNews API will return relevant news articles for the given company name.
- The sentiment analysis model (VADER) will accurately classify the sentiment of news articles.
- The summarization model (BART) will generate coherent and concise summaries.
- The TTS model (gTTS) will produce clear and understandable Hindi audio.

### Limitations
- The accuracy of sentiment analysis depends on the quality and context of the news articles.
- The summarization model may not always capture the most important points of long articles.
- The TTS model may have limitations in pronunciation for certain Hindi words.
- The application is limited to the number of articles returned by the GNews API (max 20 articles).

---

### Uploading to GitHub

1. Create a new repository on GitHub.
2. Add the `README.md` file to the root directory of your repository.
3. Push your code to the repository:
   ```bash
   git add .
   git commit -m "Initial commit with project setup and documentation"
   git push origin main
   ```
