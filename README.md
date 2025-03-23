# NewsPulse 📰  

**AI-driven News Sentiment & Insights**  

NewsPulse analyzes news articles related to a company, extracts key insights, performs sentiment analysis, and provides a Hindi audio summary.  

## 🌐 Live Demo  
- **Hugging Face:** [NewsPulse](https://huggingface.co/spaces/devan-s/NewsPulse)
- **Vercel:** [NewsPulse](https://my-huggingface-app.vercel.app/)  

---

## ✨ Features  

✔ **Sentiment Analysis** – Detects **Positive, Negative, Neutral** sentiment.  
✔ **Topic Extraction** – Identifies key themes from articles.  
✔ **Comparative Insights** – Highlights sentiment trends across news.  
✔ **Hindi Audio Summary** – Converts insights into speech.  
✔ **Interactive Web App** – Built with **Streamlit** and deployed on **Hugging Face Spaces**.  

---

## 🛠 Tech Stack  

- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **NLP Models:** VADER, KeyBERT  
- **TTS:** gTTS (Hindi Speech)  
- **APIs:** GNews (News Fetching)  
- **Deployment:** Hugging Face Spaces, Vercel  

---

## 🚀 Installation & Setup  

### Prerequisites  
- **Python 3.8+**  
- **Pip**  

### Steps  

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/NewsPulse.git
cd NewsPulse
```

2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

3️⃣ **Set Up API Key**  
Create a `.env` file and add:  
```
GNEWS_API_KEY=your_api_key_here
```

4️⃣ **Run Backend**  
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

5️⃣ **Run Frontend**  
```bash
streamlit run app.py
```

6️⃣ **Access the App**  
Go to: `http://localhost:8501`  

---

## 🖥 Usage  

1️⃣ Enter a **Company Name** (e.g., Tesla).  
2️⃣ Click **Analyze** to fetch and process news.  
3️⃣ View **sentiment insights**, **topic overlap**, and **comparative scores**.  
4️⃣ Listen to the **Hindi audio summary**.  

---

## 📸 Screenshots  
📌 **Coming Soon!**  

---

## 🤝 Contributing  

1️⃣ Fork the repo.  
2️⃣ Create a branch (`git checkout -b feature-name`).  
3️⃣ Commit & push (`git commit -m "Added feature"`).  
4️⃣ Open a **pull request**.  

---

## License 📜

© 2025 Devan. All Rights Reserved.  
This project is proprietary and cannot be copied, modified, distributed, or used without explicit permission from the owner.  

For inquiries regarding usage rights, please contact: **devandev9400@gmail.com**.

---

## 📧 Contact  

- **GitHub:** [Devan](https://github.com/Devan7117)
- **LinkedIn:** [Devan](https://www.linkedin.com/in/devan-s-9941591b7/)  

🔥 **Developed by Devan S** 🚀  
