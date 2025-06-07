# 📱 SMS Spam Classifier

AI-powered spam detection system that protects users from unwanted SMS messages using NLP and Machine Learning.

🧰 Tech Stack
- Python 3.x
- scikit-learn
- NLTK
- Streamlit
- Pandas
- NumPy

🧠 What I'm Learning
- Natural Language Processing (NLP)
- Machine Learning Classification
- Text Preprocessing Techniques
- Model Evaluation Metrics
- Web App Development with Streamlit

✅ Project Roadmap

Phase 1: Data Preparation
- [x] Dataset acquisition and exploration
- [x] Text preprocessing pipeline
- [x] Feature engineering with TF-IDF
- [x] Train-test split implementation

Phase 2: Model Development
- [x] Naive Bayes classifier implementation
- [x] Cross-validation setup
- [x] Hyperparameter optimization
- [x] Model evaluation metrics

Phase 3: Application Development
- [x] Streamlit web interface
- [x] Model serialization
- [x] Real-time prediction endpoint
- [x] User-friendly result display

Phase 4: Deployment & Optimization
- [ ] Cloud deployment
- [ ] Performance optimization
- [ ] Batch prediction support
- [ ] Model retraining pipeline

📊 Current Performance

Model Metrics:
- ✨ 97.8% Average Accuracy
- 🎯 98% Precision on Non-spam
- 🚫 95% Precision on Spam
- ⚡ Fast Real-time Predictions

🔧 How It Works

1. Text Cleaning:
   ```python
   def preprocess_text(text):
       text = text.lower()
       text = re.sub(r'[^a-zA-Z\s]', '', text)
       return ' '.join(text.split())
   ```

2. Feature Extraction:
   ```python
   vectorizer = TfidfVectorizer(
       stop_words='english',
       max_features=5000
   )
   ```

3. Classification:
   ```python
   classifier = MultinomialNB(alpha=0.1)
   model = Pipeline([
       ('tfidf', vectorizer),
       ('clf', classifier)
   ])
   ```

🚀 Quick Start

1. Setup:
   ```bash
   git clone <your-repo>
   cd spam-classifier
   pip install -r requirements.txt
   ```

2. Train Model:
   ```bash
   python spam_classifier.py
   ```

3. Launch App:
   ```bash
   streamlit run app.py
   ```

Visit `http://localhost:8501` to start classifying messages!

📚 Dataset
- 5,574 labeled SMS messages
- Binary classification (spam/ham)
- Public dataset from [Justin Markham](https://github.com/justmarkham)

🕒 Progress Log

June 7, 2025
- ✅ Implemented text preprocessing
- ✅ Added cross-validation
- ✅ Deployed Streamlit interface
- 📈 Achieved 97.8% accuracy

Next Steps:
- [ ] Add batch processing
- [ ] Implement model retraining
- [ ] Deploy to cloud platform
- [ ] Add API endpoint
