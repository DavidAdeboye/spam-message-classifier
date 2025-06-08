import pandas as pd
import numpy as np
import pickle
import requests
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def detect_scam_patterns(text):
    """Detect comprehensive scam patterns in text."""
    text = text.lower()
    
    # Common scam patterns
    patterns = {
        # Investment scams
        'investment_scam': r'(invest|returns?|profit|double|guaranteed|risk[- ]?free)',
        'quick_money': r'(hours?|days?|weeks?|quick|fast|rapid|instant)',
        'money_amount': r'([€£$₦]\d+k?|\d+[€£$₦]|bitcoin|btc|crypto)',
        
        # Prize scams
        'prize_scam': r'(won|winner|claim|prize|congratulations|selected|lucky)',
        'product_bait': r'(iphone|samsung|apple|amazon|gift[ -]?card)',
        
        # Job scams
        'job_scam': r'(job|position|salary|income|earn|hiring|remote|work[- ]?from[- ]?home)',
        'high_pay': r'(\d+k|k\/month|\d+\/month|weekly pay)',
        
        # Account/security scams
        'security_scam': r'(account|security|verify|suspended|locked|unusual|activity)',
        'urgent_action': r'(urgent|immediate|action required|warning|alert)',
        
        # Link patterns
        'suspicious_url': r'(bit\.ly|tinyurl|short\.ly|click|\.xyz|\.info|\.site|\.tech|\.netlify)',
        'form_bait': r'(form|fill|apply|register|sign[- ]?up)',
        
        # Social manipulation
        'emotional_bait': r'(babe|honey|dear|darling|beautiful|sexy|hot)',
        'pressure_tactics': r'(limited time|only|expires|last chance|now|today)',
        
        # Blackmail/threat patterns
        'blackmail': r'(saw|recorded|video|photos|boss|family|friends|private)',
        'threat': r'(send|share|expose|publish|payment|pay)',
        
        # Common spam indicators
        'spam_formatting': r'([!₦🎁💰👀💋]|[A-Z]{4,})',
        'dm_request': r'(dm|pm|message|contact)',
        
        # Delivery/shipping scams
        'delivery_scam': r'(package|delivery|shipping|track|status)',
        
        # Common spam phrases
        'no_risk': r'(no risk|guaranteed|100%|legitimate|trusted)',
        'too_good': r'(free|bonus|extra|special offer|exclusive)'
    }
    
    # Check for pattern combinations
    matches = {k: bool(re.search(v, text)) for k, v in patterns.items()}
    
    # Scam indicators
    scam_indicators = []
    
    # Investment scam detection
    if matches['investment_scam'] and matches['quick_money'] and matches['money_amount']:
        scam_indicators.append('INVESTMENT_SCAM')
    
    # Prize scam detection
    if matches['prize_scam'] and (matches['product_bait'] or matches['suspicious_url']):
        scam_indicators.append('PRIZE_SCAM')
    
    # Job scam detection
    if matches['job_scam'] and matches['high_pay'] and matches['form_bait']:
        scam_indicators.append('JOB_SCAM')
    
    # Security/account scam detection
    if matches['security_scam'] and matches['urgent_action']:
        scam_indicators.append('SECURITY_SCAM')
    
    # Romance/emotional scam detection
    if matches['emotional_bait'] and matches['suspicious_url']:
        scam_indicators.append('ROMANCE_SCAM')
    
    # Blackmail scam detection
    if matches['blackmail'] and matches['threat'] and matches['money_amount']:
        scam_indicators.append('BLACKMAIL_SCAM')
    
    # Delivery scam detection
    if matches['delivery_scam'] and matches['suspicious_url']:
        scam_indicators.append('DELIVERY_SCAM')
    
    # General spam indicators
    if matches['spam_formatting'] and matches['suspicious_url']:
        scam_indicators.append('SUSPICIOUS_FORMAT')
    
    if matches['no_risk'] and matches['too_good']:
        scam_indicators.append('TOO_GOOD_TO_BE_TRUE')
    
    # URL detection
    if matches['suspicious_url'] and (matches['prize_scam'] or matches['money_amount']):
        scam_indicators.append('SUSPICIOUS_URL')
    
    return ' '.join(scam_indicators)

def preprocess_text(text):
    """Preprocess text data with enhanced scam pattern detection."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace common obfuscation patterns
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)  # Remove emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'\d+k', 'thousand', text)  # Normalize money amounts
    text = re.sub(r'(\d+)/?(hour|day|week|month)', r'\1 \2', text)  # Normalize time periods
    
    # Add scam pattern indicators
    scam_patterns = detect_scam_patterns(text)
    
    # Remove special characters but keep some meaningful ones
    text = re.sub(r'[^\w\s$€£₦./-]', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Add detected patterns to the text
    if scam_patterns:
        text += ' ' + scam_patterns
    
    return text

def load_base_dataset():
    """Load and preprocess the base SMS dataset."""
    try:
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        df = pd.read_csv(url, sep='\t')
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        logging.info(f"Base dataset loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading base dataset: {str(e)}")
        return pd.DataFrame(columns=['label', 'message'])

def load_kaggle_dataset():
    """Load and preprocess the Kaggle spam dataset."""
    try:
        # Assuming the file is downloaded and stored locally as 'emails.csv'
        df = pd.read_csv('emails.csv')
        df = df[['label', 'text']].rename(columns={'text': 'message'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        logging.info(f"Kaggle dataset loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading Kaggle dataset: {str(e)}")
        return pd.DataFrame(columns=['label', 'message'])

def load_user_feedback():
    """Load and preprocess user feedback data."""
    try:
        df = pd.read_csv('user_feedback.csv')
        df['label'] = df['actual_label'].map({'not_spam': 0, 'spam': 1})
        df = df[['message', 'label']]
        logging.info(f"User feedback loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading user feedback: {str(e)}")
        return pd.DataFrame(columns=['message', 'label'])

def load_local_spam_dataset():
    """Load and preprocess the local spam dataset."""
    try:
        df = pd.read_csv('spam.csv')
        df.columns = ['label', 'message'] + [f'col_{i}' for i in range(len(df.columns)-2)]
        df = df[['label', 'message']]  # Keep only relevant columns
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        logging.info(f"Local spam dataset loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading local spam dataset: {str(e)}")
        return pd.DataFrame(columns=['label', 'message'])

def load_legitimate_examples():
    """Load and preprocess legitimate examples."""
    try:
        df = pd.read_csv('legitimate_examples.csv')
        logging.info(f"Legitimate examples loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading legitimate examples: {str(e)}")
        return pd.DataFrame(columns=['message', 'label'])

def load_sophisticated_spam():
    """Load and preprocess sophisticated spam examples."""
    try:
        df = pd.read_csv('sophisticated_spam.csv')
        # Convert label to numeric
        df['label'] = df['label'].map({'spam': 1, 'not_spam': 0})
        # Duplicate sophisticated spam examples
        df = pd.concat([df] * 3, ignore_index=True)
        logging.info(f"Sophisticated spam examples loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading sophisticated spam examples: {str(e)}")
        return pd.DataFrame(columns=['message', 'label'])

def load_common_scams():
    """Load common scam examples."""
    try:
        df = pd.read_csv('common_scams.csv')
        # Convert label to numeric
        df['label'] = 1  # All examples are spam
        # Duplicate common scams to increase their weight
        df = pd.concat([df] * 3, ignore_index=True)  # Triple the weight
        logging.info(f"Common scams loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading common scams: {str(e)}")
        return pd.DataFrame(columns=['message', 'label'])

def merge_datasets():
    """Merge all available datasets with priority weighting."""
    datasets = []
    
    # Load common scams first (highest priority)
    common_scams_df = load_common_scams()
    if not common_scams_df.empty:
        datasets.append(common_scams_df)
    
    # Load sophisticated spam (high priority)
    sophisticated_df = load_sophisticated_spam()
    if not sophisticated_df.empty:
        datasets.append(sophisticated_df)
    
    # Load user feedback (medium-high priority)
    feedback_df = load_user_feedback()
    if not feedback_df.empty:
        datasets.append(feedback_df)
    
    # Load base dataset (lower priority)
    base_df = load_base_dataset()
    if not base_df.empty:
        datasets.append(base_df)
    
    # Merge all datasets
    if datasets:
        merged_df = pd.concat(datasets, ignore_index=True)
        # Preprocess messages
        merged_df['message'] = merged_df['message'].apply(preprocess_text)
        # Remove duplicates but keep the first occurrence (prioritizing common scams)
        merged_df = merged_df.drop_duplicates(subset=['message'])
        logging.info(f"Merged dataset created with {len(merged_df)} total records")
        return merged_df
    else:
        logging.error("No datasets available to merge")
        return pd.DataFrame(columns=['message', 'label'])

def extract_features(text):
    """Extract additional features from text."""
    features = {}
    
    # Pattern indicators
    features['has_payment'] = bool(re.search(r'(pay|payment|money|buy|paid|cash|dollar|€|£|\$)', text.lower()))
    features['has_social_media'] = bool(re.search(r'(instagram|ig|insta|social\s*media|facebook|fb)', text.lower()))
    features['has_permission'] = bool(re.search(r'(can|could|would|use|send)', text.lower()))
    features['has_art_reference'] = bool(re.search(r'(art|mural|project|work|client)', text.lower()))
    features['has_pic_reference'] = bool(re.search(r'(pic|picture|photo|image)', text.lower()))
    
    # Suspicious combinations
    features['payment_and_permission'] = features['has_payment'] and features['has_permission']
    features['social_pic_payment'] = features['has_social_media'] and features['has_pic_reference'] and features['has_payment']
    
    # Legitimate patterns
    features['has_question'] = '?' in text
    features['has_greeting'] = bool(re.search(r'^(hi|hey|hello)', text.lower()))
    features['has_appreciation'] = bool(re.search(r'(beautiful|amazing|love|great|nice)', text.lower()))
    
    return features

def train_model(X_train, y_train, X_test, y_test):
    """Train and evaluate the model with enhanced scam detection."""
    # Create pipeline with adjusted parameters
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=10000,  # Increased feature set
            ngram_range=(1, 4),  # Include longer phrases
            min_df=1,  # Include rare patterns
            max_df=0.9  # Exclude very common words
        )),
        ('clf', MultinomialNB(
            alpha=0.01  # Lower alpha for more sensitivity
        ))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Log metrics
    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(y_test, y_pred))
    
    logging.info("\nConfusion Matrix:")
    logging.info("\n" + str(confusion_matrix(y_test, y_pred)))
    
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"\nAccuracy Score: {accuracy:.3f}")
    
    return model

def save_model(model, filename='spam_model.pkl'):
    """Save the trained model."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved successfully as {filename}")
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")

def main():
    """Main training pipeline."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logging.info("Starting model training pipeline...")
    
    # Merge all datasets
    merged_df = merge_datasets()
    
    if merged_df.empty:
        logging.error("No data available for training")
        return
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        merged_df['message'],
        merged_df['label'],
        test_size=0.2,
        random_state=42
    )
    
    # Train and evaluate model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    save_model(model)
    
    logging.info("Model training pipeline completed")

if __name__ == "__main__":
    main()