import pickle
from enhanced_spam_classifier import preprocess_text, extract_features

def test_model(model, test_cases):
    """Test the model with various test cases."""
    print("\n=== Model Testing Results ===\n")
    
    for msg, expected in test_cases.items():
        # Get prediction
        prediction = model.predict([msg])[0]
        features = extract_features(msg)
        
        # Print results
        print(f"Message: {msg}")
        print(f"Expected: {'Spam' if expected else 'Not Spam'}")
        print(f"Predicted: {'Spam' if prediction == 1 else 'Not Spam'}")
        print("\nDetected Features:")
        for k, v in features.items():
            if v:
                print(f"- {k}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # Load the model
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Test cases
    test_cases = {
        "Hey I saw your pic on Instagram, can I paint a mural of you and send to my client, I'd pay you for using your pic tho": 1,
        "Love your photography style! Do you offer photography classes?": 0,
        "I'm a digital artist and would love to use your photos for a paid project": 1,
        "Your Instagram feed is so well curated, great aesthetic!": 0,
        "Saw your content, interested in buying rights to your photos": 1,
        "Can you share tips on how you edit your Instagram photos?": 0,
        "Want to offer you money to use your social media pictures": 1,
        "Hey, are you available for a professional photoshoot next week?": 0
    }
    
    # Run tests
    test_model(model, test_cases)
