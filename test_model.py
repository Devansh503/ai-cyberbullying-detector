#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.cyberbully_detector.inference import Predictor

def test_predictor():
    print("Loading predictor...")
    predictor = Predictor()
    
    # Test with some sample text
    test_texts = [
        "You are such a loser, nobody likes you",
        "Have a great day!",
        "I hope you fail at everything you do",
        "This movie was really good"
    ]
    
    print("\nTesting predictions:")
    print("-" * 60)
    
    for text in test_texts:
        try:
            result = predictor.predict(text)
            print(f"Text: {text[:50]}...")
            print(f"Prediction: {result['prediction']}")
            print(f"Category: {result['category']}")
            print(f"Severity: {result['severity']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("-" * 60)
        except Exception as e:
            print(f"Error with text '{text}': {e}")
            print("-" * 60)

if __name__ == "__main__":
    test_predictor()