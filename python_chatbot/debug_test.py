#!/usr/bin/env python3
"""
Quick test to debug the intent classification issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import PythonChatbot

def test_specific_queries():
    """Test the specific queries the user mentioned"""
    chatbot = PythonChatbot()
    
    test_queries = [
        "how do i use python",
        "how do i make game with python"
    ]
    
    print("🔍 Testing Specific User Queries")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 30)
        
        # Check intent classification directly
        intent, confidence = chatbot.intent_classifier.classify_intent(query)
        print(f"Intent: {intent.name if intent else 'None'}")
        print(f"Confidence: {confidence:.3f}")
        if intent:
            print(f"Required confidence: {intent.required_confidence}")
            print(f"Should trigger: {'Yes' if confidence >= intent.required_confidence else 'No'}")
        
        print("\nResponse:")
        response = chatbot.chat(query)
        # Show first 200 characters of response
        print(response[:200] + ("..." if len(response) > 200 else ""))
        print()

if __name__ == "__main__":
    test_specific_queries()
