#!/usr/bin/env python3
"""
Enhanced test script to verify chatbot responses and intent classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import PythonChatbot
from intent_classifier import IntentClassifier

def test_intent_classification():
    """Test the intent classification system"""
    print("🔍 Testing Intent Classification")
    print("=" * 50)
    
    classifier = IntentClassifier('intents_config.json')
    
    test_queries = [
        ("how can i make a game with python", "game_development"),
        ("web scraping tutorial", "web_scraping"),
        ("machine learning basics", "machine_learning"),
        ("gui application", "gui_development"),
        ("python puzzle", "coding_puzzle"),
        ("automate my tasks", "automation"),
        ("read csv file", "file_handling"),
        ("database operations", "database"),
        ("getting started with python", "beginner_help"),
        ("how to create a list", "syntax_help"),
    ]
    
    correct_classifications = 0
    total_tests = len(test_queries)
    
    for query, expected_intent in test_queries:
        intent, confidence = classifier.classify_intent(query)
        actual_intent = intent.name if intent else "unknown"
        
        is_correct = actual_intent == expected_intent
        status = "✅" if is_correct else "❌"
        
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected_intent}")
        print(f"   Got: {actual_intent} (confidence: {confidence:.2f})")
        print()
        
        if is_correct:
            correct_classifications += 1
    
    accuracy = (correct_classifications / total_tests) * 100
    print(f"🎯 Intent Classification Accuracy: {accuracy:.1f}% ({correct_classifications}/{total_tests})")
    print()

def test_chatbot_responses():
    """Test various types of questions with the enhanced system"""
    chatbot = PythonChatbot()
    
    test_categories = {
        "Game Development": [
            "how can i make a game with python",
            "pygame tutorial",
            "2d game development"
        ],
        "Web Development": [
            "flask vs django",
            "build a web app",
            "create rest api"
        ],
        "Data Science": [
            "pandas dataframe",
            "data analysis python",
            "matplotlib visualization"
        ],
        "Machine Learning": [
            "sklearn tutorial",
            "neural networks",
            "train a model"
        ],
        "Automation": [
            "automate file operations",
            "python scheduling",
            "web scraping automation"
        ],
        "Beginner Questions": [
            "python basics",
            "getting started",
            "learn python programming"
        ],
        "Coding Puzzles": [
            "give me a puzzle",
            "coding challenge",
            "python problem"
        ]
    }
    
    print("🧪 Testing Enhanced Chatbot Responses")
    print("=" * 50)
    
    for category, questions in test_categories.items():
        print(f"\n📚 Category: {category}")
        print("-" * 30)
        
        for question in questions:
            print(f"\n❓ Question: '{question}'")
            
            response = chatbot.chat(question)
            
            # Check if response is specialized (not generic fallback)
            is_specialized = not (
                "The official Python documentation" in response and
                "PyPI (Python Package Index)" in response and
                "Real Python provides" in response
            )
            
            # Check if response contains relevant content
            has_relevant_content = (
                len(response) > 100 and 
                ("```python" in response or "**" in response or "🐍" in response)
            )
            
            if is_specialized and has_relevant_content:
                print("✅ GOOD SPECIALIZED RESPONSE")
                # Show first few lines
                lines = response.split('\n')[:3]
                for line in lines:
                    if line.strip():
                        print(f"   {line[:80]}...")
                        break
            elif is_specialized:
                print("⚠️  SPECIALIZED BUT BRIEF")
            else:
                print("❌ GENERIC FALLBACK RESPONSE")
    
    print("\n🎯 Test completed!")

def test_configuration_flexibility():
    """Test how easy it is to add new intents"""
    print("🔧 Testing Configuration Flexibility")
    print("=" * 50)
    
    classifier = IntentClassifier()
    
    # Add a new intent dynamically
    from intent_classifier import Intent
    
    new_intent = Intent(
        name="cybersecurity",
        keywords=["security", "cybersecurity", "encryption", "hacking", "vulnerability"],
        patterns=[r".*cyber.*security", r".*encrypt.*", r".*secure.*"],
        response_method="_get_cybersecurity_help",
        priority=2,
        required_confidence=0.3
    )
    
    classifier.add_intent(new_intent)
    
    # Test the new intent
    test_query = "python cybersecurity tools"
    intent, confidence = classifier.classify_intent(test_query)
    
    if intent and intent.name == "cybersecurity":
        print(f"✅ Successfully added and detected new intent: {intent.name}")
        print(f"   Query: '{test_query}'")
        print(f"   Confidence: {confidence:.2f}")
    else:
        print("❌ Failed to detect new intent")
    
    # Save configuration
    classifier.save_intents_to_file('test_intents_config.json')
    print("✅ Configuration saved to test_intents_config.json")
    
    print()

if __name__ == "__main__":
    print("🚀 Running Enhanced Chatbot Tests")
    print("=" * 60)
    
    # Run all tests
    test_intent_classification()
    test_chatbot_responses()
    test_configuration_flexibility()
    
    print("✨ All tests completed!")
