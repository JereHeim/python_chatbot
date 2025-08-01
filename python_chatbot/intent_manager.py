#!/usr/bin/env python3
"""
Intent Management Tool - Easy way to manage chatbot intents without coding
"""

import json
import sys
from pathlib import Path
from intent_classifier import IntentClassifier, Intent

class IntentManager:
    def __init__(self, config_file='intents_config.json'):
        self.config_file = config_file
        self.classifier = IntentClassifier(config_file)
    
    def list_intents(self):
        """List all current intents"""
        print("📋 Current Intents:")
        print("=" * 50)
        
        intents = self.classifier.get_all_intents()
        for intent in sorted(intents, key=lambda x: x.name):
            print(f"🎯 {intent.name}")
            print(f"   Keywords: {', '.join(intent.keywords[:5])}...")
            print(f"   Priority: {intent.priority}")
            print(f"   Response Method: {intent.response_method}")
            print()
    
    def add_intent(self):
        """Add a new intent interactively"""
        print("➕ Adding New Intent")
        print("=" * 30)
        
        name = input("Intent name: ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return
        
        print("Enter keywords (comma-separated):")
        keywords_input = input("> ").strip()
        keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
        
        print("Enter regex patterns (comma-separated, optional):")
        patterns_input = input("> ").strip()
        patterns = [p.strip() for p in patterns_input.split(',') if p.strip()] if patterns_input else []
        
        response_method = input("Response method name (e.g., _get_example_help): ").strip()
        if not response_method:
            response_method = f"_get_{name}_help"
        
        try:
            priority = int(input("Priority (1-5, default 2): ") or "2")
        except ValueError:
            priority = 2
        
        try:
            confidence = float(input("Required confidence (0.0-1.0, default 0.3): ") or "0.3")
        except ValueError:
            confidence = 0.3
        
        new_intent = Intent(
            name=name,
            keywords=keywords,
            patterns=patterns,
            response_method=response_method,
            priority=priority,
            required_confidence=confidence
        )
        
        self.classifier.add_intent(new_intent)
        self.save_config()
        
        print(f"✅ Added intent '{name}' successfully!")
    
    def remove_intent(self):
        """Remove an intent interactively"""
        print("➖ Removing Intent")
        print("=" * 30)
        
        self.list_intents()
        name = input("Enter intent name to remove: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return
        
        intent = self.classifier.get_intent_by_name(name)
        if not intent:
            print(f"❌ Intent '{name}' not found")
            return
        
        confirm = input(f"Are you sure you want to remove '{name}'? (y/N): ").strip().lower()
        if confirm == 'y':
            self.classifier.remove_intent(name)
            self.save_config()
            print(f"✅ Removed intent '{name}' successfully!")
        else:
            print("❌ Operation cancelled")
    
    def modify_intent(self):
        """Modify an existing intent"""
        print("✏️  Modifying Intent")
        print("=" * 30)
        
        self.list_intents()
        name = input("Enter intent name to modify: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return
        
        intent = self.classifier.get_intent_by_name(name)
        if not intent:
            print(f"❌ Intent '{name}' not found")
            return
        
        print(f"Current settings for '{name}':")
        print(f"  Keywords: {', '.join(intent.keywords)}")
        print(f"  Patterns: {', '.join(intent.patterns)}")
        print(f"  Priority: {intent.priority}")
        print(f"  Confidence: {intent.required_confidence}")
        print()
        
        # Allow modification of each field
        print("Enter new values (press Enter to keep current):")
        
        new_keywords = input(f"Keywords [{', '.join(intent.keywords)}]: ").strip()
        if new_keywords:
            keywords = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
            self.classifier.update_intent(name, keywords=keywords)
        
        new_patterns = input(f"Patterns [{', '.join(intent.patterns)}]: ").strip()
        if new_patterns:
            patterns = [p.strip() for p in new_patterns.split(',') if p.strip()]
            self.classifier.update_intent(name, patterns=patterns)
        
        new_priority = input(f"Priority [{intent.priority}]: ").strip()
        if new_priority:
            try:
                priority = int(new_priority)
                self.classifier.update_intent(name, priority=priority)
            except ValueError:
                print("⚠️  Invalid priority, keeping current value")
        
        new_confidence = input(f"Confidence [{intent.required_confidence}]: ").strip()
        if new_confidence:
            try:
                confidence = float(new_confidence)
                self.classifier.update_intent(name, required_confidence=confidence)
            except ValueError:
                print("⚠️  Invalid confidence, keeping current value")
        
        self.save_config()
        print(f"✅ Modified intent '{name}' successfully!")
    
    def test_query(self):
        """Test a query against the current intents"""
        print("🧪 Testing Query")
        print("=" * 30)
        
        query = input("Enter test query: ").strip()
        if not query:
            print("❌ Query cannot be empty")
            return
        
        intent, confidence = self.classifier.classify_intent(query)
        
        if intent:
            print(f"✅ Detected Intent: {intent.name}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Keywords matched: {[kw for kw in intent.keywords if kw.lower() in query.lower()]}")
        else:
            print("❌ No intent detected")
        
        print()
    
    def save_config(self):
        """Save current configuration to file"""
        self.classifier.save_intents_to_file(self.config_file)
        print(f"💾 Configuration saved to {self.config_file}")
    
    def backup_config(self):
        """Create a backup of the current configuration"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"intents_config_backup_{timestamp}.json"
        
        self.classifier.save_intents_to_file(backup_file)
        print(f"💾 Backup created: {backup_file}")
    
    def run(self):
        """Run the interactive management interface"""
        while True:
            print("\n🤖 Intent Management Tool")
            print("=" * 40)
            print("1. List all intents")
            print("2. Add new intent")
            print("3. Remove intent")
            print("4. Modify intent")
            print("5. Test query")
            print("6. Save configuration")
            print("7. Create backup")
            print("8. Exit")
            print()
            
            choice = input("Choose an option (1-8): ").strip()
            
            if choice == '1':
                self.list_intents()
            elif choice == '2':
                self.add_intent()
            elif choice == '3':
                self.remove_intent()
            elif choice == '4':
                self.modify_intent()
            elif choice == '5':
                self.test_query()
            elif choice == '6':
                self.save_config()
            elif choice == '7':
                self.backup_config()
            elif choice == '8':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'intents_config.json'
    
    manager = IntentManager(config_file)
    
    try:
        manager.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
