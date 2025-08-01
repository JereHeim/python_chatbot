"""
Intent classification system for the Python chatbot.
This makes the chatbot more scalable by using configuration instead of hardcoded keywords.
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Intent:
    name: str
    keywords: List[str]
    patterns: List[str]
    response_method: str
    priority: int = 1
    required_confidence: float = 0.3

class IntentClassifier:
    def __init__(self, config_file: Optional[str] = None):
        self.intents: List[Intent] = []
        self.load_intents(config_file)
    
    def load_intents(self, config_file: Optional[str] = None):
        """Load intents from configuration file or use defaults"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self._load_from_config(config)
        else:
            self._load_default_intents()
    
    def _load_from_config(self, config: Dict):
        """Load intents from configuration dictionary"""
        for intent_data in config.get('intents', []):
            intent = Intent(
                name=intent_data['name'],
                keywords=intent_data.get('keywords', []),
                patterns=intent_data.get('patterns', []),
                response_method=intent_data['response_method'],
                priority=intent_data.get('priority', 1),
                required_confidence=intent_data.get('required_confidence', 0.3)
            )
            self.intents.append(intent)
    
    def _load_default_intents(self):
        """Load default intents when no config file is provided"""
        default_intents = [
            Intent(
                name="game_development",
                keywords=["game", "gaming", "pygame", "arcade", "graphics", "sprite", "animation"],
                patterns=[r".*make.*game", r".*create.*game", r".*build.*game", r".*game.*development"],
                response_method="_get_game_development_help",
                priority=2
            ),
            Intent(
                name="gui_development",
                keywords=["gui", "desktop", "tkinter", "pyqt", "kivy", "interface", "window", "button"],
                patterns=[r".*desktop.*app", r".*gui.*application", r".*user.*interface"],
                response_method="_get_gui_development_help",
                priority=2
            ),
            Intent(
                name="web_scraping",
                keywords=["scraping", "scrape", "beautifulsoup", "selenium", "requests", "crawl", "extract"],
                patterns=[r".*web.*scrap", r".*extract.*data", r".*crawl.*website"],
                response_method="_get_web_scraping_help",
                priority=2
            ),
            Intent(
                name="machine_learning",
                keywords=["machine learning", "ai", "tensorflow", "pytorch", "sklearn", "model", "neural", "deep learning"],
                patterns=[r".*machine.*learn", r".*artificial.*intelligence", r".*neural.*network"],
                response_method="_get_ml_help",
                priority=2
            ),
            Intent(
                name="automation",
                keywords=["automation", "automate", "script", "task", "schedule", "bot", "workflow"],
                patterns=[r".*automate.*task", r".*schedule.*job", r".*automation.*script"],
                response_method="_get_automation_help",
                priority=2
            ),
            Intent(
                name="file_handling",
                keywords=["file", "csv", "json", "excel", "read", "write", "parse", "save"],
                patterns=[r".*read.*file", r".*write.*file", r".*file.*handling"],
                response_method="_get_file_handling_help",
                priority=2
            ),
            Intent(
                name="database",
                keywords=["database", "sql", "sqlite", "mysql", "postgresql", "query", "table"],
                patterns=[r".*database.*operation", r".*sql.*query", r".*store.*data"],
                response_method="_get_database_help",
                priority=2
            ),
            Intent(
                name="web_development",
                keywords=["web", "django", "flask", "fastapi", "api", "server", "website"],
                patterns=[r".*web.*development", r".*build.*website", r".*web.*app"],
                response_method="_get_web_development_help",
                priority=2
            ),
            Intent(
                name="data_science",
                keywords=["data", "pandas", "numpy", "analysis", "science", "visualization", "matplotlib"],
                patterns=[r".*data.*analysis", r".*data.*science", r".*analyze.*data"],
                response_method="_get_data_science_help",
                priority=2
            ),
            Intent(
                name="coding_puzzle",
                keywords=["puzzle", "challenge", "problem", "exercise", "give me", "show me"],
                patterns=[r".*give.*puzzle", r".*coding.*challenge", r".*python.*problem"],
                response_method="_get_python_puzzle",
                priority=3
            ),
            Intent(
                name="beginner_help",
                keywords=["start", "begin", "learn", "new", "basic", "tutorial", "beginner"],
                patterns=[r".*how.*start", r".*getting.*started", r".*learn.*python"],
                response_method="_get_beginner_python_help",
                priority=1
            ),
            Intent(
                name="syntax_help",
                keywords=["list", "dict", "string", "loop", "function", "class", "variable", "syntax"],
                patterns=[r".*how.*to.*", r".*syntax.*", r".*define.*"],
                response_method="_get_syntax_help",
                priority=1
            )
        ]
        self.intents.extend(default_intents)
    
    def classify_intent(self, query: str) -> Tuple[Optional[Intent], float]:
        """Classify user query and return best matching intent with confidence score"""
        query_lower = query.lower()
        best_intent = None
        best_score = 0.0
        
        for intent in sorted(self.intents, key=lambda x: x.priority, reverse=True):
            score = self._calculate_intent_score(query_lower, intent)
            
            if score > best_score and score >= intent.required_confidence:
                best_score = score
                best_intent = intent
        
        return best_intent, best_score
    
    def _calculate_intent_score(self, query: str, intent: Intent) -> float:
        """Calculate confidence score for an intent match"""
        keyword_score = 0.0
        pattern_score = 0.0
        
        # Keyword matching with partial matching and better scoring
        if intent.keywords:
            matched_keywords = 0
            total_possible_matches = len(intent.keywords)
            
            for keyword in intent.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in query:
                    # Exact match gets full points
                    matched_keywords += 1
                elif any(word in query for word in keyword_lower.split()):
                    # Partial match gets half points
                    matched_keywords += 0.5
            
            keyword_score = min(matched_keywords / total_possible_matches, 1.0)
        
        # Pattern matching
        if intent.patterns:
            pattern_matches = sum(1 for pattern in intent.patterns if re.search(pattern, query, re.IGNORECASE))
            pattern_score = min(pattern_matches / len(intent.patterns), 1.0)
        
        # If no keywords but has patterns, use only pattern score
        if not intent.keywords and intent.patterns:
            final_score = pattern_score
        # If no patterns but has keywords, use only keyword score
        elif intent.keywords and not intent.patterns:
            final_score = keyword_score
        # If has both, combine them
        else:
            final_score = (keyword_score * 0.7) + (pattern_score * 0.3)
        
        # Apply minimum threshold boost for very relevant matches
        if final_score > 0.1 and any(key_term in query for key_term in ['game', 'gui', 'scraping', 'machine learning', 'automate', 'database', 'puzzle', 'help', 'how to']):
            final_score = max(final_score, 0.25)
        
        # Boost score for high-priority intents if there's any match
        if final_score > 0 and intent.priority > 1:
            final_score *= (1 + (intent.priority - 1) * 0.05)
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def add_intent(self, intent: Intent):
        """Add a new intent dynamically"""
        self.intents.append(intent)
    
    def remove_intent(self, intent_name: str):
        """Remove an intent by name"""
        self.intents = [intent for intent in self.intents if intent.name != intent_name]
    
    def update_intent(self, intent_name: str, **kwargs):
        """Update an existing intent"""
        for intent in self.intents:
            if intent.name == intent_name:
                for key, value in kwargs.items():
                    if hasattr(intent, key):
                        setattr(intent, key, value)
                break
    
    def save_intents_to_file(self, filename: str):
        """Save current intents to a JSON configuration file"""
        config = {
            "intents": [
                {
                    "name": intent.name,
                    "keywords": intent.keywords,
                    "patterns": intent.patterns,
                    "response_method": intent.response_method,
                    "priority": intent.priority,
                    "required_confidence": intent.required_confidence
                }
                for intent in self.intents
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get_all_intents(self) -> List[Intent]:
        """Get all loaded intents"""
        return self.intents.copy()
    
    def get_intent_by_name(self, name: str) -> Optional[Intent]:
        """Get a specific intent by name"""
        for intent in self.intents:
            if intent.name == name:
                return intent
        return None
