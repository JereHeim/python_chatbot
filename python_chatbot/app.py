from flask import Flask, request, jsonify, render_template
import random
import spacy
from flask import session
import requests
from bs4 import BeautifulSoup, Tag
import urllib.parse
import re
from typing import List, Dict
from intent_classifier import IntentClassifier




app = Flask(__name__)
app.secret_key = 'your-very-secret-key'  # Replace with a secure secret key

# # Load medium model with vectors once
# nlp = spacy.load("en_core_web_md")

class PythonChatbot:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.python_keywords = ['python', 'programming', 'code', 'script', 'function', 'class', 'library', 'package']
        self.python_sites = [
            'docs.python.org',
            'stackoverflow.com',
            'github.com',
            'pypi.org',
            'realpython.com',
            'python.org',
            'geeksforgeeks.org'
        ]
        # Initialize the intent classifier
        self.intent_classifier = IntentClassifier('intents_config.json')
    
    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search the web for Python-related information"""
        # Enhance query with Python context if not already present
        python_query = self._enhance_python_query(query)
        
        try:
            # Use Google search with Python-focused query
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(python_query)}"
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            
            # Look for search result links
            for link in soup.find_all('a'):
                if isinstance(link, Tag):
                    href = link.get('href')
                    if href and isinstance(href, str) and href.startswith('/url?q='):
                        # Extract the actual URL from Google's redirect
                        actual_url = href.split('/url?q=')[1].split('&')[0]
                        actual_url = urllib.parse.unquote(actual_url)
                        
                        # Prioritize Python-related sites
                        if self._is_python_related_site(actual_url):
                            # Get the title
                            title_elem = link.find('h3')
                            title = title_elem.get_text().strip() if title_elem else actual_url
                            
                            if actual_url.startswith('http') and len(results) < num_results:
                                results.append({
                                    'title': title,
                                    'url': actual_url,
                                    'snippet': ''
                                })
            
            # If no Python-specific results, try fallback
            if not results:
                fallback_results = self._get_python_fallback_results(query)
                results.extend(fallback_results[:num_results])
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            # Return Python-focused fallback results
            return self._get_python_fallback_results(query)[:num_results]
    
    def _enhance_python_query(self, query: str) -> str:
        """Enhance query with Python context if needed"""
        query_lower = query.lower()
        
        # Check if query already contains Python-related terms
        has_python_context = any(keyword in query_lower for keyword in self.python_keywords)
        
        if not has_python_context:
            return f"python {query}"
        return query
    
    def _is_python_related_site(self, url: str) -> bool:
        """Check if URL is from a Python-related site"""
        return any(site in url.lower() for site in self.python_sites)
    
    def _get_python_fallback_results(self, query: str) -> List[Dict]:
        """Get fallback results from Python-specific sites"""
        enhanced_query = urllib.parse.quote(f"python {query}")
        fallback_sites = [
            f"https://docs.python.org/3/search.html?q={enhanced_query}",
            f"https://stackoverflow.com/search?q=python+{enhanced_query}",
            f"https://pypi.org/search/?q={enhanced_query}",
            f"https://realpython.com/?s={enhanced_query}"
        ]
        
        results = []
        for url in fallback_sites:
            try:
                response = self.session.head(url, timeout=5)
                if response.status_code == 200:
                    site_name = url.split('//')[1].split('/')[0]
                    results.append({
                        'title': f"Python: {query} - {site_name}",
                        'url': url,
                        'snippet': ''
                    })
            except:
                continue
        
        return results
    
    def extract_content(self, url: str) -> str:
        """Extract readable content from a webpage with better error handling"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check if the page requires JavaScript
            if self._requires_javascript(response.text):
                return self._get_fallback_content_for_site(url)
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text from common content containers
            content_selectors = [
                'article', 'main', '.content', '#content', 
                '.post', '.entry', '.documentation', '.rst-content',
                'p', '.body', '.page-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if len(text) > 50:  # Only include substantial text
                        content += text + " "
                        if len(content) > 2000:  # Limit content length
                            break
                if content:
                    break
            
            if not content:
                # Fallback to all paragraph text
                paragraphs = soup.find_all('p')
                content = " ".join([p.get_text().strip() for p in paragraphs[:10]])
            
            # If still no content, provide site-specific fallback
            if not content or len(content) < 100:
                return self._get_fallback_content_for_site(url)
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content).strip()
            return content[:2000]  # Limit to 2000 characters
            
        except Exception as e:
            print(f"Content extraction error for {url}: {e}")
            return self._get_fallback_content_for_site(url)
    
    def _requires_javascript(self, html_content: str) -> bool:
        """Check if page requires JavaScript to display content"""
        js_indicators = [
            "Please activate JavaScript",
            "Please enable JavaScript",
            "JavaScript is required",
            "This site requires JavaScript",
            "Enable JavaScript to proceed"
        ]
        return any(indicator in html_content for indicator in js_indicators)
    
    def _get_fallback_content_for_site(self, url: str) -> str:
        """Provide fallback content when site can't be scraped"""
        if 'docs.python.org' in url:
            return """The official Python documentation is the most comprehensive resource for Python programming. 
            It includes tutorials, language reference, library reference, and how-to guides. 
            Topics covered include: basic syntax, data structures, functions, classes, modules, 
            standard library, and advanced features like decorators and metaclasses."""
        
        elif 'pypi.org' in url:
            return """PyPI (Python Package Index) is the official repository for Python packages. 
            You can find thousands of third-party libraries and tools here. 
            Use 'pip install package_name' to install packages. Popular packages include: 
            requests (HTTP library), numpy (numerical computing), pandas (data analysis), 
            django (web framework), flask (micro web framework), and many more."""
        
        elif 'stackoverflow.com' in url:
            return """Stack Overflow is a community-driven Q&A platform where developers help each other. 
            It contains millions of Python-related questions and answers covering everything from 
            basic syntax to advanced programming concepts, debugging help, and best practices."""
        
        elif 'realpython.com' in url:
            return """Real Python provides high-quality Python tutorials, articles, and courses. 
            Topics range from beginner to advanced, covering web development, data science, 
            testing, deployment, and Python best practices with practical examples."""
        
        elif 'github.com' in url:
            return """GitHub hosts millions of Python repositories with open-source code examples, 
            libraries, frameworks, and complete projects. It's a great place to learn from 
            real-world Python code and contribute to the Python community."""
        
        else:
            return f"This resource contains Python-related information but requires JavaScript to display content. Visit {url} directly in your browser for the full content."
    
    def generate_answer(self, query: str, search_results: List[Dict]) -> str:
        """Generate a Python-focused answer using intent classification"""
        # First, try to classify the intent
        intent, confidence = self.intent_classifier.classify_intent(query)
        
        if intent and confidence >= intent.required_confidence:
            # Use the intent's response method
            try:
                response_method = getattr(self, intent.response_method)
                if intent.response_method == "_get_syntax_help":
                    # Special case for syntax help that needs the query
                    return response_method(query)
                else:
                    return response_method()
            except AttributeError:
                # Fallback if method doesn't exist
                print(f"Warning: Method {intent.response_method} not found")
                return self._get_general_python_help(query)
        
        # If no intent matched or low confidence, try web content
        if search_results:
            all_content = []
            for result in search_results[:3]:
                content = self.extract_content(result['url'])
                if (content and len(content) > 50 and 
                    not self._is_generic_content(content)):
                    all_content.append({
                        'title': result['title'],
                        'url': result['url'],
                        'content': content
                    })
            
            if all_content:
                return self._format_web_content_response(query, all_content)
        
        # Final fallback
        return self._get_general_python_help(query)
    
    def _is_generic_content(self, content: str) -> bool:
        """Check if content is generic fallback text"""
        generic_indicators = [
            "The official Python documentation",
            "PyPI (Python Package Index)",
            "Stack Overflow is a community",
            "Real Python provides",
            "GitHub hosts millions"
        ]
        return any(indicator in content for indicator in generic_indicators)
    
    def _format_web_content_response(self, query: str, all_content: List[Dict]) -> str:
        """Format response using web content"""
        answer = f"🐍 **Python Help: {query}**\n\n"
        
        for i, item in enumerate(all_content, 1):
            answer += f"**📚 Source {i}: {item['title']}**\n"
            content_preview = item['content'][:800]
            
            # Highlight code snippets if present
            if any(code_indicator in content_preview.lower() for code_indicator in ['def ', 'import ', 'class ', 'python', '>>>']):
                answer += f"💻 {content_preview}...\n"
            else:
                answer += f"{content_preview}...\n"
            
            answer += f"🔗 **Visit:** {item['url']}\n\n"
        
        # Add Python-specific tips
        answer += self._get_python_tips(query)
        return answer
    
    def _get_comprehensive_python_help(self, query: str) -> str:
        """Provide comprehensive Python help when web content isn't available"""
        query_lower = query.lower()
        
        # Game development specific help
        if any(word in query_lower for word in ['game', 'gaming', 'pygame', 'arcade', 'graphics']):
            return self._get_game_development_help()
        # GUI/Desktop application help
        elif any(word in query_lower for word in ['gui', 'desktop', 'tkinter', 'pyqt', 'kivy', 'interface']):
            return self._get_gui_development_help()
        # Web scraping help
        elif any(word in query_lower for word in ['scraping', 'scrape', 'beautifulsoup', 'selenium', 'requests']):
            return self._get_web_scraping_help()
        # Machine learning/AI help
        elif any(word in query_lower for word in ['machine learning', 'ai', 'tensorflow', 'pytorch', 'sklearn', 'model']):
            return self._get_ml_help()
        # Automation help
        elif any(word in query_lower for word in ['automation', 'automate', 'script', 'task', 'schedule']):
            return self._get_automation_help()
        # File handling help
        elif any(word in query_lower for word in ['file', 'csv', 'json', 'excel', 'read', 'write']):
            return self._get_file_handling_help()
        # Database help
        elif any(word in query_lower for word in ['database', 'sql', 'sqlite', 'mysql', 'postgresql']):
            return self._get_database_help()
        # Puzzle requests
        elif any(word in query_lower for word in ['puzzle', 'challenge', 'problem', 'exercise', 'give me']):
            return self._get_python_puzzle()
        # Detect the type of help needed
        elif any(word in query_lower for word in ['start', 'begin', 'learn', 'new', 'basic', 'how can i use']):
            return self._get_beginner_python_help()
        elif any(word in query_lower for word in ['web', 'django', 'flask', 'fastapi', 'api']):
            return self._get_web_development_help()
        elif any(word in query_lower for word in ['data', 'pandas', 'numpy', 'analysis', 'science']):
            return self._get_data_science_help()
        elif any(word in query_lower for word in ['list', 'dict', 'string', 'loop', 'function']):
            return self._get_syntax_help(query_lower)
        else:
            return self._get_general_python_help(query)
    
    def _get_beginner_python_help(self) -> str:
        """Comprehensive help for Python beginners"""
        return """🐍 **Getting Started with Python**

**1. Installation & Setup:**
- Download Python from python.org
- Install a code editor (VS Code, PyCharm)
- Use pip to install packages

**2. Basic Syntax:**
```python
# Variables
name = "Python"
age = 30

# Functions
def greet(name):
    return f"Hello, {name}!"

# Loops
for i in range(5):
    print(i)
```

**3. Essential Concepts:**
- Variables and data types
- Functions and modules
- Lists, dictionaries, and loops
- File handling and exceptions

**4. Next Steps:**
- Practice with small projects
- Learn about libraries (requests, pandas)
- Explore web development with Flask/Django

🔗 **Resources:**
- Official Tutorial: docs.python.org/3/tutorial
- Interactive Practice: repl.it
- Community: reddit.com/r/learnpython
"""

    def _get_web_development_help(self) -> str:
        """Help for web development with Python"""
        return """🌐 **Python Web Development**

**Popular Frameworks:**
- **Flask**: Lightweight, great for beginners
- **Django**: Full-featured, batteries included
- **FastAPI**: Modern, fast, great for APIs

**Flask Example:**
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

**Key Concepts:**
- Routes and URL patterns
- Templates (Jinja2)
- Forms and user input
- Database integration
- RESTful APIs

🔗 **Learn More:** flask.palletsprojects.com
"""

    def _get_data_science_help(self) -> str:
        """Help for data science with Python"""
        return """📊 **Python for Data Science**

**Essential Libraries:**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning

**Getting Started:**
```python
import pandas as pd
import numpy as np

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.head())
```

**Common Tasks:**
- Loading and cleaning data
- Statistical analysis
- Creating visualizations
- Building predictive models

🔗 **Resources:**
- Jupyter Notebooks for interactive coding
- Kaggle for datasets and competitions
"""

    def _get_syntax_help(self, query: str) -> str:
        """Help with specific Python syntax"""
        if 'list' in query:
            return """📝 **Python Lists**

```python
# Creating lists
my_list = [1, 2, 3, 'hello']
empty_list = []

# Common operations
my_list.append(4)        # Add item
my_list[0]               # Access first item
my_list[-1]              # Access last item
my_list[1:3]             # Slice
len(my_list)             # Length
```
"""
        elif 'dict' in query:
            return """📚 **Python Dictionaries**

```python
# Creating dictionaries
my_dict = {'name': 'John', 'age': 30}
empty_dict = {}

# Common operations
my_dict['name']          # Access value
my_dict['city'] = 'NYC'  # Add new key-value
my_dict.keys()           # Get all keys
my_dict.values()         # Get all values
```
"""
        elif 'function' in query:
            return """⚙️ **Python Functions**

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default parameters
def add(a, b=0):
    return a + b

# Calling functions
result = greet("Python")
sum_result = add(5, 3)
```
"""
        else:
            return self._get_general_python_help(query)

    def _get_general_python_help(self, query: str) -> str:
        """General Python help"""
        return f"""🐍 **Python Help for: {query}**

I can help you with various Python topics including:

**📚 Learning Resources:**
- Official Python Documentation
- Real Python tutorials
- Stack Overflow community
- GitHub repositories

**💡 Common Topics:**
- Basic syntax and data types
- Functions and classes
- Web development (Flask, Django)
- Data science (Pandas, NumPy)
- File handling and APIs

**🛠️ Getting Help:**
- Use help() function in Python
- Read error messages carefully
- Break problems into smaller parts
- Practice with small examples

Would you like help with a specific Python concept?
"""

    def _get_python_puzzle(self) -> str:
        """Get a random Python puzzle for the user"""
        import random
        puzzle = random.choice(python_puzzles)
        
        return f"""🧩 **Python Coding Challenge**

**Problem:** {puzzle['puzzle']}
**Difficulty:** {puzzle['difficulty'].title()}

**💡 Hint:** {puzzle['hint']}

**📝 Try to solve this on your own first, then check the solution below:**

<details>
<summary>Click to reveal solution</summary>

```python
{puzzle['solution']}
```

</details>

**🚀 Challenge yourself:**
- Try solving without looking at the hint
- Can you make the solution more efficient?
- Add error handling to make it more robust
- Test with different inputs

Want another challenge? Just ask for another puzzle! 🎯
"""

    def _get_game_development_help(self) -> str:
        """Help specifically for game development with Python"""
        return """🎮 **Python Game Development**

**Popular Game Libraries:**
- **Pygame**: Most popular, great for 2D games
- **Arcade**: Modern alternative to Pygame
- **Panda3D**: For 3D games
- **Kivy**: Cross-platform games and apps

**Simple Pygame Example:**
```python
import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("My First Game")

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Fill the screen with color
    screen.fill((0, 128, 255))  # Blue background
    
    # Update the display
    pygame.display.flip()

pygame.quit()
sys.exit()
```

**Game Development Steps:**
1. **Install Pygame**: `pip install pygame`
2. **Plan Your Game**: Decide on genre, mechanics
3. **Create Basic Window**: Set up display and game loop
4. **Add Sprites**: Characters, objects, backgrounds
5. **Handle Input**: Keyboard, mouse controls
6. **Add Game Logic**: Collisions, scoring, levels
7. **Sound & Music**: pygame.mixer for audio

**Game Ideas for Beginners:**
- Pong (ball bouncing game)
- Snake game
- Space Invaders
- Platformer
- Puzzle games

🔗 **Resources:**
- Pygame Documentation: pygame.org
- Real Python Game Tutorial
- YouTube: "Python Game Development" tutorials
"""

    def _get_gui_development_help(self) -> str:
        """Help for GUI/Desktop application development"""
        return """🖥️ **Python GUI Development**

**Popular GUI Libraries:**
- **Tkinter**: Built into Python, simple to use
- **PyQt5/6**: Professional, feature-rich
- **wxPython**: Native look and feel
- **Kivy**: Modern, touch-friendly
- **Dear PyGui**: Fast, modern GUI

**Simple Tkinter Example:**
```python
import tkinter as tk
from tkinter import messagebox

def button_click():
    messagebox.showinfo("Hello", "Button clicked!")

# Create main window
root = tk.Tk()
root.title("My GUI App")
root.geometry("300x200")

# Add widgets
label = tk.Label(root, text="Welcome to my app!")
label.pack(pady=10)

button = tk.Button(root, text="Click Me!", command=button_click)
button.pack(pady=10)

entry = tk.Entry(root)
entry.pack(pady=10)

# Start the GUI
root.mainloop()
```

**GUI Development Steps:**
1. **Choose Library**: Tkinter for simple, PyQt for advanced
2. **Design Layout**: Plan your interface
3. **Create Widgets**: Buttons, labels, text fields
4. **Handle Events**: Button clicks, key presses
5. **Add Functionality**: Connect GUI to your logic

🔗 **Learn More:**
- Tkinter Tutorial: docs.python.org/3/library/tkinter.html
- PyQt Tutorial: riverbankcomputing.com
"""

    def _get_web_scraping_help(self) -> str:
        """Help for web scraping with Python"""
        return """🕷️ **Python Web Scraping**

**Essential Libraries:**
- **Requests**: HTTP requests
- **BeautifulSoup**: HTML parsing
- **Selenium**: Browser automation
- **Scrapy**: Advanced scraping framework

**Basic Scraping Example:**
```python
import requests
from bs4 import BeautifulSoup

# Make a request to the website
url = "https://example.com"
response = requests.get(url)

# Parse the HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Extract data
titles = soup.find_all('h1')
for title in titles:
    print(title.text.strip())

# Find specific elements
price = soup.find('span', class_='price')
if price:
    print(f"Price: {price.text}")
```

**Installation:**
```bash
pip install requests beautifulsoup4 selenium
```

**Scraping Steps:**
1. **Inspect Website**: Use browser dev tools
2. **Make Request**: Use requests library
3. **Parse HTML**: Use BeautifulSoup
4. **Extract Data**: Find elements by tag, class, id
5. **Handle Errors**: Check response status
6. **Be Respectful**: Follow robots.txt, add delays

**Advanced: Selenium for Dynamic Content:**
```python
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://example.com")

# Wait for page to load
driver.implicitly_wait(10)

# Find elements
element = driver.find_element(By.CLASS_NAME, "content")
print(element.text)

driver.quit()
```

⚠️ **Important:**
- Respect website terms of service
- Don't overload servers (add delays)
- Handle errors gracefully
- Consider using APIs when available
"""

    def _get_ml_help(self) -> str:
        """Help for machine learning with Python"""
        return """🤖 **Python Machine Learning**

**Essential Libraries:**
- **Scikit-learn**: General ML algorithms
- **TensorFlow**: Deep learning
- **PyTorch**: Deep learning research
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization

**Simple ML Example:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
```

**ML Project Steps:**
1. **Define Problem**: Classification, regression, clustering
2. **Collect Data**: CSV files, APIs, databases
3. **Explore Data**: Pandas, matplotlib for analysis
4. **Prepare Data**: Clean, normalize, feature engineering
5. **Choose Algorithm**: Linear regression, random forest, neural networks
6. **Train Model**: Fit algorithm to training data
7. **Evaluate**: Test accuracy, precision, recall
8. **Deploy**: Save model, create API

**Installation:**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Learning Path:**
- Start with supervised learning (classification/regression)
- Learn data preprocessing techniques
- Understand evaluation metrics
- Practice with real datasets (Kaggle)
"""

    def _get_automation_help(self) -> str:
        """Help for automation with Python"""
        return """⚙️ **Python Automation**

**Popular Automation Libraries:**
- **Schedule**: Task scheduling
- **PyAutoGUI**: GUI automation
- **Selenium**: Web automation
- **Paramiko**: SSH automation
- **Watchdog**: File system monitoring

**File Automation Example:**
```python
import os
import shutil
from datetime import datetime

def organize_downloads():
    downloads_path = "C:/Users/YourName/Downloads"
    
    # Create folders
    folders = {
        'images': ['.jpg', '.png', '.gif'],
        'documents': ['.pdf', '.docx', '.txt'],
        'videos': ['.mp4', '.avi', '.mkv']
    }
    
    for folder in folders:
        os.makedirs(f"{downloads_path}/{folder}", exist_ok=True)
    
    # Move files
    for filename in os.listdir(downloads_path):
        if os.path.isfile(f"{downloads_path}/{filename}"):
            file_ext = os.path.splitext(filename)[1].lower()
            
            for folder, extensions in folders.items():
                if file_ext in extensions:
                    shutil.move(
                        f"{downloads_path}/{filename}",
                        f"{downloads_path}/{folder}/{filename}"
                    )
                    break

# Run the automation
organize_downloads()
```

**Email Automation:**
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(to_email, subject, message):
    from_email = "your_email@gmail.com"
    password = "your_password"  # Use app passwords for Gmail
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(message, 'plain'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()
```

**Scheduled Tasks:**
```python
import schedule
import time

def job():
    print("Running scheduled task...")
    # Your automation code here

# Schedule tasks
schedule.every(10).minutes.do(job)
schedule.every().hour.do(job)
schedule.every().day.at("10:30").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
```

**Common Automation Tasks:**
- File organization and cleanup
- Email sending and processing
- Web scraping and monitoring
- Database backups
- Report generation
- Social media posting
"""

    def _get_file_handling_help(self) -> str:
        """Help for file handling with Python"""
        return """📁 **Python File Handling**

**Basic File Operations:**
```python
# Reading files
with open('file.txt', 'r') as file:
    content = file.read()
    print(content)

# Writing files
with open('output.txt', 'w') as file:
    file.write("Hello, World!")

# Appending to files
with open('log.txt', 'a') as file:
    file.write("New log entry\\n")
```

**Working with CSV Files:**
```python
import csv

# Reading CSV
with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row['column_name'])

# Writing CSV
data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
]

with open('output.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['name', 'age'])
    writer.writeheader()
    writer.writerows(data)
```

**JSON Handling:**
```python
import json

# Reading JSON
with open('data.json', 'r') as file:
    data = json.load(file)
    print(data)

# Writing JSON
data = {'name': 'Python', 'version': 3.9}
with open('output.json', 'w') as file:
    json.dump(data, file, indent=2)
```

**Excel Files with Pandas:**
```python
import pandas as pd

# Reading Excel
df = pd.read_excel('data.xlsx')
print(df.head())

# Writing Excel
df.to_excel('output.xlsx', index=False)

# Multiple sheets
with pd.ExcelWriter('workbook.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

**File System Operations:**
```python
import os
import shutil

# Check if file exists
if os.path.exists('file.txt'):
    print("File exists")

# List directory contents
files = os.listdir('.')
for file in files:
    print(file)

# Copy files
shutil.copy('source.txt', 'destination.txt')

# Move files
shutil.move('old_location.txt', 'new_location.txt')

# Delete files
os.remove('unwanted_file.txt')
```

**Installation for Advanced Features:**
```bash
pip install pandas openpyxl xlsxwriter
```
"""

    def _get_database_help(self) -> str:
        """Help for database operations with Python"""
        return """🗄️ **Python Database Operations**

**SQLite (Built-in):**
```python
import sqlite3

# Connect to database (creates if doesn't exist)
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE
    )
''')

# Insert data
cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", 
               ("Alice", "alice@email.com"))

# Query data
cursor.execute("SELECT * FROM users")
results = cursor.fetchall()
for row in results:
    print(row)

# Close connection
conn.commit()
conn.close()
```

**MySQL with PyMySQL:**
```python
import pymysql

# Install: pip install pymysql
connection = pymysql.connect(
    host='localhost',
    user='your_username',
    password='your_password',
    database='your_database'
)

with connection.cursor() as cursor:
    # Insert
    sql = "INSERT INTO users (name, email) VALUES (%s, %s)"
    cursor.execute(sql, ('Bob', 'bob@email.com'))
    
    # Select
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
    for row in results:
        print(row)

connection.commit()
connection.close()
```

**Using Pandas for Database Operations:**
```python
import pandas as pd
import sqlite3

# Read from database
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM users", conn)
print(df)

# Write to database
new_data = pd.DataFrame({
    'name': ['Charlie', 'Diana'],
    'email': ['charlie@email.com', 'diana@email.com']
})
new_data.to_sql('users', conn, if_exists='append', index=False)

conn.close()
```

**PostgreSQL with psycopg2:**
```python
import psycopg2

# Install: pip install psycopg2-binary
conn = psycopg2.connect(
    host="localhost",
    database="your_db",
    user="your_user",
    password="your_password"
)

cur = conn.cursor()
cur.execute("SELECT version();")
record = cur.fetchone()
print(record)

cur.close()
conn.close()
```

**ORM with SQLAlchemy:**
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Install: pip install sqlalchemy
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

engine = create_engine('sqlite:///example.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# Add user
new_user = User(name='Eve', email='eve@email.com')
session.add(new_user)
session.commit()

# Query users
users = session.query(User).all()
for user in users:
    print(f"{user.name}: {user.email}")
```

**Common Libraries:**
- **sqlite3**: Built-in, good for small projects
- **pymysql/mysql-connector**: MySQL databases
- **psycopg2**: PostgreSQL databases
- **sqlalchemy**: ORM for any database
- **pandas**: Great for data analysis with databases
"""

    def _get_python_tips(self, query: str) -> str:
        """Get relevant Python tips based on query"""
        tips = """
💡 **Python Tips:**
- Use virtual environments (venv) for projects
- Follow PEP 8 style guidelines
- Write docstrings for your functions
- Use list comprehensions for cleaner code
- Handle exceptions with try/except blocks
"""
        return tips

    def _get_testing_help(self) -> str:
        """Help for testing Python code"""
        return """🧪 **Python Testing**

**Popular Testing Frameworks:**
- **unittest**: Built-in testing framework
- **pytest**: More feature-rich and user-friendly
- **doctest**: Test code in docstrings
- **mock**: Create mock objects for testing

**Basic unittest Example:**
```python
import unittest

class TestMathFunctions(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)
    
    def test_division(self):
        self.assertEqual(10 / 2, 5)
        
    def test_zero_division(self):
        with self.assertRaises(ZeroDivisionError):
            10 / 0

if __name__ == '__main__':
    unittest.main()
```

**Pytest Example:**
```python
# test_math.py
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

# Run with: pytest test_math.py
```

**Installation:**
```bash
pip install pytest pytest-cov
```
"""

    def _get_performance_help(self) -> str:
        """Help for Python performance optimization"""
        return """⚡ **Python Performance Optimization**

**Profiling Tools:**
- **cProfile**: Built-in profiler
- **line_profiler**: Line-by-line profiling
- **memory_profiler**: Memory usage profiling

**Common Optimization Techniques:**
1. **Use built-in functions** (they're implemented in C)
2. **List comprehensions** over loops
3. **Generator expressions** for memory efficiency
4. **Set lookups** instead of list searches
5. **Local variables** are faster than global

**Example - Optimized Code:**
```python
# Slow
def slow_function(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result

# Fast
def fast_function(items):
    return [item * 2 for item in items if item > 0]

# Memory efficient for large datasets
def generator_function(items):
    return (item * 2 for item in items if item > 0)
```

**Profiling Example:**
```python
import cProfile

def your_function():
    # Your code here
    pass

cProfile.run('your_function()')
```
"""

    def _get_packages_help(self) -> str:
        """Help for Python packages and libraries"""
        return """📦 **Python Packages & Libraries**

**Package Management:**
```bash
# Install a package
pip install package_name

# Install specific version
pip install package_name==1.2.3

# Install from requirements file
pip install -r requirements.txt

# List installed packages
pip list

# Show package information
pip show package_name

# Uninstall package
pip uninstall package_name
```

**Virtual Environments:**
```bash
# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\\Scripts\\activate

# Activate (Mac/Linux)
source myenv/bin/activate

# Deactivate
deactivate
```

**Popular Libraries by Category:**

**🌐 Web Development:**
- Flask, Django, FastAPI
- requests, httpx

**📊 Data Science:**
- pandas, numpy, matplotlib
- seaborn, plotly, scikit-learn

**🤖 Machine Learning:**
- tensorflow, pytorch, keras
- sklearn, xgboost

**🎮 Game Development:**
- pygame, panda3d, arcade

**🖥️ GUI Development:**
- tkinter, PyQt, kivy

**⚙️ Automation:**
- selenium, pyautogui, schedule

**Creating Your Own Package:**
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="your_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
    ],
)
```
"""

    def chat(self, message: str) -> str:
        """Main chat method that processes user messages"""
        try:
            # Search for relevant information
            search_results = self.search_web(message)
            
            # Generate response based on search results
            response = self.generate_answer(message, search_results)
            
            return response
        except Exception as e:
            return f"I'm having trouble processing your request. Please try asking about a specific Python topic. Error: {str(e)}"


# Initialize the chatbot
chatbot = PythonChatbot()



# # FAQ dictionary with keywords and answers
# faq_keywords = {
#     "list": "A list is a collection data type in Python that is ordered and mutable.",
#     "function": "Use the 'def' keyword to define functions: def function_name(parameters):",
#     "dictionary": "A dictionary stores key-value pairs. It's unordered and mutable.",
#     "loop": "Python has for loops and while loops to repeat actions.",
#     "string": "A string is a sequence of characters enclosed in quotes.",
#     "class": "Classes are blueprints for creating objects in Python.",
#     "tuple": "A tuple is an ordered, immutable collection of items.",
#     "set": "A set is an unordered collection of unique elements.",
#     "module": "A module is a file containing Python definitions and statements.",
#     "package": "A package is a collection of Python modules organized in directories.",
#     "list comprehension": "A concise way to create lists using a single line syntax.",
#     "lambda": "An anonymous function defined with the lambda keyword.",
#     "exception": "An error that occurs during program execution, handled using try-except.",
#     "iterator": "An object representing a stream of data, returned by iter().",
#     "generator": "A special iterator created using functions and the yield keyword.",
#     "decorator": "A function that modifies another function's behavior without changing its code.",
#     "variable": "A name that refers to a value stored in memory.",
#     "boolean": "A data type with two possible values: True or False.",
#     "if statement": "A conditional statement that runs code if a condition is True.",
#     "import": "Used to bring modules and functions into your script.",
#     "recursion": "A function that calls itself to solve smaller instances of a problem.",
#     "file handling": "Reading and writing files using open(), read(), write(), and close().",
#     "list slicing": "Extracting parts of a list using slice notation like list[start:end].",
#     "python version": "Common versions include Python 2.x and Python 3.x; Python 3 is current.",
#     "boolean operators": "Operators like and, or, and not used to combine boolean expressions.",
#     "data types": "Types like int, float, str, list, tuple, dict, set define the kind of data.",
# }

# Cache FAQ tokens once globally for similarity matching
# faq_tokens = {key: nlp(key) for key in faq_keywords.keys()}

# Enhanced intent detection
INTENTS = {
    "ask_hint": ["hint", "give me a hint", "suggestion", "need help"],
    "ask_solution": ["solution", "show answer", "what is the answer"],
    "confused": ["I don't get it", "I'm stuck", "no idea", "confused"],
    "ask_puzzle": ["give me a puzzle", "coding problem", "show a challenge"]
}


python_topics = {
    "Variables": "Variables are used to store information. Example: x = 10",
    "Data Types": "Common types: int, float, str, list, dict, tuple, set, bool.",
    "Functions": "Functions are defined using 'def'. Example: def greet():",
    "Loops": "Use 'for' and 'while' loops to repeat code.",
    "Conditionals": "Use if/elif/else to control the flow.",
    "Classes": "Used to create objects. Defined using 'class' keyword.",
    "Modules": "Modules are files containing Python code. Use import to include them.",
    "Exceptions": "Handle errors using try/except blocks.",
}

# Python puzzles with hints and solutions
python_puzzles = [
    {
        "puzzle": "Write a function that returns the factorial of a number.",
        "difficulty": "easy",
        "hint": "Use recursion or a loop multiplying numbers from 1 to n.",
        "solution": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    },
    {
        "puzzle": "Create a function that checks if a string is a palindrome.",
        "difficulty": "medium",
        "hint": "Compare the string to its reverse.",
        "solution": "def is_palindrome(s): return s == s[::-1]"
    },
    {
        "puzzle": "Write a function that merges two sorted lists into a sorted list.",
        "difficulty": "hard",
        "hint": "Use two pointers to traverse both lists and merge.",
        "solution": (
            "def merge_sorted(a, b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] < b[j]: result.append(a[i]); i+=1\n"
            "        else: result.append(b[j]); j+=1\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result"
        )
    },
    {
        "puzzle": "Implement a function to find the nth Fibonacci number using recursion.",
        "difficulty": "medium",
        "hint": "Use the base cases for n=0 and n=1, else sum of fib(n-1) and fib(n-2).",
        "solution": (
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n-1) + fib(n-2)"
        )
    },
    {
        "puzzle": "Write a function to count the number of vowels in a given string.",
        "difficulty": "easy",
        "hint": "Check each character if it is in 'aeiou'.",
        "solution": (
            "def count_vowels(s):\n"
            "    return sum(c in 'aeiouAEIOU' for c in s)"
        )
    },
    {
        "puzzle": "Create a function that removes duplicates from a list while preserving order.",
        "difficulty": "medium",
        "hint": "Use a set to track seen items and iterate through list.",
        "solution": (
            "def remove_duplicates(lst):\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            result.append(item)\n"
            "    return result"
        )
    },
    {
        "puzzle": "Write a function that checks whether two strings are anagrams.",
        "difficulty": "medium",
        "hint": "Sort both strings and compare, or count characters.",
        "solution": (
            "def are_anagrams(s1, s2):\n"
            "    return sorted(s1) == sorted(s2)"
        )
    },
    {
        "puzzle": "Implement a function to flatten a nested list.",
        "difficulty": "hard",
        "hint": "Use recursion to flatten lists inside lists.",
        "solution": (
            "def flatten(lst):\n"
            "    result = []\n"
            "    for el in lst:\n"
            "        if isinstance(el, list):\n"
            "            result.extend(flatten(el))\n"
            "        else:\n"
            "            result.append(el)\n"
            "    return result"
        )
    },
    {
        "puzzle": "Create a function that finds the maximum sum subarray in a list of integers (Kadane’s Algorithm).",
        "difficulty": "hard",
        "hint": "Keep track of current max and global max while iterating.",
        "solution": (
            "def max_subarray_sum(arr):\n"
            "    max_ending_here = max_so_far = arr[0]\n"
            "    for x in arr[1:]:\n"
            "        max_ending_here = max(x, max_ending_here + x)\n"
            "        max_so_far = max(max_so_far, max_ending_here)\n"
            "    return max_so_far"
        )
    },
    {
        "puzzle": "Write a function to determine if a number is prime.",
        "difficulty": "easy",
        "hint": "Check divisibility up to sqrt(n).",
        "solution": (
            "def is_prime(n):\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True"
        )
    }
]

coding_helper = [
    {
        "language": "C",
        "code": '#include <stdio.h>\nint main() {\n    printf("Hello, world\\n");\n    return 0;\n}',
        "hint": "Every C program starts with main(). Don't forget semicolons!",
        "explanation": "This program prints 'Hello, world' to the console. #include <stdio.h> allows usage of printf."
    },
    {
        "language": "Python",
        "code": 'def greet():\n    print("Hello, world")\n\ngreet()',
        "hint": "Define a function using def, then call it.",
        "explanation": "Defines greet() function that prints a message, then calls it."
    },
    {
        "language": "JavaScript",
        "code": 'function greet() {\n    console.log("Hello, world");\n}\n\ngreet();',
        "hint": "Use function keyword and console.log to print.",
        "explanation": "Defines a greet function and calls it to output to the console."
    },
    {
        "language": "Python",
        "code": 'for i in range(5):\n    print(i)',
        "hint": "Use a for loop with range to iterate 0 to 4.",
        "explanation": "Prints numbers from 0 to 4 using a for loop."
    },
    {
        "language": "C",
        "code": '#include <stdio.h>\n\nint factorial(int n) {\n    if (n <= 1)\n        return 1;\n    else\n        return n * factorial(n - 1);\n}\n\nint main() {\n    printf("%d\\n", factorial(5));\n    return 0;\n}',
        "hint": "Use recursion to calculate factorial.",
        "explanation": "factorial() calls itself to compute factorial of n recursively."
    }
]
coding_trainings = [
    {
        "language": "Python",
        "codeWithBlanks": 'def multiply(x):\n    return x <input type="text" class="blank-input" data-answer="*"> <input type="text" class="blank-input" data-answer="2">',
        "correctAnswers": ['*', '2'],
        "hint": "The function multiplies x by a number.",
        "explanation": "This function named multiply returns x multiplied by 2."
    },
    {
        "language": "Python",
        "codeWithBlanks": 'for i in range(<input type="text" class="blank-input" data-answer="5">):\n    print(i)',
        "correctAnswers": ['5'],
        "hint": "Loops 5 times printing the numbers.",
        "explanation": "This loop prints numbers from 0 to 4."
    },
    {
        "language": "Python",
        "codeWithBlanks": 'print(<input type="text" class="blank-input" data-answer="&quot;Hello, world&quot;">)',
        "correctAnswers": ['"Hello, world"'],
        "hint": "What string do we want to display?",
        "explanation": "You need to pass a string to `print()` using double quotes."
    },
    {
        "language": "Python",
        "codeWithBlanks": 'my_list = [1, 2, 3]\nmy_list.<input type="text" class="blank-input" data-answer="append">(<input type="text" class="blank-input" data-answer="4">)\nprint(my_list)',
        "correctAnswers": ['append', '4'],
        "hint": "What method adds an item to the end of a list?",
        "explanation": "The append() method adds an element to the end of a list."
    },
    {
        "language": "Python",
        "codeWithBlanks": 'def greet(name):\n    <input type="text" class="blank-input" data-answer="return"> f<input type="text" class="blank-input" data-answer="&quot;Hello, {name}!&quot;">\n\nprint(greet(<input type="text" class="blank-input" data-answer="&quot;Python&quot;">))',
        "correctAnswers": ['return', '"Hello, {name}!"', '"Python"'],
        "hint": "Functions use a keyword to send back values, and f-strings for formatting.",
        "explanation": "Functions use 'return' to send back values, and f-strings allow variable interpolation in strings."
    },
    {
        "language": "Python",
        "codeWithBlanks": 'if <input type="text" class="blank-input" data-answer="5"> > <input type="text" class="blank-input" data-answer="3">:\n    print(<input type="text" class="blank-input" data-answer="&quot;Five is greater&quot;">)\n<input type="text" class="blank-input" data-answer="else">:\n    print(<input type="text" class="blank-input" data-answer="&quot;Three is greater&quot;">)',
        "correctAnswers": ['5', '3', '"Five is greater"', 'else', '"Three is greater"'],
        "hint": "Complete the if-else statement with numbers and strings.",
        "explanation": "This demonstrates a basic if-else conditional statement in Python."
    }
]

# Function to extract keywords from text using spaCy


# def extract_keywords(text):
#     """Extract nouns and proper nouns as spaCy tokens."""
#     doc = nlp(text.lower())
#     keywords = [token for token in doc if token.pos_ in {"NOUN", "PROPN"}]
#     return keywords

# def answer_question(question, threshold=0.7):
#     question = question.lower().strip()
#     question_keywords = extract_keywords(question)
#     if not question_keywords:
#         return "Sorry, I couldn't understand your question." #Try asking for a Python puzzle!"

#     best_match = None
#     best_score = 0

#     for faq_key, faq_token in faq_tokens.items():
#         # Compute average similarity of all question keywords to this faq token
#         sim_scores = [qk.similarity(faq_token) for qk in question_keywords]
#         avg_score = sum(sim_scores) / len(sim_scores) if sim_scores else 0
#         if avg_score > best_score and avg_score >= threshold:
#             best_score = avg_score
#             best_match = faq_key

#     if best_match:
#         return faq_keywords[best_match]
#     else:
#         return "Sorry, I don't have an answer for that." #Try asking for a Python puzzle!"

def get_random_puzzle():
    puzzle = random.choice(python_puzzles)
    return puzzle

@app.route('/topics')
def topics():
    return render_template('topics.html', topics=python_topics)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    user_message = data.get('message', '') if data else ''
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get response from chatbot
    response = chatbot.chat(user_message)
    
    return jsonify({'response': response})


@app.route('/training')
def training():
    return render_template('training.html', trainings=coding_trainings)


if __name__ == '__main__':
    app.run(debug=True)
# app.py - A simple Flask app for Python FAQs and puzzles
# It uses spaCy for keyword extraction and provides a chat interface.
# The app includes a training section with coding exercises and solutions.
# It also supports hints and solutions for coding puzzles.