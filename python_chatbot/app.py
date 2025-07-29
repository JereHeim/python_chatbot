from flask import Flask, request, jsonify, render_template
import random
import spacy
from flask import session

app = Flask(__name__)
app.secret_key = 'your-very-secret-key'  # Replace with a secure secret key

# Load medium model with vectors once
nlp = spacy.load("en_core_web_md")

# FAQ dictionary with keywords and answers
faq_keywords = {
    "list": "A list is a collection data type in Python that is ordered and mutable.",
    "function": "Use the 'def' keyword to define functions: def function_name(parameters):",
    "dictionary": "A dictionary stores key-value pairs. It's unordered and mutable.",
    "loop": "Python has for loops and while loops to repeat actions.",
    "string": "A string is a sequence of characters enclosed in quotes.",
    "class": "Classes are blueprints for creating objects in Python.",
    "tuple": "A tuple is an ordered, immutable collection of items.",
    "set": "A set is an unordered collection of unique elements.",
    "module": "A module is a file containing Python definitions and statements.",
    "package": "A package is a collection of Python modules organized in directories.",
    "list comprehension": "A concise way to create lists using a single line syntax.",
    "lambda": "An anonymous function defined with the lambda keyword.",
    "exception": "An error that occurs during program execution, handled using try-except.",
    "iterator": "An object representing a stream of data, returned by iter().",
    "generator": "A special iterator created using functions and the yield keyword.",
    "decorator": "A function that modifies another function's behavior without changing its code.",
    "variable": "A name that refers to a value stored in memory.",
    "boolean": "A data type with two possible values: True or False.",
    "if statement": "A conditional statement that runs code if a condition is True.",
    "import": "Used to bring modules and functions into your script.",
    "recursion": "A function that calls itself to solve smaller instances of a problem.",
    "file handling": "Reading and writing files using open(), read(), write(), and close().",
    "list slicing": "Extracting parts of a list using slice notation like list[start:end].",
    "python version": "Common versions include Python 2.x and Python 3.x; Python 3 is current.",
    "boolean operators": "Operators like and, or, and not used to combine boolean expressions.",
    "data types": "Types like int, float, str, list, tuple, dict, set define the kind of data.",
}

# Cache FAQ tokens once globally for similarity matching
faq_tokens = {key: nlp(key) for key in faq_keywords.keys()}

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
        "code": """\
#include <stdio.h>
int main() {
    printf("Hello, world\\n");
    return 0;
}""",
        "hint": "Every C program starts with main(). Don't forget semicolons!",
        "explanation": "This program prints 'Hello, world' to the console. #include <stdio.h> allows usage of printf."
    },
    {
        "language": "Python",
        "code": """\
def greet():
    print("Hello, world")

greet()""",
        "hint": "Define a function using def, then call it.",
        "explanation": "Defines greet() function that prints a message, then calls it."
    },
    {
        "language": "JavaScript",
        "code": """\
function greet() {
    console.log("Hello, world");
}

greet();""",
        "hint": "Use function keyword and console.log to print.",
        "explanation": "Defines a greet function and calls it to output to the console."
    },
    {
        "language": "Python",
        "code": """\
for i in range(5):
    print(i)""",
        "hint": "Use a for loop with range to iterate 0 to 4.",
        "explanation": "Prints numbers from 0 to 4 using a for loop."
    },
    {
        "language": "C",
        "code": """\
#include <stdio.h>

int factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorial(n - 1);
}

int main() {
    printf("%d\\n", factorial(5));
    return 0;
}""",
        "hint": "Use recursion to calculate factorial.",
        "explanation": "factorial() calls itself to compute factorial of n recursively."
    }
]
coding_trainings = [
    {
        "language": "Python",
        "codeWithBlanks": "def {{blank0}}(x):\n    return x * {{blank1}}",
        "blanks": ["multiply", "2"],
        "hint": "The function multiplies x by a number.",
        "explanation": "This function named multiply returns x multiplied by 2."
    },
    {
        "language": "Python",
        "codeWithBlanks": "for i in range({{blank0}}):\n    print(i)",
        "blanks": ["5"],
        "hint": "Loops 5 times printing the numbers.",
        "explanation": "This loop prints numbers from 0 to 4."
    }
]


def extract_keywords(text):
    """Extract nouns and proper nouns as spaCy tokens."""
    doc = nlp(text.lower())
    keywords = [token for token in doc if token.pos_ in {"NOUN", "PROPN"}]
    return keywords

def answer_question(question, threshold=0.7):
    question = question.lower().strip()
    question_keywords = extract_keywords(question)
    if not question_keywords:
        return "Sorry, I couldn't understand your question. Try asking for a Python puzzle!"

    best_match = None
    best_score = 0

    for faq_key, faq_token in faq_tokens.items():
        # Compute average similarity of all question keywords to this faq token
        sim_scores = [qk.similarity(faq_token) for qk in question_keywords]
        avg_score = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        if avg_score > best_score and avg_score >= threshold:
            best_score = avg_score
            best_match = faq_key

    if best_match:
        return faq_keywords[best_match]
    else:
        return "Sorry, I don't have an answer for that. Try asking for a Python puzzle!"

def get_random_puzzle():
    puzzle = random.choice(python_puzzles)
    return puzzle

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()

    if user_message == "":
        response = "Hi! Ask me about Python concepts like lists, functions, classes, loops, or ask for a Python puzzle!"
    elif "puzzle" in user_message.lower():
        puzzle_index = random.randint(0, len(python_puzzles) - 1)
        session['last_puzzle_index'] = puzzle_index
        puzzle = python_puzzles[puzzle_index]
        response = f"Puzzle ({puzzle['difficulty']}): {puzzle['puzzle']}\nType 'hint' or 'solution' for help."
    elif user_message.lower() == "hint":
        if 'last_puzzle_index' in session:
            puzzle = python_puzzles[session['last_puzzle_index']]
            response = f"Hint: {puzzle['hint']}"
        else:
            response = "Try asking for a puzzle first!"
    elif user_message.lower() == "solution":
        if 'last_puzzle_index' in session:
            puzzle = python_puzzles[session['last_puzzle_index']]
            response = f"Solution:\n{puzzle['solution']}"
        else:
            response = "Try asking for a puzzle first!"
    else:
        response = answer_question(user_message)

    return jsonify({'response': response})

@app.route('/training')
def training():
    return render_template('training.html', trainings=coding_trainings)


if __name__ == '__main__':
    app.run(debug=True)
# app.py - A simple Flask app for Python FAQs and puzzles
# It uses spaCy for keyword extraction and provides a chat interface.