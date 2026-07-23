import json
import random
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_data():
    required_packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package_name in required_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                pass


ensure_nltk_data()

lemmatizer = WordNetLemmatizer()

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = {"is", "am", "are", "the", "a", "an", "to", "of", "and", "in"}


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in STOPWORDS and word.strip() != ""
    ]

    return " ".join(cleaned_tokens)


def load_intents(filepath="intents.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(intents_data):
    all_patterns = []
    all_tags = []

    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            all_patterns.append(preprocess(pattern))
            all_tags.append(intent["tag"])

    return all_patterns, all_tags


def get_best_intent(user_text, vectorizer, tfidf_matrix, all_tags, threshold=0.35):
    cleaned_input = preprocess(user_text)

    if cleaned_input.strip() == "":
        return None, 0.0

    user_vector = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score >= threshold:
        return all_tags[best_index], best_score

    return None, best_score


def get_intent_response(tag, intents_data):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return None


def handle_vocabulary(word, intents_data, want_hindi=False):
    word = word.lower().strip()
    for entry in intents_data["vocabulary"]:
        if entry["word"] == word:
            if want_hindi:
                return f"'{word}' in Hindi means: {entry['hindi']}"
            return f"'{word}' means: {entry['meaning']}"

    return f"Sorry, I don't have '{word}' in my vocabulary list yet."


def handle_grammar(topic, intents_data):
    topic = topic.lower().strip()
    for entry in intents_data["grammar"]:
        if entry["topic"] == topic:
            return entry["explanation"]

    available_topics = ", ".join(item["topic"] for item in intents_data["grammar"])
    return (
        f"I don't have a grammar lesson on '{topic}' yet.\n"
        f"Try one of these topics: {available_topics}"
    )


def run_quiz(intents_data):
    questions = intents_data["quiz"]
    score = 0

    print("\nBot: Great! Let's start a short English quiz. Answer using the option text.\n")

    for i, q in enumerate(questions, start=1):
        print(f"Q{i}: {q['question']}")
        for option in q["options"]:
            print(f"   - {option}")

        user_answer = input("Your answer: ").strip().lower()

        if user_answer == q["answer"].lower():
            print("Bot: Correct! Well done.\n")
            score += 1
        else:
            print(f"Bot: Not quite. The correct answer is '{q['answer']}'.\n")

    print(f"Bot: Quiz finished! You scored {score} out of {len(questions)}.\n")


def chat():
    intents_data = load_intents("intents.json")

    all_patterns, all_tags = build_corpus(intents_data)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_patterns)

    print("=" * 60)
    print(" English Learning Chatbot for Rural Schools ")
    print("=" * 60)
    print("Bot: Hello! Type 'help' to see what I can do, or 'bye' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input == "":
            print("Bot: Please type something.\n")
            continue

        lower_input = user_input.lower()

        if lower_input in ("bye", "goodbye", "exit", "quit"):
            print("Bot: Goodbye! Keep practising your English every day.\n")
            break

        if lower_input == "quiz":
            run_quiz(intents_data)
            continue

        if lower_input.startswith("hindi "):
            word = user_input[6:]
            print("Bot:", handle_vocabulary(word, intents_data, want_hindi=True))
            print()
            continue

        if lower_input.startswith("meaning "):
            word = user_input[8:].replace("of ", "").strip()
            print("Bot:", handle_vocabulary(word, intents_data, want_hindi=False))
            print()
            continue

        if lower_input.startswith("grammar "):
            topic = user_input[8:].strip()
            print("Bot:", handle_grammar(topic, intents_data))
            print()
            continue

        best_tag, score = get_best_intent(user_input, vectorizer, tfidf_matrix, all_tags)

        if best_tag is not None:
            response = get_intent_response(best_tag, intents_data)
            print("Bot:", response, "\n")
        else:
            print("Bot:", random.choice(intents_data["fallback"]), "\n")


if __name__ == "__main__":
    chat()
