import json
import random
import string

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gemini_helper import ask_gemini, transcribe, speak, get_api_key

st.set_page_config(page_title="English Learning Chatbot", page_icon="📚")


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


@st.cache_data
def load_intents(filepath="intents.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def build_matcher(_intents_data):
    all_patterns = []
    all_tags = []

    for intent in _intents_data["intents"]:
        for pattern in intent["patterns"]:
            all_patterns.append(preprocess(pattern))
            all_tags.append(intent["tag"])

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_patterns)
    return vectorizer, tfidf_matrix, all_tags


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
    word = word.lower().strip(" .!?")
    for entry in intents_data["vocabulary"]:
        if entry["word"] == word:
            if want_hindi:
                return f"'{word}' in Hindi means: {entry['hindi']}"
            return f"'{word}' means: {entry['meaning']}"
    return f"Sorry, I don't have '{word}' in my vocabulary list yet."


def handle_grammar(topic, intents_data):
    topic = topic.lower().strip(" .!?")
    for entry in intents_data["grammar"]:
        if entry["topic"] == topic:
            return entry["explanation"]

    available_topics = ", ".join(item["topic"] for item in intents_data["grammar"])
    return (
        f"I don't have a grammar lesson on '{topic}' yet.\n\n"
        f"Try one of these topics: {available_topics}"
    )


def start_quiz():
    st.session_state.quiz_active = True
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0

    first_q = st.session_state.quiz_questions[0]
    options_text = "\n".join(f"- {opt}" for opt in first_q["options"])

    add_bot_message(
        f"Great! Let's start a short English quiz.\n\n"
        f"**Q1: {first_q['question']}**\n{options_text}\n\n"
        f"Type your answer below."
    )


def answer_quiz(user_answer):
    questions = st.session_state.quiz_questions
    index = st.session_state.quiz_index
    current_q = questions[index]

    if user_answer.strip().strip(".!?").lower() == current_q["answer"].lower():
        st.session_state.quiz_score += 1
        feedback = "✅ Correct! Well done."
    else:
        feedback = f"❌ Not quite. The correct answer is **{current_q['answer']}**."

    st.session_state.quiz_index += 1

    if st.session_state.quiz_index < len(questions):
        next_q = questions[st.session_state.quiz_index]
        options_text = "\n".join(f"- {opt}" for opt in next_q["options"])
        add_bot_message(
            f"{feedback}\n\n"
            f"**Q{st.session_state.quiz_index + 1}: {next_q['question']}**\n{options_text}"
        )
    else:
        score = st.session_state.quiz_score
        total = len(questions)
        add_bot_message(
            f"{feedback}\n\n"
            f"🎉 Quiz finished! You scored **{score} out of {total}**."
        )
        st.session_state.quiz_active = False


def add_user_message(text):
    st.session_state.messages.append({"role": "user", "content": text})


def add_bot_message(text):
    st.session_state.messages.append({"role": "assistant", "content": text})


def process_input(user_input, intents_data, vectorizer, tfidf_matrix, all_tags):
    add_user_message(user_input)
    lower_input = user_input.lower().strip().strip(".!?")

    if st.session_state.quiz_active:
        answer_quiz(user_input)
        return

    if lower_input in ("bye", "goodbye", "exit", "quit"):
        add_bot_message("Goodbye! Keep practising your English every day. 👋")
        return

    if lower_input == "quiz":
        start_quiz()
        return

    if lower_input.startswith("hindi "):
        word = user_input.strip()[6:]
        add_bot_message(handle_vocabulary(word, intents_data, want_hindi=True))
        return

    if lower_input.startswith("meaning "):
        word = user_input.strip()[8:].replace("of ", "").strip()
        add_bot_message(handle_vocabulary(word, intents_data, want_hindi=False))
        return

    if lower_input.startswith("grammar "):
        topic = user_input.strip()[8:].strip()
        add_bot_message(handle_grammar(topic, intents_data))
        return

    best_tag, score = get_best_intent(user_input, vectorizer, tfidf_matrix, all_tags)

    if best_tag is not None:
        add_bot_message(get_intent_response(best_tag, intents_data))
    elif get_api_key():
        # Not in the built-in list -> ask Gemini (history excludes this new message)
        with st.spinner("Thinking..."):
            add_bot_message(ask_gemini(user_input, st.session_state.messages[:-1]))
    else:
        add_bot_message(random.choice(intents_data["fallback"]))


intents_data = load_intents("intents.json")
vectorizer, tfidf_matrix, all_tags = build_matcher(intents_data)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.quiz_active = False
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_questions = intents_data["quiz"]
    st.session_state.speak_pending = False
    st.session_state.last_audio = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I am your English learning buddy. 📚\n\n"
            "Type **help** to see what I can do, or use the buttons in the sidebar."
        ),
    })

st.title("📚 English Learning Chatbot")
st.caption("For Rural Schools — NLTK + scikit-learn, with Google Gemini and voice")

with st.sidebar:
    st.header("Quick Actions")

    if st.button("❓ Help"):
        process_input("help", intents_data, vectorizer, tfidf_matrix, all_tags)
        st.rerun()

    if st.button("📝 Start Quiz"):
        process_input("quiz", intents_data, vectorizer, tfidf_matrix, all_tags)
        st.rerun()

    st.divider()
    st.subheader("AI & Voice")
    st.text_input(
        "Gemini API key",
        type="password",
        key="gemini_key_input",
        help="Free key from aistudio.google.com. Not needed if it is set in Streamlit Secrets.",
    )
    voice_on = st.toggle("🔊 Speak replies", value=True)

    st.divider()
    st.subheader("Vocabulary List")
    for entry in intents_data["vocabulary"]:
        st.write(f"**{entry['word']}** — {entry['meaning']}")

    st.divider()
    st.subheader("Grammar Topics")
    for entry in intents_data["grammar"]:
        st.write(f"- {entry['topic']}")

    st.divider()
    if st.button("🔄 Restart Chat"):
        st.session_state.clear()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Read the newest bot reply aloud
if voice_on and st.session_state.get("speak_pending"):
    st.session_state.speak_pending = False
    last = st.session_state.messages[-1]
    if last["role"] == "assistant":
        try:
            st.audio(speak(last["content"]), format="audio/mp3", autoplay=True)
        except Exception:
            pass  # audio failing must never break the chat

# Microphone input
audio = st.audio_input("🎤 Speak to the bot")
if audio is not None:
    audio_bytes = audio.getvalue()
    audio_id = hash(audio_bytes)
    if st.session_state.get("last_audio") != audio_id:
        st.session_state.last_audio = audio_id
        if not get_api_key():
            st.warning("Add your Gemini API key in the sidebar to use voice.")
        else:
            with st.spinner("Listening..."):
                spoken = transcribe(audio_bytes)
            if spoken:
                process_input(spoken, intents_data, vectorizer, tfidf_matrix, all_tags)
                st.session_state.speak_pending = True
                st.rerun()
            else:
                st.warning("Sorry, I could not understand that. Please try again.")

user_input = st.chat_input("Type a message... (try 'hi', 'help', 'meaning water', 'quiz')")

if user_input:
    process_input(user_input, intents_data, vectorizer, tfidf_matrix, all_tags)
    st.session_state.speak_pending = True
    st.rerun()
