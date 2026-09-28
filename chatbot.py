import io
import json
import os
import random
import re
import string
import wave
from getpass import getpass

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Optional libraries: the bot still works without them ----
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

try:
    from gtts import gTTS
    import pygame
except ImportError:
    gTTS = None
    pygame = None

# ---- Gemini settings ----
MODEL = "gemini-2.5-flash"  # change if Google retires this model name
SYSTEM_PROMPT = (
    "You are a friendly English-learning buddy for school students in rural India. "
    "Use simple words and short sentences. When it helps, add a Hindi translation. "
    "Gently correct grammar mistakes. Keep answers under 120 words."
)
RECORD_SECONDS = 6
SAMPLE_RATE = 16000

CLIENT = None       # set in setup_gemini()
VOICE_ON = False    # spoken replies on/off
HISTORY = []        # [(role, text), ...] for Gemini context


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


# =====================  Gemini + voice  =====================

def setup_gemini():
    """Read the key from GEMINI_API_KEY, or ask for it (press Enter to skip)."""
    global CLIENT
    if genai is None:
        print("(Gemini is off: run  pip install google-genai  to enable it.)\n")
        return

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = getpass("Paste your Gemini API key (or press Enter to skip): ").strip()
    if key:
        CLIENT = genai.Client(api_key=key)
        print("Gemini is ON.\n")
    else:
        print("Gemini is OFF. The bot will use its built-in answers only.\n")


def ask_gemini(user_text):
    contents = [
        types.Content(role=role, parts=[types.Part(text=text)])
        for role, text in HISTORY[-8:]
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    try:
        resp = CLIENT.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        answer = resp.text or "Sorry, I could not answer that."
    except Exception as e:
        return f"Gemini error: {e}"

    HISTORY.append(("user", user_text))
    HISTORY.append(("model", answer))
    return answer


def listen():
    """Record from the microphone, then let Gemini turn speech into text."""
    if sd is None:
        print("Bot: Voice input needs:  pip install sounddevice numpy\n")
        return ""
    if CLIENT is None:
        print("Bot: Voice input needs a Gemini API key.\n")
        return ""

    print(f"Bot: Listening for {RECORD_SECONDS} seconds... speak now!")
    try:
        audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="int16")
        sd.wait()
    except Exception as e:
        print(f"Bot: I could not use the microphone ({e}).\n")
        return ""

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())

    try:
        resp = CLIENT.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
                "Transcribe this audio exactly. Reply with only the transcription.",
            ],
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"Bot: I could not understand the audio ({e}).\n")
        return ""


def speak(text):
    """Read a reply aloud (gTTS is free, needs internet, no key)."""
    if gTTS is None or pygame is None:
        return
    clean = re.sub(r"[*_`#>]", "", text)[:600]
    try:
        buf = io.BytesIO()
        gTTS(clean, lang="en").write_to_fp(buf)
        buf.seek(0)
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(buf, "mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception:
        pass  # never crash the chat because audio failed


def say(text):
    print("Bot:", text, "\n")
    if VOICE_ON:
        speak(text)


def get_input(prompt):
    """Typed input. Type 'voice' (or 'v') to speak instead."""
    text = input(prompt).strip()
    if text.lower() in ("voice", "v"):
        text = listen()
        if text:
            print(f"(heard) {text}")
    return text


# =====================  Quiz + chat loop  =====================

def run_quiz(intents_data):
    questions = intents_data["quiz"]
    score = 0

    print("\nBot: Great! Let's start a short English quiz. Answer using the option text.")
    print("     (Type 'voice' to answer by speaking.)\n")

    for i, q in enumerate(questions, start=1):
        print(f"Q{i}: {q['question']}")
        for option in q["options"]:
            print(f"   - {option}")

        user_answer = get_input("Your answer: ").strip().strip(".").lower()

        if user_answer == q["answer"].lower():
            print("Bot: Correct! Well done.\n")
            score += 1
        else:
            print(f"Bot: Not quite. The correct answer is '{q['answer']}'.\n")

    say(f"Quiz finished! You scored {score} out of {len(questions)}.")


def chat():
    global VOICE_ON
    intents_data = load_intents("intents.json")

    all_patterns, all_tags = build_corpus(intents_data)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_patterns)

    print("=" * 60)
    print(" English Learning Chatbot for Rural Schools ")
    print("=" * 60)
    setup_gemini()
    print("Bot: Hello! Type 'help' to see what I can do, or 'bye' to exit.")
    print("     Extra: type 'voice' to speak, 'speak on' / 'speak off' for spoken replies.\n")

    while True:
        user_input = get_input("You: ")

        if user_input == "":
            print("Bot: Please type something.\n")
            continue

        lower_input = user_input.lower().strip(".!?")

        if lower_input in ("bye", "goodbye", "exit", "quit"):
            say("Goodbye! Keep practising your English every day.")
            break

        if lower_input == "speak on":
            if gTTS is None or pygame is None:
                print("Bot: Spoken replies need:  pip install gtts pygame\n")
            else:
                VOICE_ON = True
                say("Spoken replies are on.")
            continue

        if lower_input == "speak off":
            VOICE_ON = False
            print("Bot: Spoken replies are off.\n")
            continue

        if lower_input == "quiz":
            run_quiz(intents_data)
            continue

        if lower_input.startswith("hindi "):
            word = user_input[6:]
            say(handle_vocabulary(word, intents_data, want_hindi=True))
            continue

        if lower_input.startswith("meaning "):
            word = user_input[8:].replace("of ", "").strip()
            say(handle_vocabulary(word, intents_data, want_hindi=False))
            continue

        if lower_input.startswith("grammar "):
            topic = user_input[8:].strip()
            say(handle_grammar(topic, intents_data))
            continue

        best_tag, score = get_best_intent(user_input, vectorizer, tfidf_matrix, all_tags)

        if best_tag is not None:
            say(get_intent_response(best_tag, intents_data))
        elif CLIENT is not None:
            say(ask_gemini(user_input))
        else:
            say(random.choice(intents_data["fallback"]))


if __name__ == "__main__":
    chat()
