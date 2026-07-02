import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import sqlite3

import numpy as np
import pandas as pd
import re

import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from topic_modelling import Topic_modeling

app = Flask(__name__)

# Load saved components
voting_model = joblib.load("Models/voting_model.pkl")
tfidf = joblib.load("Models/tfidf_vectorizer.pkl")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/logon')
def logon():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('signin.html')




@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    if not message.strip():
        return render_template('home.html', type=None, alert="Please enter a message.", info=None)

    processed = preprocess_text(message)
    X = tfidf.transform([processed])
    pred = int(voting_model.predict(X)[0])
    print(pred)

    type_map = {
        0: ("Age", "This message is related to age-based cyberbullying.", "Be aware of age-related discrimination and report if necessary."),
        1: ("Ethnicity", "This message is related to ethnicity-based cyberbullying.", "Ethnic bullying is serious. Consider reporting and seeking support."),
        2: ("Gender", "This message is related to gender-based cyberbullying.", "Gender-based bullying can be harmful. Reach out for help if needed."),
        3: ("Not Cyberbullying", "This message does not appear to be cyberbullying.", "No action needed. Continue to use social media responsibly."),
        4: ("Other Cyberbullying", "This message is related to other forms of cyberbullying.", "Stay vigilant and report any suspicious activity."),
        5: ("Religion", "This message is related to religion-based cyberbullying.", "Religious bullying is unacceptable. Report and seek support if affected.")
    }

    type_str, alert, info = type_map.get(pred, ("Unknown", "Unable to classify the message.", "Please try again with a different message."))

    # Topic modeling integration using imported function

    df = pd.DataFrame({'sentence': [message]})
    topic_label, topic_keywords = Topic_modeling(df)

    return render_template('home.html', type=type_str, alert=alert, info=info, topic_label=topic_label, topic_keywords=topic_keywords)



@app.errorhandler(404)
def page_not_found(e):
    return render_template('notfound.html'), 404


if __name__ == "__main__":
    app.run(debug=True)
