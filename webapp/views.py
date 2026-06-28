import os
import pandas as pd
import pdfplumber
import re
import requests
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64



#  AUTH 
def signup_view(request):
    if request.method == "POST":
        if request.POST.get("password1") != request.POST.get("password2"):
            messages.error(request, "Passwords do not match")
            return redirect("signup")

        User.objects.create_user(
            username=request.POST.get("username"),
            password=request.POST.get("password1")
        )
        return redirect("login")

    return render(request, "signup.html")


def login_view(request):
    if request.method == "POST":
        user = authenticate(
            request,
            username=request.POST.get("username"),
            password=request.POST.get("password")
        )
        if user:
            login(request, user)
            return redirect("home")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required(login_url="login")
def index(request):
    return render(request, "index.html")


# PROJECT 1 : Cinnamon quality classification
CINNAMON_RANGES = {
    "moisture": (10.03, 13.93),
    "ash": (5.53, 7.49),
    "oil": (0.61, 1.49),
    "acid": (0.20, 0.79),
    "chromium": (0.0010, 0.0039),
    "coumarin": (0.0021, 0.0194),
}

def validate(values, ranges):
    errors = []

    for field, value in values.items():
        lo, hi = ranges[field]

        if not (lo <= value <= hi):
            errors.append(f"{field} must be between {lo} and {hi}")

    return errors

@login_required(login_url="login")
def project1(request):

    if request.method != "POST":
        return render(request, "project1.html")

    #  INPUT 
    try:
        inputs = {
            "moisture": float(request.POST.get("moisture")),
            "ash": float(request.POST.get("ash")),
            "oil": float(request.POST.get("oil")),
            "acid": float(request.POST.get("acid")),
            "chromium": float(request.POST.get("chromium")),
            "coumarin": float(request.POST.get("coumarin")),
        }
    except:
        return render(request, "project1.html", {"error": "Invalid input"})

    #  VALIDATION
    errors = validate(inputs, CINNAMON_RANGES)
    if errors:
        return render(request, "project1.html", {"error": errors})

    #  LOAD DATA 
    path = r"C:\Users\b0459\OneDrive\Desktop\project inter\2026\cinnamon\balanced_cinnamon_quality_dataset.csv"

    df = pd.read_csv(path)

    #  FIX COLUMN NAMES 
    df = df.rename(columns={
        "Moisture (%)": "moisture",
        "Ash (%)": "ash",
        "Volatile_Oil (%)": "oil",
        "Acid_Insoluble_Ash (%)": "acid",
        "Chromium (mg/kg)": "chromium",
        "Coumarin (mg/kg)": "coumarin",
        "Quality_Label": "quality_label"
    })

    # Remove ID column if exists
    if "Sample_ID" in df.columns:
        df = df.drop("Sample_ID", axis=1)


    X = df.drop("quality_label", axis=1)
    y = df["quality_label"]


    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    sample_df = pd.DataFrame([inputs])
    sample_df = sample_df[X.columns]   # correct order

    #  MODELS 
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    lr_model = LogisticRegression(max_iter=500)
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=3)

    #  TRAIN 
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    # PREDICTIONS 
    lr_pred = le.inverse_transform(lr_model.predict(sample_df))[0]
    dt_pred = le.inverse_transform(dt_model.predict(sample_df))[0]
    rf_pred = le.inverse_transform(rf_model.predict(sample_df))[0]
    knn_pred = le.inverse_transform(knn_model.predict(sample_df))[0]

    #  ACCURACY 
    lr_acc = lr_model.score(X_test, y_test) * 100
    dt_acc = dt_model.score(X_test, y_test) * 100
    rf_acc = rf_model.score(X_test, y_test) * 100
    knn_acc = knn_model.score(X_test, y_test) * 100

    #  CONFIDENCE 
    lr_conf = max(lr_model.predict_proba(sample_df)[0]) * 100
    dt_conf = max(dt_model.predict_proba(sample_df)[0]) * 100
    rf_conf = max(rf_model.predict_proba(sample_df)[0]) * 100
    knn_conf = max(knn_model.predict_proba(sample_df)[0]) * 100

    #  BEST MODEL 
    scores = [lr_acc, dt_acc, rf_acc, knn_acc]
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"]

    max_score = max(scores)

    best_models = [models[i] for i, s in enumerate(scores) if s == max_score]

    if len(best_models) > 1:
      best_model = ", ".join(best_models)
    else:
       best_model = best_models[0]
    #  GRAPH  
    plt.figure(figsize=(6,4))
    plt.bar(["LR", "DT", "RF", "KNN"], scores)
    plt.title("Model Accuracy Comparison")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    graph = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()

    #  FINAL OUTPUT 
    final_prediction = rf_pred
    final_conf = rf_conf

    #  SUGGESTION 
    if final_prediction.lower() == "low":
        suggestion = "suggestion: Reduce moisture and coumarin levels"
    elif final_prediction.lower() == "medium":
        suggestion = "suggestion: Improve volatile oil content"
    elif final_prediction.lower() == "high":
        suggestion = "suggestion: Maintain current quality"
    else:
        suggestion = "Unknown result"

    # RETURN 
    return render(request, "proj1result.html", {
        "final_prediction": final_prediction,
        "final_conf": round(final_conf, 2),

        "lr": lr_pred,
        "dt": dt_pred,
        "rf": rf_pred,
        "knn": knn_pred,

        "lr_acc": round(lr_acc, 2),
        "dt_acc": round(dt_acc, 2),
        "rf_acc": round(rf_acc, 2),
        "knn_acc": round(knn_acc, 2),

        "lr_conf": round(lr_conf, 2),
        "dt_conf": round(dt_conf, 2),
        "rf_conf": round(rf_conf, 2),
        "knn_conf": round(knn_conf, 2),

        "best": best_model,
        "inputs": inputs,
        "suggestion": suggestion,

        "graph": graph
    })

# PROJECT 2 : Crop Recommendation 
@login_required(login_url="login")
def project2(request):

    if request.method != "POST":
        return render(request, "project2.html")

    #  INPUT 
    try:
        inputs = {
            "N": float(request.POST.get("n")),
            "P": float(request.POST.get("p")),
            "K": float(request.POST.get("k")),
            "temperature": float(request.POST.get("temperature")),
            "humidity": float(request.POST.get("humidity")),
            "ph": float(request.POST.get("ph")),
            "rainfall": float(request.POST.get("rainfall")),
        }
    except:
        return render(request, "project2.html", {"error": "Invalid input"})

    #  LOAD DATA 
    path = r"C:\Users\b0459\OneDrive\Desktop\project inter\2026\Crop_recommendation\Crop_recommendation.csv"
    df = pd.read_csv(path)

    X = df.drop("label", axis=1)
    y = df["label"]

    # ENCODE 
    le = LabelEncoder()
    y = le.fit_transform(y)

    #  SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #  SAMPLE 
    sample_df = pd.DataFrame([inputs])
    sample_df = sample_df[X.columns]

    #  MODELS
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    lr_model = LogisticRegression(max_iter=500)
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=3)

    #  TRAIN 
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    # PREDICTIONS 
    lr_pred = le.inverse_transform(lr_model.predict(sample_df))[0]
    dt_pred = le.inverse_transform(dt_model.predict(sample_df))[0]
    rf_pred = le.inverse_transform(rf_model.predict(sample_df))[0]
    knn_pred = le.inverse_transform(knn_model.predict(sample_df))[0]

    #  ACCURACY 
    lr_acc = lr_model.score(X_test, y_test) * 100
    dt_acc = dt_model.score(X_test, y_test) * 100
    rf_acc = rf_model.score(X_test, y_test) * 100
    knn_acc = knn_model.score(X_test, y_test) * 100

    #  CONFIDENCE 
    lr_conf = max(lr_model.predict_proba(sample_df)[0]) * 100
    dt_conf = max(dt_model.predict_proba(sample_df)[0]) * 100
    rf_probs = rf_model.predict_proba(sample_df)[0]
    rf_conf = max(rf_probs) * 100
    knn_conf = max(knn_model.predict_proba(sample_df)[0]) * 100

    #  BEST MODEL
    scores = [lr_acc, dt_acc, rf_acc, knn_acc]
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"]

    max_score = max(scores)
    best_models = [models[i] for i, s in enumerate(scores) if s == max_score]
    best_model = ", ".join(best_models)

    # TOP 3 
    top_indices = rf_probs.argsort()[-3:][::-1]

    top3 = []
    for i, idx in enumerate(top_indices):
        crop = le.inverse_transform([idx])[0]
        confidence = round(rf_probs[idx] * 100, 2)

        top3.append({
            "rank": i + 1,
            "crop": crop,
            "confidence": confidence,
            "bar_width": confidence
        })

    # FINAL 
    final_prediction = top3[0]["crop"]
    final_conf = top3[0]["confidence"]

    # GRAPH 
    plt.figure(figsize=(6,4))
    plt.bar(["LR", "DT", "RF", "KNN"], scores)
    plt.title("Model Accuracy Comparison")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    graph = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()

    #  RETURN 
    return render(request, "proj2result.html", {
        "result": final_prediction,
        "confidence": final_conf,
        "top3": top3,

        "lr": lr_pred,
        "dt": dt_pred,
        "rf": rf_pred,
        "knn": knn_pred,

        "lr_acc": round(lr_acc, 2),
        "dt_acc": round(dt_acc, 2),
        "rf_acc": round(rf_acc, 2),
        "knn_acc": round(knn_acc, 2),

        "lr_conf": round(lr_conf, 2),
        "dt_conf": round(dt_conf, 2),
        "rf_conf": round(rf_conf, 2),
        "knn_conf": round(knn_conf, 2),

        "best": best_model,
        "inputs": inputs,
        "graph": graph
    })


#  PROJECT 3 : Resume Analizer AI
#  JOB PROFILES 

JOB_PROFILES = {
    "data_scientist": {
        "label": "Data Scientist", "icon": "🔬",
        "skills": {
            "Python": ["python"],
            "Machine Learning": ["machine learning", "ml", "sklearn"],
            "Data Analysis": ["data analysis", "analytics"],
            "SQL": ["sql", "mysql", "postgresql"],
            "Deep Learning": ["deep learning", "cnn", "rnn"],
            "Statistics": ["statistics", "probability"],
            "Pandas / NumPy": ["pandas", "numpy"],
            "Visualization": ["matplotlib", "seaborn"],
            "TensorFlow": ["tensorflow", "keras", "pytorch"],
            "Communication": ["communication", "presentation"]
        }
    },

    "web_developer": {
        "label": "Web Developer", "icon": "🌐",
        "skills": {
            "Python": ["python"],
            "Django": ["django", "flask"],
            "JavaScript": ["javascript", "js", "node"],
            "HTML & CSS": ["html", "css"],
            "React": ["react"],
            "REST API": ["api", "rest"],
            "SQL": ["sql"],
            "Git": ["git"],
            "Docker": ["docker"],
            "Communication": ["communication"]
        }
    },

    "ml_engineer": {
        "label": "ML Engineer", "icon": "🤖",
        "skills": {
            "Python": ["python"],
            "Machine Learning": ["ml", "machine learning"],
            "Deep Learning": ["deep learning"],
            "TensorFlow": ["tensorflow", "pytorch"],
            "Docker": ["docker"],
            "SQL": ["sql"],
            "Git": ["git"],
            "Cloud": ["aws", "gcp"],
            "Communication": ["communication"]
        }
    },

    "backend_developer": {
        "label": "Backend Developer", "icon": "⚙️",
        "skills": {
            "Python": ["python"],
            "Django": ["django", "flask"],
            "SQL": ["sql"],
            "REST API": ["api", "rest"],
            "Docker": ["docker"],
            "Git": ["git"],
            "Redis": ["redis"],
            "Testing": ["pytest", "testing"],
            "Communication": ["communication"]
        }
    }
}


# BONUS

BONUS_KEYWORDS = {
    "Internship": ["intern"],
    "Project": ["project", "developed"],
    "Leadership": ["leader", "managed", "led"],
    "Research": ["research"]
}


# EXPERIENCE 

def extract_experience(text):
    match = re.search(r'(\d+)\+?\s*(years|yrs)', text)
    return int(match.group(1)) if match else 0


# UTIL FUNCTIONS 

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text


def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9% ]', ' ', text.lower())


def check_skill(aliases, text):
    return any(alias in text for alias in aliases)


def check_bonus(text):
    return [k for k, v in BONUS_KEYWORDS.items() if any(a in text for a in v)]


def calculate_score(matched, total, bonus):
    base = (matched / total) * 100
    bonus_score = min(len(bonus) * 2, 10)
    return round(min(base + bonus_score, 100), 1)


def get_grade(score):
    if score >= 80: return "A", "Excellent"
    elif score >= 60: return "B", "Good"
    elif score >= 40: return "C", "Average"
    else: return "D", "Needs Improvement"


# SUGGESTIONS 

def get_suggestion(score, missing, role, text):
    suggestions = []

    if score >= 80:
        suggestions.append(f"Excellent fit for {role}")
    elif score >= 60:
        suggestions.append(f"Good fit for {role}")
    else:
        suggestions.append(f"Needs improvement for {role}")

    if "%" not in text:
        suggestions.append("Add measurable achievements (%, results)")

    leadership_words = [
        "lead", "led", "leader", "managed", "team",
        "mentor", "mentored", "collaborated",
        "scrum", "meeting", "review"
    ]

    if not any(word in text for word in leadership_words):
        suggestions.append("Highlight leadership or teamwork")

    if missing:
        suggestions.append("Improve: " + ", ".join(list(missing.keys())[:3]))

    return suggestions


#  AUTO DETECT PROFILE

def find_best_profile(text):
    experience = extract_experience(text)

    best_key = None
    best_score = -1
    best_data = None

    for key, profile in JOB_PROFILES.items():
        skills = profile["skills"]

        matched = {}
        missing = {}

        for skill, aliases in skills.items():

            if check_skill(aliases, text):
                matched[skill] = True

            else:
                # FIXED COMMUNICATION LOGIC
                if skill == "Communication":
                    if not any(word in text for word in [
                        "communication", "lead", "led", "leader",
                        "managed", "team", "mentor", "mentored",
                        "collaborated", "scrum", "meeting", "review"
                    ]):
                        missing[skill] = True
                else:
                    missing[skill] = True

        bonus = check_bonus(text)

        total = len(skills)
        matched_count = len(matched)

        score = calculate_score(matched_count, total, bonus)

        # EXPERIENCE BOOST
        if experience >= 4:
            if key in ["backend_developer", "ml_engineer"]:
                score += 5
            if key == "web_developer":
                score -= 2

        #  BACKEND BOOST
        if "aws" in text or "microservices" in text:
            if key == "backend_developer":
                score += 5

        #  REDUCE WEB BIAS
        if key == "web_developer":
            if "django" in text and "react" in text:
                score -= 3

        #  REALISTIC SCORE CAP
        score = min(score, 95)

        if score > best_score:
            best_score = score
            best_key = key
            best_data = {
                "profile": profile,
                "matched": matched,
                "missing": missing,
                "bonus": bonus,
                "score": score,
                "total": total,
                "matched_count": matched_count
            }

    return best_key, best_data

def fetch_real_jobs(role):
    url = "https://remotive.com/api/remote-jobs"

    try:
        res = requests.get(url)
        data = res.json()

        jobs = []
        keywords = role.lower().split()

        for job in data["jobs"]:
            title = job["title"].lower()

            if any(k in title for k in keywords):
                jobs.append({
                    "title": job["title"],
                    "company": job["company_name"],
                    "location": job["candidate_required_location"],
                    "link": job["url"]
                })

            if len(jobs) >= 5:
                break

        return jobs

    except:
        return []


# MAIN VIEW 

@login_required(login_url="login")
def project3(request):

    if request.method != "POST":
        return render(request, "project3.html")

    file = request.FILES.get("resume")

    if not file:
        return render(request, "project3.html", {"error": "Upload PDF"})

    if not file.name.endswith(".pdf"):
        return render(request, "project3.html", {"error": "PDF only"})

    text = extract_text_from_pdf(file)

    if not text.strip():
        return render(request, "project3.html", {"error": "Empty PDF"})

    text = clean_text(text)

    key, data = find_best_profile(text)

    profile = data["profile"]
    job_label = profile["label"]

    #  fallback for weak resumes
    if data["score"] < 20:
       job_label = "No suitable role found"

    #  FULL STACK DETECTION 
    if "react" in text and "django" in text:
        job_label = "Full Stack Developer"

    matched = data["matched"]
    missing = data["missing"]
    bonus = data["bonus"]

    score = data["score"]
    total = data["total"]
    matched_count = data["matched_count"]
    missing_count = total - matched_count

    match_percent = round((matched_count / total) * 100)

    grade, grade_label = get_grade(score)

    suggestion = get_suggestion(score, missing, job_label, text)

    real_jobs = fetch_real_jobs(job_label)

    return render(request, "proj3result.html", {
        "job_label": job_label,
        "score": score,
        "grade": grade,
        "grade_label": grade_label,
        "matched": matched,
        "missing": missing,
        "suggestion": suggestion,
        "total_skills": total,
        "matched_count": matched_count,
        "missing_count": missing_count,
        "match_percent": match_percent,
        "found_bonus": bonus,
        "real_jobs": real_jobs
    })