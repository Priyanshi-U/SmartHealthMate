#
# ==================================================
# Smart HealthMate — Your AI Wellness Companion
# Theme: Mint Blue (pastel) | Voice Toggle | Risk Gauge
# Developed by PRI | Smart HealthMate
# ==================================================

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, simpledialog
import threading
import pandas as pd
import numpy as np
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import os
import math
import random
import matplotlib
matplotlib.use('Agg')  # avoid conflicts with Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------
# 0. NLTK: ensure data
# -------------------------
def ensure_nltk():
    try:
        _ = word_tokenize("test sentence")
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass

ensure_nltk()

# -------------------------
# 1. Seed dataset (small & extendable)
# -------------------------
data = {
    'disease': [
        'Common Cold', 'Flu', 'Migraine', 'Diabetes',
        'Hypertension', 'Asthma', 'Gastritis', 'Food Poisoning'
    ],
    'symptoms': [
        'cough fever sore_throat runny_nose',
        'fever body_ache fatigue cough',
        'headache nausea sensitivity_to_light',
        'fatigue frequent_urination thirst slow_healing',
        'headache dizziness blurred_vision high_bp',
        'shortness_of_breath chest_tightness wheezing',
        'stomach_pain nausea bloating heartburn',
        'vomiting diarrhea stomach_pain fever'
    ]
}
df = pd.DataFrame(data)

# -------------------------
# 2. Model training (simple & explainable)
# -------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['symptoms'])
y = df['disease']
model = MultinomialNB()
model.fit(X, y)

def retrain_model_with_df():
    global vectorizer, model, X, y
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['symptoms'])
    y = df['disease']
    model = MultinomialNB()
    model.fit(X, y)

# -------------------------
# 3. Health, lifestyle & doctor mappings
# -------------------------
health_tips = {
    'Common Cold': 'Drink warm fluids, rest, and stay hydrated.',
    'Flu': 'Rest, hydrate, and consult a doctor for severe symptoms.',
    'Migraine': 'Rest in a dark room and avoid triggers; consider OTC pain relief if needed.',
    'Diabetes': 'Reduce sugar intake, exercise, and consult an endocrinologist.',
    'Hypertension': 'Reduce salt, monitor BP and consult a cardiologist.',
    'Asthma': 'Avoid triggers and use your inhaler as prescribed; consult pulmonologist.',
    'Gastritis': 'Avoid spicy/acidic foods and eat smaller meals.',
    'Food Poisoning': 'Hydrate and seek medical help if vomiting/fever persists.'
}

lifestyle_templates = {
    'default': [
        "Stay hydrated and get enough rest.",
        "Eat small, balanced meals; avoid junk food.",
        "Maintain a consistent sleep schedule (7-8 hours)."
    ],
    'Diabetes': [
        "Prefer low glycemic index foods and avoid sugary drinks.",
        "Monitor blood sugar and follow your doctor's dietary plan."
    ],
    'Hypertension': [
        "Limit sodium, avoid processed foods, and practice relaxation techniques."
    ],
    'Asthma': [
        "Avoid smoke and known allergens; follow inhaler plan if prescribed."
    ]
}

doctor_suggestions = {
    'Common Cold': 'General Physician',
    'Flu': 'General Physician',
    'Migraine': 'Neurologist',
    'Diabetes': 'Endocrinologist',
    'Hypertension': 'Cardiologist',
    'Asthma': 'Pulmonologist',
    'Gastritis': 'Gastroenterologist',
    'Food Poisoning': 'General Physician'
}

# -------------------------
# 4. NLP helpers
# -------------------------
def clean_text(text):
    if not text:
        return ""
    try:
        words = word_tokenize(text.lower())
    except LookupError:
        ensure_nltk()
        words = word_tokenize(text.lower())
    sw = set(stopwords.words('english'))
    words = [w for w in words if w.isalpha() and w not in sw]
    return ' '.join(words)

def predict_with_confidence(symptom_text):
    cleaned = clean_text(symptom_text)
    if cleaned.strip() == "":
        return "Unknown", 0.0
    vec = vectorizer.transform([cleaned])
    try:
        probs = model.predict_proba(vec)[0]
        idx = np.argmax(probs)
        disease = model.classes_[idx]
        confidence = float(probs[idx])
    except Exception:
        disease = model.predict(vec)[0]
        confidence = 0.6
    return disease, confidence

# -------------------------
# 5. Voice engine & recognizer
# -------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voice_enabled = True  # default; GUI toggle will change

def speak(text):
    if not voice_enabled:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

recognizer = sr.Recognizer()

def listen_once(timeout=5, phrase_time_limit=6):
    """
    Try to listen once from microphone. Returns recognized text or empty string.
    Note: uses Google web recognizer (needs internet). If it fails, return ''.
    """
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.RequestError:
            return ""
        except sr.UnknownValueError:
            return ""
    except Exception as e:
        print("Microphone error:", e)
        return ""

# -------------------------
# 6. Risk assessment logic
# -------------------------
def compute_risk_score(answers):
    score = 0
    if answers.get('smoking','no').lower().startswith('y'):
        score += 30
    try:
        h = float(answers.get('sleep_hours', 7))
        if h < 5:
            score += 25
        elif h < 6:
            score += 15
        elif h < 7:
            score += 8
    except:
        pass
    try:
        e = int(answers.get('exercise_per_week', 0))
        if e >= 4:
            score -= 10
        elif e >= 1:
            score -= 5
    except:
        pass
    if answers.get('chronic','no').lower().startswith('y'):
        score += 25
    score = max(0, min(100, score))
    return score

def interpret_risk(score):
    if score >= 70:
        return "High risk — seek medical advice soon."
    elif score >= 40:
        return "Moderate risk — monitor and improve lifestyle."
    else:
        return "Low risk — maintain healthy lifestyle."

# -------------------------
# 7. Utility: motivation & emotion tone
# -------------------------
motivational_quotes = [
    "Small steps every day add up to big changes.",
    "Your health is an investment, not an expense.",
    "Rest, breathe, and be kind to your body.",
    "Recovery often starts with a single step—take it today."
]

def pick_quote():
    return random.choice(motivational_quotes)

def tone_for_risk(score):
    if score >= 70:
        return "serious"
    elif score >= 40:
        return "concerned"
    else:
        return "friendly"

# -------------------------
# 8. History saving
# -------------------------
HISTORY_FILE = "healthmate_history.csv"
def save_history(record):
    keys = ['timestamp', 'symptoms', 'disease', 'confidence', 'doctor', 'risk_score']
    df_rec = pd.DataFrame([record], columns=keys)
    if os.path.exists(HISTORY_FILE):
        df_rec.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df_rec.to_csv(HISTORY_FILE, index=False)

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    else:
        return pd.DataFrame(columns=['timestamp','symptoms','disease','confidence','doctor','risk_score'])

# -------------------------
# 9. GUI (Mint Blue pastel aesthetic)
# -------------------------
root = tk.Tk()
root.title("Smart HealthMate — Your AI Wellness Companion")
root.geometry("900x700")
root.configure(bg="#eaf9f6")  # soft mint background

# style
title_font = ("Helvetica", 22, "bold")
subtitle_font = ("Arial", 11)
body_font = ("Arial", 11)
accent = "#0b6b63"  # deep mint
card_bg = "#ffffff"
muted = "#6b6b6b"

# --- Top animated welcome card ---
welcome_frame = tk.Frame(root, bg=card_bg, bd=0, relief='flat')
welcome_frame.place(relx=0.03, rely=0.02, relwidth=0.94, relheight=0.14)

# WELCOME LABEL: PRII replaced with PRI
welcome_label = tk.Label(welcome_frame, text="👋 Welcome, PRI!",
                         font=title_font, bg=card_bg, fg=accent)

welcome_label.pack(anchor='w', padx=12, pady=(10,0))

welcome_sub = tk.Label(welcome_frame, text="Smart HealthMate — Your AI Wellness Companion", font=subtitle_font, bg=card_bg, fg=muted)
welcome_sub.pack(anchor='w', padx=12, pady=(0,8))

# Tk-safe animation using after()
def animate_welcome(step=13, max_step=22):
    if step <= max_step:
        welcome_label.config(font=("Helvetica", step, "bold"))
        root.after(40, animate_welcome, step+1, max_step)

# start animation safely (no thread)
animate_welcome()

# --- Left: chat & log card ---
chat_frame = tk.Frame(root, bg=card_bg)
chat_frame.place(relx=0.03, rely=0.18, relwidth=0.58, relheight=0.68)

chat_title = tk.Label(chat_frame, text="Conversation & Assistant",
                      font=("Helvetica", 14, "bold"), bg=card_bg, fg=accent)
chat_title.pack(anchor='w', padx=10, pady=(8,4))

log = scrolledtext.ScrolledText(chat_frame, width=60, height=20, wrap=tk.WORD,
                                font=body_font, bg="#fbfffd", relief="flat")
log.pack(padx=10, pady=(0,8))
log.insert(tk.END, "AI: Press 'Start Chat' to begin voice-guided conversation, or type in the manual box below.\n\n")

# input row
input_frame = tk.Frame(chat_frame, bg=card_bg)
input_frame.pack(fill='x', padx=10, pady=(0,10))

entry_symptoms = tk.Entry(input_frame, font=body_font, width=50)
entry_symptoms.grid(row=0, column=0, padx=(0,8), pady=6)

def clear_input():
    entry_symptoms.delete(0, tk.END)

# --- Right: insights card (prediction + risk + buttons) ---
insight_frame = tk.Frame(root, bg=card_bg)
insight_frame.place(relx=0.64, rely=0.18, relwidth=0.33, relheight=0.68)

insight_title = tk.Label(insight_frame, text="Insights & Suggestions",
                         font=("Helvetica", 14, "bold"), bg=card_bg, fg=accent)
insight_title.pack(anchor='w', padx=10, pady=(8,4))

# prediction display card
pred_var = tk.StringVar(value="Prediction: —")
pred_label = tk.Label(insight_frame, textvariable=pred_var, font=("Arial", 12,
                                                                   "bold"), bg=card_bg, fg="#203737")
pred_label.pack(anchor='w', padx=12, pady=(6,2))

conf_var = tk.StringVar(value="Confidence: —")
conf_label = tk.Label(insight_frame, textvariable=conf_var, font=("Arial", 11),
                      bg=card_bg, fg=muted)
conf_label.pack(anchor='w', padx=12, pady=(0,8))

doc_var = tk.StringVar(value="Suggested Doctor: —")
doc_label = tk.Label(insight_frame, textvariable=doc_var, font=("Arial", 11),
                     bg=card_bg, fg=muted)
doc_label.pack(anchor='w', padx=12, pady=(0,8))

# knowledge card (scrollable)
kc_frame = tk.Frame(insight_frame, bg="#f6fffd", bd=1,
                    relief="solid")
kc_frame.pack(fill='both', padx=12, pady=(6,6), expand=False)

kc_text = tk.Text(kc_frame, height=8, width=35, wrap=tk.WORD, bg="#f6fffd",
                  bd=0, font=("Arial",10))
kc_text.pack(padx=6, pady=6)
kc_text.insert(tk.END, "Health knowledge will appear here after prediction.\n")
kc_text.config(state='disabled')

# risk gauge canvas (matplotlib small gauge)
gauge_canvas_frame = tk.Frame(insight_frame, bg=card_bg)
gauge_canvas_frame.pack(padx=12, pady=(6,6), fill='x')

fig = Figure(figsize=(3,1.2), dpi=90)
ax = fig.add_subplot(111, polar=True)
ax.axis('off')  # hide background by default

canvas_gauge = FigureCanvasTkAgg(fig, master=gauge_canvas_frame)
canvas_gauge.get_tk_widget().pack()

def draw_gauge(score):
    ax.clear()
    ax.set_ylim(0,10)
    sectors = [
        (-math.pi, -math.pi*0.6, 'green'),
        (-math.pi*0.6, -math.pi*0.3, 'yellow'),
        (-math.pi*0.3, 0, 'red')
    ]
    for start, end, color in sectors:
        t = np.linspace(start, end, 50)
        ax.bar(t, 10, width=(end-start)/50,
               bottom=0.0, color=color, edgecolor='white', linewidth=0.2)
    angle = -math.pi + (score/100) * math.pi
    ax.arrow(angle, 0, 0, 6.5, width=0.03,
             head_width=0.18, head_length=0.6, length_includes_head=True, color='#222')
    ax.add_patch(matplotlib.patches.Circle((0,0), 0.2,
                                          transform=ax.transData._b, color='#222'))
    ax.set_axis_off()
    canvas_gauge.draw()

draw_gauge(0)

# --- Buttons and toggles below the insight card ---
controls_frame = tk.Frame(insight_frame, bg=card_bg)
controls_frame.pack(fill='x', padx=12, pady=(6,6))

voice_var = tk.BooleanVar(value=True)  # toggle variable
def toggle_voice():
    global voice_enabled
    voice_enabled = voice_var.get()

voice_toggle = ttk.Checkbutton(controls_frame, text="Voice (ON/OFF)",
                               variable=voice_var, command=toggle_voice)
voice_toggle.grid(row=0, column=0, sticky='w', padx=(0,6))

privacy_var = tk.BooleanVar(value=True)
def toggle_privacy():
    if privacy_var.get():
        privacy_label.config(text="Privacy: Local Only")
    else:
        privacy_label.config(text="Privacy: Local (no external APIs guaranteed)")

privacy_cb = ttk.Checkbutton(controls_frame, text="Privacy Mode",
                             variable=privacy_var, command=toggle_privacy)
privacy_cb.grid(row=0, column=1, sticky='w', padx=(6,6))

save_history_btn = tk.Button(controls_frame, text="View History",
                             bg="#0b6b63", fg="white", command=lambda:
                                 show_history_window())
save_history_btn.grid(row=0, column=2, padx=(6,6))

privacy_label = tk.Label(controls_frame, text="Privacy: Local Only", bg=card_bg,
                         fg=muted)
privacy_label.grid(row=1, column=0, columnspan=3, sticky='w', pady=(6,0))

# --- Bottom control buttons (Start Chat, Predict, Clear) ---
bottom_frame = tk.Frame(root, bg=card_bg)
bottom_frame.place(relx=0.03, rely=0.86, relwidth=0.94, relheight=0.12)

def append_log(text, who='AI'):
    log.insert(tk.END, f"{who}: {text}\n")
    log.see(tk.END)

# CHAT FLOW (threaded)
def chat_flow_voice(log_widget):
    try:
        # MDS in speech
        speak("Hello! I'm Smart HealthMate. I'll ask a few short questions to help.")
        append_log("Hello! I'm Smart HealthMate. I'll ask a few short questions to help.", 'AI')

        time.sleep(0.4)

        speak("Please tell me your main symptoms. Speak now.")
        append_log("Please tell me your main symptoms. Speak now.", 'AI')
        user_sym = listen_once()
        if not user_sym:
            speak("I couldn't hear you. Please type symptoms in the box and press 'Predict' or use 'Submit (typed)'.")
            append_log("No voice input. Please type symptoms and use Submit (typed).", 'AI')
            return
        append_log(user_sym, 'You')

        disease, conf = predict_with_confidence(user_sym)
        conf_pct = round(conf * 100, 1)
        pred_var.set(f"Prediction: {disease}")
        conf_var.set(f"Confidence: {conf_pct}%")
        doc_var.set(f"Suggested Doctor: {doctor_suggestions.get(disease,'General Physician')}")
        kc_text.config(state='normal')
        kc_text.delete('1.0', tk.END)
        kc_text.insert(tk.END,
                       f"{disease}\n\nCauses: (common causes summary)\n- See general physician for accurate diagnosis.\n\nPrevention & Precautions:\n- {health_tips.get(disease)}")
        kc_text.config(state='disabled')
        append_log(f"Predicted {disease} ({conf_pct}%).", 'AI')
        speak(f"My prediction is {disease} with {conf_pct} percent confidence. {risk_text}")

        speak("Do you have fever? Say yes or no.")
        append_log("Do you have fever? (yes/no)", 'AI')
        fever = listen_once()
        if fever:
            append_log(fever, 'You')
        else:
            fever = "no"
            append_log("no response (assumed no)", 'System')

        speak("How many hours do you sleep on average per night? Say a number like 7.")
        append_log("How many hours do you sleep on average per night?", 'AI')
        sleep = listen_once()
        if sleep:
            append_log(sleep, 'You')
        else:
            sleep = "7"

        speak("Do you smoke? Say yes or no.")
        append_log("Do you smoke? (yes/no)", 'AI')
        smoke = listen_once()
        if smoke:
            append_log(smoke, 'You')
        else:
            smoke = "no"

        speak("How many days per week do you exercise? Say 0 to 7.")
        append_log("How many days per week do you exercise?", 'AI')
        ex = listen_once()
        if ex:
            append_log(ex, 'You')
        else:
            ex = "0"

        speak("Do you have any chronic conditions like diabetes or hypertension? Say yes or no.")
        append_log("Do you have any chronic conditions? (yes/no)", 'AI')
        chronic = listen_once()
        if chronic:
            append_log(chronic, 'You')
        else:
            chronic = "no"

        answers = {
            'fever': fever,
            'sleep_hours': sleep,
            'smoking': smoke,
            'exercise_per_week': ex,
            'chronic': chronic
        }
        risk_score = compute_risk_score(answers)
        risk_text = interpret_risk(risk_score)
        draw_gauge(risk_score)
        append_log("=== Summary ===", 'System')
        lifestyle_notes = []
        try:
            if float(sleep) < 6:
                lifestyle_notes.append("Increase sleep to at least 7 hours.")
        except:
            pass
        if smoke.lower().startswith('y'):
            lifestyle_notes.append("Consider reducing/quitting smoking.")
        try:
            if int(ex) < 2:
                lifestyle_notes.append("Try to exercise at least 3 times a week.")
        except:
            pass
        if chronic.lower().startswith('y'):
            lifestyle_notes.append("Follow up regularly for chronic conditions.")

        lifestyle_str = " ".join(lifestyle_notes) if lifestyle_notes else "Maintain healthy habits: balanced diet, hydration, rest."

        basic_tip = health_tips.get(disease, "Stay hydrated and seek medical advice if symptoms worsen.")
        doctor = doctor_suggestions.get(disease, "General Physician")
        summary = (f"Prediction: {disease} ({conf_pct}%)\nDoctor: {doctor}\nTip: {basic_tip}\nLifestyle: {lifestyle_str}\nRisk: {int(risk_score)}/100 - {risk_text}\n")
        append_log(summary, 'AI')
        speak(f"I predict {disease} with {conf_pct} percent confidence. {risk_text}")
        rec = {
            'timestamp': pd.Timestamp.now(),
            'symptoms': user_sym,
            'disease': disease,
            'confidence': conf_pct,
            'doctor': doctor,
            'risk_score': int(risk_score)
        }
        save_history([rec[k] for k in
                      ['timestamp','symptoms','disease','confidence','doctor','risk_score']])
    except Exception as e:
        append_log(f"Chat error: {e}", 'System')
        speak("Sorry, something went wrong during the chat.")

# Manual typed prediction flow (quick)
def submit_manual():
    user_text = entry_symptoms.get().strip()
    if not user_text:
        messagebox.showwarning("Input required", "Please type your symptoms or use Start Chat.")
        return
    append_log(user_text, 'You')
    disease, conf = predict_with_confidence(user_text)
    conf_pct = round(conf * 100, 1)
    pred_var.set(f"Prediction: {disease}")
    conf_var.set(f"Confidence: {conf_pct}%")
    doc = doctor_suggestions.get(disease, 'General Physician')
    doc_var.set(f"Suggested Doctor: {doc}")
    kc_text.config(state='normal')
    kc_text.delete('1.0', tk.END)
    kc_text.insert(tk.END,
                   f"{disease}\n\nPrevention & Precautions:\n- {health_tips.get(disease,'Follow general precautions.')}")
    kc_text.config(state='disabled')
    fever = messagebox.askyesno("Follow-up", "Do you have fever?")
    sleep = simpledialog.askstring("Sleep", "How many hours do you sleep on average per night?", parent=root)
    smoke = messagebox.askyesno("Smoking", "Do you smoke?")
    ex = simpledialog.askstring("Exercise", "How many days a week do you exercise? (0-7)", parent=root)
    chronic = messagebox.askyesno("Chronic", "Do you have any chronic condition like diabetes or hypertension?")
    answers = {
        'fever': "yes" if fever else "no",
        'sleep_hours': sleep if sleep else "7",
        'smoking': "yes" if smoke else "no",
        'exercise_per_week': ex if ex else "0",
        'chronic': "yes" if chronic else "no"
    }
    risk_score = compute_risk_score(answers)
    risk_text = interpret_risk(risk_score)
    draw_gauge(risk_score)
    lifestyle_notes = []
    try:
        if float(answers['sleep_hours']) < 6:
            lifestyle_notes.append("Increase sleep to at least 7 hours.")
    except:
        pass
    if answers['smoking'].startswith('y'):
        lifestyle_notes.append("Consider reducing/quitting smoking.")
    try:
        if int(answers['exercise_per_week']) < 2:
            lifestyle_notes.append("Try to exercise 3 times a week.")
    except:
        pass
    if answers['chronic'].startswith('y'):
        lifestyle_notes.append("Monitor chronic conditions closely.")
    lifestyle_str = " ".join(lifestyle_notes) if lifestyle_notes else "Maintain healthy habits."
    basic_tip = health_tips.get(disease, "")
    summary = (f"Prediction: {disease} ({conf_pct}%)\nDoctor: {doc}\nTip: {basic_tip}\nLifestyle: {lifestyle_str}\nRisk: {int(risk_score)}/100 - {risk_text}\n")
    append_log(summary, 'AI')
    speak(f"I predict {disease} with {conf_pct} percent confidence. {risk_text}")
    rec = {
        'timestamp': pd.Timestamp.now(),
        'symptoms': user_text,
        'disease': disease,
        'confidence': conf_pct,
        'doctor': doc,
        'risk_score': int(risk_score)
    }
    save_history([rec[k] for k in
                  ['timestamp','symptoms','disease','confidence','doctor','risk_score']])
    clear_input()

# Buttons
btn_frame_main = tk.Frame(chat_frame, bg=card_bg)
btn_frame_main.pack(pady=(0,8))

start_btn = tk.Button(btn_frame_main, text="🔊 Start Chat (Voice)",
                      bg=accent, fg="white", command=lambda:
                          threading.Thread(target=chat_flow_voice, args=(log,), daemon=True).start())
start_btn.grid(row=0, column=0, padx=6)

submit_btn = tk.Button(btn_frame_main, text="Submit (typed)",
                       bg="#2c8f82", fg="white", command=submit_manual)
submit_btn.grid(row=0, column=1, padx=6)

clear_btn = tk.Button(btn_frame_main, text="🧹 Clear Input",
                      bg="#e77b7b", fg="white", command=clear_input)
clear_btn.grid(row=0, column=2, padx=6)

# History window function
def show_history_window():
    hist = load_history()
    win = tk.Toplevel(root)
    win.title("History - Smart HealthMate")
    win.geometry("700x400")
    t = scrolledtext.ScrolledText(win, width=90, height=22)
    t.pack(padx=8, pady=8)
    if hist.empty:
        t.insert(tk.END, "No history yet.\n")
    else:
        for i, r in hist.iterrows():
            t.insert(tk.END,
                     f"{r['timestamp']} | Symptoms: {r['symptoms']} | Disease: {r['disease']} | Confidence: {r['confidence']}% | Doctor: {r['doctor']} | Risk: {r['risk_score']}\n")
    tk.Button(win, text="Export CSV",
              command=lambda: export_history_csv(win)).pack(pady=8)

def export_history_csv(parent):
    hist = load_history()
    if hist.empty:
        messagebox.showinfo("Export", "No history to export.")
        return
    save_path = "healthmate_history_export.csv"
    hist.to_csv(save_path, index=False)
    messagebox.showinfo("Exported", f"History exported to {save_path}")

# footer & motivator
quote_var = tk.StringVar(value=pick_quote())
quote_label = tk.Label(root, textvariable=quote_var, font=("Arial", 10,
                                                           "italic"), bg="#eaf9f6", fg=muted)
quote_label.place(relx=0.03, rely=0.98, anchor='w')

# FOOTER: PRII replaced with PRI
footer = tk.Label(root, text="Developed by PRI | Smart HealthMate",
                 font=("Arial", 9), bg="#eaf9f6", fg=muted)

footer.place(relx=0.98, rely=0.98, anchor='e')

# FINAL RUN: PRII replaced with PRI in welcome speech
threading.Thread(target=lambda:
                 speak("Welcome PRI, to Smart HealthMate. Toggle voice if you prefer silence, then press Start Chat or use the typed input."),
                 daemon=True).start()

root.mainloop()                    