"""
recommender.py
----------------
Recommendation Engine + Adaptation Module + Output Module.
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "exercises.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

GOAL_CATEGORY_RATIO = {
    "fat_loss": {"cardio": 0.6, "strength": 0.4},
    "muscle_gain": {"cardio": 0.1, "strength": 0.9},
}

INTENSITY_TABLE = {
    "Too Easy":  {"sets": 4, "reps": "15-20", "rest": 45},
    "Perfect":   {"sets": 3, "reps": "12-15", "rest": 60},
    "Too Hard":  {"sets": 2, "reps": "8-10",  "rest": 90},
}


def load_exercise_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_model_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "difficulty_model.pkl"))
    goal_enc = joblib.load(os.path.join(MODEL_DIR, "goal_encoder.pkl"))
    diff_enc = joblib.load(os.path.join(MODEL_DIR, "difficulty_encoder.pkl"))
    label_enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, goal_enc, diff_enc, label_enc


def predict_difficulty(model, goal_enc, diff_enc, label_enc,
                        age, goal, duration, exercise_difficulty, fitness_level):
    row = pd.DataFrame([{
        "age": age,
        "goal_enc": goal_enc.transform([goal])[0],
        "duration": duration,
        "exercise_difficulty_enc": diff_enc.transform([exercise_difficulty])[0],
        "fitness_level": fitness_level,
    }])
    pred = model.predict(row)[0]
    return label_enc.inverse_transform([pred])[0]


def generate_workout_plan(age, goal, duration, body_part, fitness_level,
                           model, goal_enc, diff_enc, label_enc,
                           adaptive_bias=0, num_exercises=4, seed=None):
    import random
    if seed is not None:
        random.seed(seed)

    df = load_exercise_data()
    pool = df[df["muscle_group"] == body_part].copy()
    if pool.empty:
        pool = df.copy()

    ratio = GOAL_CATEGORY_RATIO[goal]
    n_cardio = max(0, round(num_exercises * ratio["cardio"]))
    n_strength = num_exercises - n_cardio

    cardio_pool = pool[pool["category"] == "cardio"]
    strength_pool = pool[pool["category"] == "strength"]

    picks = []
    picks.extend(cardio_pool.sample(min(n_cardio, len(cardio_pool)), replace=False).to_dict("records")
                 if len(cardio_pool) else [])
    picks.extend(strength_pool.sample(min(n_strength, len(strength_pool)), replace=False).to_dict("records")
                 if len(strength_pool) else [])

    remaining = num_exercises - len(picks)
    if remaining > 0 and len(pool) > len(picks):
        leftover = pool[~pool["exercise_name"].isin([p["exercise_name"] for p in picks])]
        picks.extend(leftover.sample(min(remaining, len(leftover)), replace=False).to_dict("records"))

    plan = []
    for ex in picks:
        predicted = predict_difficulty(
            model, goal_enc, diff_enc, label_enc,
            age, goal, duration, ex["difficulty_level"], fitness_level,
        )
        levels = ["Too Easy", "Perfect", "Too Hard"]
        idx = levels.index(predicted)
        idx = max(0, min(len(levels) - 1, idx - adaptive_bias))
        effective_label = levels[idx]
        intensity = INTENSITY_TABLE[effective_label]

        plan.append({
            "exercise_name": ex["exercise_name"],
            "muscle_group": ex["muscle_group"],
            "equipment": ex["equipment"],
            "base_difficulty": ex["difficulty_level"],
            "predicted_label": predicted,
            "effective_label": effective_label,
            "sets": intensity["sets"],
            "reps": intensity["reps"],
            "rest_sec": intensity["rest"],
        })

    return plan


def adapt_bias_from_feedback(feedback: str) -> int:
    return {"Too Easy": 1, "Perfect": 0, "Too Hard": -1}.get(feedback, 0)
