"""
train_model.py
---------------
Difficulty Prediction Module: simulates labeled user-feedback data and
trains a Random Forest Classifier to predict "Too Easy" / "Perfect" / "Too Hard".
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DIFFICULTY_MAP = {"beginner": 0, "intermediate": 1, "advanced": 2}


def simulate_training_data(n_samples: int = 4000) -> pd.DataFrame:
    ages = np.random.randint(16, 60, n_samples)
    goals = np.random.choice(["fat_loss", "muscle_gain"], n_samples)
    durations = np.random.randint(15, 75, n_samples)
    exercise_difficulty = np.random.choice(["beginner", "intermediate", "advanced"], n_samples)
    fitness_level = np.random.randint(1, 6, n_samples)

    rows = []
    for i in range(n_samples):
        age = ages[i]
        goal = goals[i]
        duration = durations[i]
        ex_diff = DIFFICULTY_MAP[exercise_difficulty[i]]
        level = fitness_level[i]

        load_score = ex_diff * 2 - level + (duration / 30) - (1 if age > 45 else 0)
        load_score += np.random.normal(0, 0.6)

        if load_score > 2.0:
            label = "Too Hard"
        elif load_score < -1.0:
            label = "Too Easy"
        else:
            label = "Perfect"

        rows.append({
            "age": age, "goal": goal, "duration": duration,
            "exercise_difficulty": exercise_difficulty[i],
            "fitness_level": level, "label": label,
        })

    return pd.DataFrame(rows)


def train_and_evaluate():
    df = simulate_training_data()

    goal_enc = LabelEncoder()
    diff_enc = LabelEncoder()
    label_enc = LabelEncoder()

    df["goal_enc"] = goal_enc.fit_transform(df["goal"])
    df["exercise_difficulty_enc"] = diff_enc.fit_transform(df["exercise_difficulty"])
    df["label_enc"] = label_enc.fit_transform(df["label"])

    features = ["age", "goal_enc", "duration", "exercise_difficulty_enc", "fitness_level"]
    X = df[features]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1_score": f1_score(y_test, y_pred, average="macro"),
    }

    print("Model Performance Metrics")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v*100:.1f}%")

    return model, goal_enc, diff_enc, label_enc, metrics


if __name__ == "__main__":
    model, goal_enc, diff_enc, label_enc, metrics = train_and_evaluate()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(model, os.path.join(out_dir, "difficulty_model.pkl"))
    joblib.dump(goal_enc, os.path.join(out_dir, "goal_encoder.pkl"))
    joblib.dump(diff_enc, os.path.join(out_dir, "difficulty_encoder.pkl"))
    joblib.dump(label_enc, os.path.join(out_dir, "label_encoder.pkl"))

    print(f"\nModel + encoders saved to {out_dir}/")
