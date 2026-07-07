"""
generate_dataset.py
--------------------
Creates the structured exercise dataset used by the recommendation engine.
"""

import pandas as pd
import os

EXERCISES = [
    # ── CHEST (12) ──────────────────────────────────────────────────────────
    ("Push Ups",                       "chest",     "bodyweight",  "beginner",     "strength", "muscle_gain"),
    ("Incline Dumbbell Press",         "chest",     "dumbbell",    "intermediate", "strength", "muscle_gain"),
    ("Straight-Arm Dumbbell Pullover", "chest",     "dumbbell",    "intermediate", "strength", "muscle_gain"),
    ("One-Arm Kettlebell Floor Press", "chest",     "kettlebells", "intermediate", "strength", "muscle_gain"),
    ("Barbell Bench Press",            "chest",     "barbell",     "advanced",     "strength", "muscle_gain"),
    ("Chest Fly",                      "chest",     "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Cable Crossover",                "chest",     "cable",       "advanced",     "strength", "muscle_gain"),
    ("Decline Bench Press",            "chest",     "barbell",     "advanced",     "strength", "muscle_gain"),
    ("Push-Up Plus",                   "chest",     "bodyweight",  "beginner",     "strength", "muscle_gain"),
    ("Pec Deck",                       "chest",     "machine",     "beginner",     "strength", "muscle_gain"),
    ("Landmine Press",                 "chest",     "barbell",     "intermediate", "strength", "muscle_gain"),
    ("Dumbbell Pullover",              "chest",     "dumbbell",    "intermediate", "strength", "muscle_gain"),

    # ── BACK (12) ───────────────────────────────────────────────────────────
    ("Pull Ups",                       "back",      "bodyweight",  "advanced",     "strength", "muscle_gain"),
    ("Bent-Over Row",                  "back",      "barbell",     "intermediate", "strength", "muscle_gain"),
    ("Lat Pulldown",                   "back",      "cable",       "beginner",     "strength", "muscle_gain"),
    ("Single-Arm Dumbbell Row",        "back",      "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Superman Hold",                  "back",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Deadlift",                       "back",      "barbell",     "advanced",     "strength", "muscle_gain"),
    ("T-Bar Row",                      "back",      "barbell",     "advanced",     "strength", "muscle_gain"),
    ("Seated Cable Row",               "back",      "cable",       "beginner",     "strength", "muscle_gain"),
    ("Face Pull",                      "back",      "cable",       "beginner",     "strength", "muscle_gain"),
    ("Good Mornings",                  "back",      "barbell",     "intermediate", "strength", "muscle_gain"),
    ("Hyperextension",                 "back",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Chin Ups",                       "back",      "bodyweight",  "intermediate", "strength", "muscle_gain"),

    # ── LEGS (15) ───────────────────────────────────────────────────────────
    ("Bodyweight Squat",               "legs",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Lunges",                         "legs",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Barbell Squat",                  "legs",      "barbell",     "advanced",     "strength", "muscle_gain"),
    ("Leg Press",                      "legs",      "machine",     "intermediate", "strength", "muscle_gain"),
    ("Jump Squats",                    "legs",      "bodyweight",  "intermediate", "cardio",   "fat_loss"),
    ("Bulgarian Split Squat",          "legs",      "dumbbell",    "advanced",     "strength", "muscle_gain"),
    ("Calf Raises",                    "legs",      "bodyweight",  "beginner",     "strength", "muscle_gain"),
    ("Romanian Deadlift",              "legs",      "barbell",     "intermediate", "strength", "muscle_gain"),
    ("Leg Curl",                       "legs",      "machine",     "beginner",     "strength", "muscle_gain"),
    ("Leg Extension",                  "legs",      "machine",     "beginner",     "strength", "muscle_gain"),
    ("Sumo Squat",                     "legs",      "dumbbell",    "intermediate", "strength", "muscle_gain"),
    ("Step Ups",                       "legs",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Wall Sit",                       "legs",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Glute Bridge",                   "legs",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Hip Thrust",                     "legs",      "barbell",     "intermediate", "strength", "muscle_gain"),

    # ── ARMS (12) ───────────────────────────────────────────────────────────
    ("Bicep Curl",                     "arms",      "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Hammer Curl",                    "arms",      "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Tricep Dips",                    "arms",      "bodyweight",  "intermediate", "strength", "muscle_gain"),
    ("Skull Crushers",                 "arms",      "barbell",     "intermediate", "strength", "muscle_gain"),
    ("Diamond Push Ups",               "arms",      "bodyweight",  "advanced",     "strength", "muscle_gain"),
    ("Concentration Curl",             "arms",      "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Preacher Curl",                  "arms",      "barbell",     "intermediate", "strength", "muscle_gain"),
    ("Cable Curl",                     "arms",      "cable",       "beginner",     "strength", "muscle_gain"),
    ("Tricep Pushdown",                "arms",      "cable",       "beginner",     "strength", "muscle_gain"),
    ("Overhead Tricep Extension",      "arms",      "dumbbell",    "intermediate", "strength", "muscle_gain"),
    ("Close-Grip Bench Press",         "arms",      "barbell",     "advanced",     "strength", "muscle_gain"),
    ("Zottman Curl",                   "arms",      "dumbbell",    "intermediate", "strength", "muscle_gain"),

    # ── SHOULDERS (11) ──────────────────────────────────────────────────────
    ("Shoulder Press",                 "shoulders", "dumbbell",    "intermediate", "strength", "muscle_gain"),
    ("Lateral Raise",                  "shoulders", "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Front Raise",                    "shoulders", "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Arnold Press",                   "shoulders", "dumbbell",    "advanced",     "strength", "muscle_gain"),
    ("Pike Push Ups",                  "shoulders", "bodyweight",  "intermediate", "strength", "muscle_gain"),
    ("Upright Row",                    "shoulders", "barbell",     "intermediate", "strength", "muscle_gain"),
    ("Reverse Fly",                    "shoulders", "dumbbell",    "beginner",     "strength", "muscle_gain"),
    ("Cable Lateral Raise",            "shoulders", "cable",       "intermediate", "strength", "muscle_gain"),
    ("Barbell Overhead Press",         "shoulders", "barbell",     "advanced",     "strength", "muscle_gain"),
    ("Bent-Over Rear Delt Fly",        "shoulders", "dumbbell",    "intermediate", "strength", "muscle_gain"),
    ("Seated Dumbbell Press",          "shoulders", "dumbbell",    "beginner",     "strength", "muscle_gain"),

    # ── CORE (13) ───────────────────────────────────────────────────────────
    ("Plank",                          "core",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Crunches",                       "core",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Russian Twist",                  "core",      "bodyweight",  "intermediate", "strength", "fat_loss"),
    ("Leg Raises",                     "core",      "bodyweight",  "intermediate", "strength", "fat_loss"),
    ("Mountain Climbers",              "core",      "bodyweight",  "intermediate", "cardio",   "fat_loss"),
    ("Bicycle Crunch",                 "core",      "bodyweight",  "beginner",     "cardio",   "fat_loss"),
    ("Side Plank",                     "core",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("Ab Wheel Rollout",               "core",      "bodyweight",  "advanced",     "strength", "fat_loss"),
    ("Dead Bug",                       "core",      "bodyweight",  "intermediate", "strength", "fat_loss"),
    ("Hanging Leg Raise",              "core",      "bodyweight",  "advanced",     "strength", "muscle_gain"),
    ("Cable Crunch",                   "core",      "cable",       "intermediate", "strength", "fat_loss"),
    ("Flutter Kicks",                  "core",      "bodyweight",  "beginner",     "strength", "fat_loss"),
    ("V-Ups",                          "core",      "bodyweight",  "intermediate", "strength", "fat_loss"),

    # ── CARDIO (12) ─────────────────────────────────────────────────────────
    ("Jumping Jacks",                  "cardio",    "bodyweight",  "beginner",     "cardio",   "fat_loss"),
    ("Burpees",                        "cardio",    "bodyweight",  "advanced",     "cardio",   "fat_loss"),
    ("High Knees",                     "cardio",    "bodyweight",  "beginner",     "cardio",   "fat_loss"),
    ("Jump Rope",                      "cardio",    "jump_rope",   "intermediate", "cardio",   "fat_loss"),
    ("Sprint Intervals",               "cardio",    "bodyweight",  "advanced",     "cardio",   "fat_loss"),
    ("Box Jumps",                      "cardio",    "bodyweight",  "intermediate", "cardio",   "fat_loss"),
    ("Stair Climbing",                 "cardio",    "bodyweight",  "beginner",     "cardio",   "fat_loss"),
    ("Rowing Machine",                 "cardio",    "machine",     "intermediate", "cardio",   "fat_loss"),
    ("Cycling",                        "cardio",    "machine",     "beginner",     "cardio",   "fat_loss"),
    ("Bear Crawl",                     "cardio",    "bodyweight",  "intermediate", "cardio",   "fat_loss"),
    ("Lateral Shuffle",                "cardio",    "bodyweight",  "beginner",     "cardio",   "fat_loss"),
    ("Battle Ropes",                   "cardio",    "cable",       "advanced",     "cardio",   "fat_loss"),
]

COLUMNS = ["exercise_name", "muscle_group", "equipment", "difficulty_level", "category", "goal_tag"]


def build_dataset() -> pd.DataFrame:
    return pd.DataFrame(EXERCISES, columns=COLUMNS)


if __name__ == "__main__":
    df = build_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "exercises.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} exercises to {out_path}")
