"""
Smart Gym Workout Planner (ML + Adaptive)
Run with: streamlit run app.py
"""

import streamlit as st
from src.recommender import (
    generate_workout_plan,
    load_model_artifacts,
    adapt_bias_from_feedback,
)

st.set_page_config(page_title="Smart Gym Workout Planner", page_icon="🏋️", layout="centered")

if "adaptive_bias" not in st.session_state:
    st.session_state.adaptive_bias = 0
if "plan" not in st.session_state:
    st.session_state.plan = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None


@st.cache_resource
def get_model_artifacts():
    return load_model_artifacts()


model, goal_enc, diff_enc, label_enc = get_model_artifacts()

st.title("🏋️ Smart Gym Workout Planner (ML + Adaptive)")

age = st.slider("Age", min_value=14, max_value=70, value=21)
goal = st.selectbox("Goal", ["muscle_gain", "fat_loss"])
body_part = st.selectbox("Body Part", ["chest", "back", "legs", "arms", "shoulders", "core", "cardio"])
duration = st.slider("Workout Time (minutes)", min_value=15, max_value=90, value=30)
fitness_level = st.slider("Self-Rated Fitness Level (1 = new, 5 = experienced)", 1, 5, 2)

generate = st.button("Generate Workout Plan", type="primary")

if generate:
    st.session_state.last_inputs = dict(
        age=age, goal=goal, body_part=body_part, duration=duration, fitness_level=fitness_level,
    )
    st.session_state.plan = generate_workout_plan(
        age=age, goal=goal, duration=duration, body_part=body_part, fitness_level=fitness_level,
        model=model, goal_enc=goal_enc, diff_enc=diff_enc, label_enc=label_enc,
        adaptive_bias=st.session_state.adaptive_bias,
    )

if st.session_state.plan:
    st.markdown("### 📋 Your Personalized Workout Plan")
    for i, ex in enumerate(st.session_state.plan, start=1):
        st.markdown(f"**{i}. {ex['exercise_name']}**")
        st.write(f"- Target: {ex['muscle_group']}")
        st.write(f"- Equipment: {ex['equipment']}")
        st.write(f"- Difficulty Level: {ex['effective_label']}")
        st.write(f"- Sets: {ex['sets']} | Reps: {ex['reps']} | Rest: {ex['rest_sec']} sec")
        st.divider()

    st.markdown("### 🔁 Workout Feedback")
    st.write("How was this workout?")
    fb_col1, fb_col2, fb_col3 = st.columns(3)
    feedback = None
    if fb_col1.button("Perfect"):
        feedback = "Perfect"
    if fb_col2.button("Too Easy"):
        feedback = "Too Easy"
    if fb_col3.button("Too Hard"):
        feedback = "Too Hard"

    if feedback:
        st.session_state.adaptive_bias = adapt_bias_from_feedback(feedback)
        st.success(f"Feedback recorded: **{feedback}**. Adjusting next plan (bias={st.session_state.adaptive_bias:+d}).")
        li = st.session_state.last_inputs
        st.session_state.plan = generate_workout_plan(
            age=li["age"], goal=li["goal"], duration=li["duration"],
            body_part=li["body_part"], fitness_level=li["fitness_level"],
            model=model, goal_enc=goal_enc, diff_enc=diff_enc, label_enc=label_enc,
            adaptive_bias=st.session_state.adaptive_bias,
        )
        st.rerun()
else:
    st.info("Set your details above and click **Generate Workout Plan** to get started.")
