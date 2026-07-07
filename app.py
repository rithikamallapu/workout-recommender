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

st.set_page_config(
    page_title="Smart Gym Workout Planner",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #8888AA;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.2);
}

/* ── Exercise number badge ── */
.ex-number {
    display: inline-block;
    background: linear-gradient(135deg, #6C63FF, #3ECFCF);
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    text-align: center;
    line-height: 32px;
    margin-right: 10px;
}

/* ── Exercise title ── */
.ex-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #FFFFFF;
    display: inline;
}

/* ── Difficulty badge ── */
.badge-easy   { background: rgba(62,207,207,0.18); color:#3ECFCF; border:1px solid #3ECFCF44; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.badge-perfect{ background: rgba(108,99,255,0.18); color:#A89FFF; border:1px solid #6C63FF44; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.badge-hard   { background: rgba(255,100,130,0.18); color:#FF6482; border:1px solid #FF648244; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }

/* ── Stat pills inside cards ── */
.stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:0.8rem; }
.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #CCCCDD;
}
.stat-pill span { color:#6C63FF; font-weight:600; }

/* ── Section heading ── */
.section-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: #E0E0FF;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(108,99,255,0.3);
}

/* ── Feedback buttons ── */
div[data-testid="column"] .stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button {
    background: linear-gradient(135deg, #6C63FF, #9B93FF) !important;
    border: none !important; color: white !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: rgba(62,207,207,0.15) !important;
    border: 1px solid #3ECFCF !important; color: #3ECFCF !important;
}
div[data-testid="column"]:nth-child(3) .stButton > button {
    background: rgba(255,100,130,0.15) !important;
    border: 1px solid #FF6482 !important; color: #FF6482 !important;
}

/* ── Primary generate button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(108,99,255,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(26, 26, 46, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div { background: #6C63FF !important; }

/* ── Info / success boxes ── */
.stAlert { border-radius: 12px !important; }

/* ── Metric boxes ── */
.metric-box {
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 800; color: #A89FFF; }
.metric-lbl { font-size: 0.8rem; color: #8888AA; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
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

# ── Sidebar — Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Your Profile")
    st.markdown("---")

    age = st.slider("🎂 Age", min_value=14, max_value=70, value=21)
    goal = st.selectbox(
        "🎯 Goal",
        ["muscle_gain", "fat_loss"],
        format_func=lambda x: "💪 Muscle Gain" if x == "muscle_gain" else "🔥 Fat Loss",
    )
    body_part = st.selectbox(
        "🦾 Target Muscle",
        ["chest", "back", "legs", "arms", "shoulders", "core", "cardio"],
        format_func=lambda x: {
            "chest": "🫁 Chest", "back": "🔙 Back", "legs": "🦵 Legs",
            "arms": "💪 Arms", "shoulders": "🏔️ Shoulders",
            "core": "⭕ Core", "cardio": "❤️ Cardio",
        }[x],
    )
    duration = st.slider("⏱️ Duration (minutes)", min_value=15, max_value=90, value=30)
    fitness_level = st.slider("⭐ Fitness Level", 1, 5, 2,
                               help="1 = complete beginner · 5 = very experienced")

    st.markdown("---")
    generate = st.button("🚀 Generate Workout", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#8888AA;font-size:0.78rem;text-align:center'>"
        "Powered by Random Forest ML<br>87 exercises · 10k training samples"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-header">🏋️ Smart Gym Workout Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Personalized workouts powered by Machine Learning — adaptive to your feedback</div>', unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-box"><div class="metric-val">87</div><div class="metric-lbl">Exercises</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box"><div class="metric-val">7</div><div class="metric-lbl">Muscle Groups</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box"><div class="metric-val">85.8%</div><div class="metric-lbl">Model Accuracy</div></div>', unsafe_allow_html=True)
with c4:
    bias_display = {-1: "🔥 Harder", 0: "⚖️ Balanced", 1: "😌 Easier"}.get(st.session_state.adaptive_bias, "⚖️ Balanced")
    st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.3rem">{bias_display}</div><div class="metric-lbl">Adaptive Mode</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Generate plan ─────────────────────────────────────────────────────────────
if generate:
    st.session_state.last_inputs = dict(
        age=age, goal=goal, body_part=body_part, duration=duration, fitness_level=fitness_level,
    )
    with st.spinner("🤖 Generating your personalized plan..."):
        st.session_state.plan = generate_workout_plan(
            age=age, goal=goal, duration=duration, body_part=body_part,
            fitness_level=fitness_level, model=model, goal_enc=goal_enc,
            diff_enc=diff_enc, label_enc=label_enc,
            adaptive_bias=st.session_state.adaptive_bias,
        )

# ── Display plan ──────────────────────────────────────────────────────────────
BADGE = {
    "Too Easy":  '<span class="badge-easy">😌 Too Easy</span>',
    "Perfect":   '<span class="badge-perfect">✅ Perfect</span>',
    "Too Hard":  '<span class="badge-hard">🔥 Too Hard</span>',
}

EQUIPMENT_ICON = {
    "bodyweight": "🤸", "dumbbell": "🏋️", "barbell": "🏋️",
    "cable": "🔗", "machine": "⚙️", "kettlebells": "🫙",
    "jump_rope": "🪢",
}

MUSCLE_ICON = {
    "chest": "🫁", "back": "🔙", "legs": "🦵", "arms": "💪",
    "shoulders": "🏔️", "core": "⭕", "cardio": "❤️",
}

if st.session_state.plan:
    st.markdown('<div class="section-heading">📋 Your Personalized Workout Plan</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2])

    with left_col:
        for i, ex in enumerate(st.session_state.plan, start=1):
            eq_icon  = EQUIPMENT_ICON.get(ex["equipment"], "🏋️")
            mus_icon = MUSCLE_ICON.get(ex["muscle_group"], "💪")
            badge    = BADGE.get(ex["effective_label"], "")

            st.markdown(f"""
            <div class="glass-card">
                <div>
                    <span class="ex-number">{i}</span>
                    <span class="ex-title">{ex['exercise_name']}</span>
                    &nbsp;&nbsp;{badge}
                </div>
                <div class="stat-row">
                    <div class="stat-pill">{mus_icon} <span>{ex['muscle_group'].title()}</span></div>
                    <div class="stat-pill">{eq_icon} <span>{ex['equipment'].replace('_',' ').title()}</span></div>
                    <div class="stat-pill">🔢 Sets: <span>{ex['sets']}</span></div>
                    <div class="stat-pill">🔁 Reps: <span>{ex['reps']}</span></div>
                    <div class="stat-pill">⏸️ Rest: <span>{ex['rest_sec']}s</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-heading">🔁 Workout Feedback</div>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#8888AA;font-size:0.9rem'>How did this workout feel? "
            "Your feedback adapts the next plan.</p>",
            unsafe_allow_html=True,
        )

        fb_col1, fb_col2, fb_col3 = st.columns(3)
        feedback = None
        if fb_col1.button("✅ Perfect"):
            feedback = "Perfect"
        if fb_col2.button("😌 Too Easy"):
            feedback = "Too Easy"
        if fb_col3.button("🔥 Too Hard"):
            feedback = "Too Hard"

        if feedback:
            st.session_state.adaptive_bias = adapt_bias_from_feedback(feedback)
            st.success(f"Got it! Next plan adjusted → **{feedback}** (bias {st.session_state.adaptive_bias:+d})")
            li = st.session_state.last_inputs
            st.session_state.plan = generate_workout_plan(
                age=li["age"], goal=li["goal"], duration=li["duration"],
                body_part=li["body_part"], fitness_level=li["fitness_level"],
                model=model, goal_enc=goal_enc, diff_enc=diff_enc, label_enc=label_enc,
                adaptive_bias=st.session_state.adaptive_bias,
            )
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📊 Plan Summary</div>', unsafe_allow_html=True)

        total_sets  = sum(e["sets"] for e in st.session_state.plan)
        categories  = list({e["muscle_group"] for e in st.session_state.plan})
        equipments  = list({e["equipment"] for e in st.session_state.plan})

        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-row" style="flex-direction:column;gap:8px;">
                <div class="stat-pill">🔢 Total Sets: <span>{total_sets}</span></div>
                <div class="stat-pill">🦾 Muscles: <span>{', '.join(c.title() for c in categories)}</span></div>
                <div class="stat-pill">🏋️ Equipment: <span>{', '.join(e.replace('_',' ').title() for e in equipments)}</span></div>
                <div class="stat-pill">⏱️ Est. Duration: <span>~{li['duration']} min</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:3rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">🏋️</div>
        <div style="font-size:1.3rem;font-weight:700;color:#E0E0FF;margin-bottom:0.5rem;">
            Ready to Train?
        </div>
        <div style="color:#8888AA;font-size:0.95rem;">
            Set your profile in the sidebar and click <strong style="color:#6C63FF">Generate Workout</strong> to get your personalized plan.
        </div>
    </div>
    """, unsafe_allow_html=True)
