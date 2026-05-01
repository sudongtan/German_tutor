"""
Streamlit UI for the German Tutor.

Session flow:
  mode → topic discussion: topic → level → answer loop → done
  mode → nuance comparison: english_input → nuance_answer loop → done
Each step is driven by st.session_state.step and communicates with the
respective LangGraph app via invoke() and Command(resume=...).
"""

import uuid

import streamlit as st
from langgraph.types import Command
from pydantic import ValidationError

from src.german_tutor.audio import transcribe_audio
from src.german_tutor.nuance_comparison import nuance_app
from src.german_tutor.schemas import UserAnswer, UserTopic
from src.german_tutor.topic_discussion import app as topic_app

st.title("German Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "step" not in st.session_state:
    st.session_state.step = "mode"
if "pending_topic" not in st.session_state:
    st.session_state.pending_topic = ""
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0
if "nuance_session" not in st.session_state:
    st.session_state.nuance_session = 0

topic_config = {"configurable": {"thread_id": f"topic_{st.session_state.thread_id}"}}
nuance_config = {
    "configurable": {
        "thread_id": f"nuance_{st.session_state.thread_id}_{st.session_state.nuance_session}"
    }
}

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Mode selection ──────────────────────────────────────────────────────────
if st.session_state.step == "mode":
    st.write("What would you like to do?")
    col1, col2 = st.columns(2)
    if col1.button("Topic Discussion", use_container_width=True):
        st.session_state.step = "topic"
        st.rerun()
    if col2.button("Nuance Comparison", use_container_width=True):
        st.session_state.step = "english_input"
        st.rerun()

# ── Topic Discussion ────────────────────────────────────────────────────────
elif st.session_state.step == "topic":
    user_input = st.chat_input("What topic would you like to practice?")
    if user_input:
        try:
            validated = UserTopic(topic=user_input)
        except ValidationError as e:
            st.error(e.errors()[0]["msg"])
            st.stop()
        st.session_state.messages.append({"role": "user", "content": validated.topic})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "What level would you like to practice at?",
            }
        )
        st.session_state.pending_topic = validated.topic
        st.session_state.step = "level"
        st.rerun()

elif st.session_state.step == "level":
    cols = st.columns(4)
    selected_level = None
    for level, col in zip(["A2", "B1", "B2", "C1"], cols):
        if col.button(level, use_container_width=True):
            selected_level = level

    if selected_level:
        st.session_state.messages.append({"role": "user", "content": selected_level})
        try:
            result = topic_app.invoke(
                {
                    "user_topic": st.session_state.pending_topic,
                    "user_level": selected_level,
                },
                topic_config,
            )
        except RuntimeError as e:
            st.error(f"Something went wrong: {e}")
            st.stop()
        st.session_state.messages.append(
            {"role": "assistant", "content": result["ai_question"]}
        )
        st.session_state.step = "answer"
        st.rerun()

elif st.session_state.step == "answer":
    audio = st.audio_input(
        "Record your answer in German", key=f"audio_{st.session_state.audio_key}"
    )
    placeholder = 'Type answer | "1" examples | "2" expressions | "next" | "stop"'
    user_input = st.chat_input(placeholder)

    answer_text = None
    if audio:
        try:
            answer_text = transcribe_audio(audio.getvalue())
        except RuntimeError as e:
            st.error(f"Transcription failed: {e}")
            st.stop()
        st.session_state.audio_key += 1
    elif user_input:
        answer_text = user_input

    if answer_text:
        try:
            validated = UserAnswer(answer=answer_text)
        except ValidationError as e:
            st.error(e.errors()[0]["msg"])
            st.stop()
        st.session_state.messages.append({"role": "user", "content": validated.answer})
        try:
            result = topic_app.invoke(Command(resume=validated.answer), topic_config)
        except RuntimeError as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        graph_done = len(topic_app.get_state(topic_config).next) == 0

        if graph_done:
            st.session_state.step = "done"
        elif result["hint"]:
            st.session_state.messages.append(
                {"role": "assistant", "content": result["hint"]}
            )
        elif result["moved_to_next"]:
            st.session_state.messages.append(
                {"role": "assistant", "content": result["ai_question"]}
            )
        else:
            feedback_text = (
                f"**Score: {result['feedback_score']}/10**\n\n"
                f"**Correction:** {result['feedback_corrected_answer']}\n\n"
                f"**Explanation:** {result['feedback_explanation']}"
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": feedback_text}
            )

        st.rerun()

# ── Nuance Comparison ───────────────────────────────────────────────────────
elif st.session_state.step == "english_input":
    user_input = st.chat_input("Enter an English word or concept to compare in German:")
    if user_input:
        try:
            validated = UserAnswer(answer=user_input)
        except ValidationError as e:
            st.error(e.errors()[0]["msg"])
            st.stop()
        st.session_state.messages.append({"role": "user", "content": validated.answer})
        st.session_state.messages.append(
            {"role": "assistant", "content": "Looking up German equivalents..."}
        )
        try:
            result = nuance_app.invoke(
                {"english_input": validated.answer}, nuance_config
            )
        except RuntimeError as e:
            st.error(f"Something went wrong: {e}")
            st.stop()
        # Replace the "looking up..." placeholder with the actual comparison
        st.session_state.messages[-1] = {
            "role": "assistant",
            "content": result["comparison"],
        }
        st.session_state.step = "nuance_answer"
        st.rerun()

elif st.session_state.step == "nuance_answer":
    if st.button("Compare another word", use_container_width=False):
        st.session_state.nuance_session += 1
        st.session_state.step = "english_input"
        st.rerun()

    user_input = st.chat_input('Ask a follow-up question, or type "stop" to finish:')
    if user_input:
        try:
            validated = UserAnswer(answer=user_input)
        except ValidationError as e:
            st.error(e.errors()[0]["msg"])
            st.stop()
        st.session_state.messages.append({"role": "user", "content": validated.answer})
        try:
            result = nuance_app.invoke(Command(resume=validated.answer), nuance_config)
        except RuntimeError as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        graph_done = len(nuance_app.get_state(nuance_config).next) == 0

        if graph_done:
            st.session_state.step = "done"
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["follow_up_answers"][-1],
                }
            )

        st.rerun()

# ── Done ────────────────────────────────────────────────────────────────────
elif st.session_state.step == "done":
    st.info("Session complete! Refresh the page to start a new topic.")
