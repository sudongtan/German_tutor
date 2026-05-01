import uuid

import streamlit as st
from langgraph.types import Command
from pydantic import ValidationError

from src.german_tutor.audio import transcribe_audio
from src.german_tutor.schemas import UserAnswer, UserTopic
from src.german_tutor.tutor import app

st.title("German Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "step" not in st.session_state:
    st.session_state.step = "topic"  # "topic" | "level" | "answer" | "done"
if "pending_topic" not in st.session_state:
    st.session_state.pending_topic = ""
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

config = {"configurable": {"thread_id": st.session_state.thread_id}}

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if st.session_state.step == "topic":
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
            result = app.invoke(
                {
                    "user_topic": st.session_state.pending_topic,
                    "user_level": selected_level,
                },
                config,
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
            result = app.invoke(Command(resume=validated.answer), config)
        except RuntimeError as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        graph_done = len(app.get_state(config).next) == 0

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

elif st.session_state.step == "done":
    st.info("Session complete! Refresh the page to start a new topic.")
