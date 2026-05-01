import uuid

import streamlit as st
from langgraph.types import Command
from pydantic import ValidationError

from src.german_tutor.schemas import UserAnswer, UserTopic
from src.german_tutor.tutor import app

st.title("German Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "step" not in st.session_state:
    st.session_state.step = "topic"  # "topic" | "answer" | "done"

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
        try:
            result = app.invoke({"user_topic": validated.topic}, config)
        except RuntimeError as e:
            st.error(f"Something went wrong: {e}")
            st.stop()
        question = result["ai_question"]
        st.session_state.messages.append({"role": "assistant", "content": question})
        st.session_state.step = "answer"
        st.rerun()

elif st.session_state.step == "answer":
    placeholder = 'Answer in German, or type "stop" to finish'
    user_input = st.chat_input(placeholder)
    if user_input:
        try:
            validated = UserAnswer(answer=user_input)
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
        else:
            feedback = result["ai_feedback"]
            new_question = result["ai_question"]
            st.session_state.messages.append({"role": "assistant", "content": feedback})
            st.session_state.messages.append({"role": "assistant", "content": new_question})

        st.rerun()

elif st.session_state.step == "done":
    st.info("Session complete! Refresh the page to start a new topic.")
