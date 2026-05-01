# German Tutor

An AI-powered German language tutor built with LangGraph and Streamlit. The tutor generates German questions on a topic of your choice, evaluates your answers, and loops until you type `stop`.

## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — conversation graph with interrupt-based human-in-the-loop
- **[Streamlit](https://streamlit.io)** — chat UI
- **[OpenRouter](https://openrouter.ai)** — unified API for any LLM (OpenAI, Anthropic, Google, etc.)
- **[Pydantic](https://docs.pydantic.dev)** — input validation and settings management
- **[uv](https://github.com/astral-sh/uv)** — dependency management

## Setup

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd german_tutor
uv sync --all-groups
```

**2. Set up environment variables**

Copy the example and fill in your key:
```bash
cp .env.example .env
```

Get your API key at [openrouter.ai/keys](https://openrouter.ai/keys).

**3. Run the app**
```bash
uv run streamlit run app.py
```

## Switching models

Change `LLM_MODEL` in [`src/german_tutor/config.py`](src/german_tutor/config.py) to any model available on OpenRouter:

```python
LLM_MODEL = "anthropic/claude-opus-4-5"
LLM_MODEL = "google/gemini-pro-1.5"
LLM_MODEL = "openai/gpt-4o"
```

## Project structure

```
src/german_tutor/
├── tutor.py      # LangGraph graph and nodes
├── prompts.py    # LLM prompt strings
├── schemas.py    # Pydantic input validation models
└── config.py     # Settings (model, API key)

tests/
├── test_schemas.py           # Input validation tests
├── test_prompts.py           # Prompt template tests
├── test_config.py            # Settings validation tests
└── test_generate_response.py # LLM call error handling tests

app.py            # Streamlit UI
```

## Running tests

```bash
uv run pytest tests/ -v
```

## CI

GitHub Actions runs lint and tests on every push and PR to `main` and `dev`. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
