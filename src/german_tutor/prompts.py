GENERATE_QUESTIONS_BATCH = (
    "Generate exactly 10 questions about {topic} in German at {level} level.{avoid}"
)

AVOID_REPEAT = " Do not ask any of these questions again: {previous}"

GENERATE_FEEDBACK = (
    "You are a German language tutor. The student is at {level} level. "
    "Evaluate the user's German answer. "
    "Provide a corrected version of their answer, an explanation of any errors, "
    "and a score from 1 to 10."
)

GENERATE_EXAMPLES = (
    "Generate 2-3 example answers in German at {level} level for this question: {question}. "
    "Include English translations."
)

GENERATE_EXPRESSIONS = (
    "List 5-6 useful German expressions or vocabulary at {level} level for answering "
    "this question: {question}. Include English translations."
)
