"""
Prompt templates for all LLM calls in the application.

All templates use str.format() placeholders. Required keys are documented
inline with each constant.
"""

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

GENERATE_EQUIVALENTS = (
    'Give German equivalents for the English concept or word: "{english}". '
    "Rank by how commonly they are used in modern spoken German. "
    "Avoid rare or outdated words."
)

GENERATE_NUANCE_COMPARISON = (
    "You are a German language expert. Compare these German words or expressions "
    'that translate to "{english}": {equivalents}. '
    "Use the available tools to look up corpus frequency and synonyms (with register labels) "
    "for each word to inform your analysis. "
    "Then write a clear comparison explaining nuances, register differences, and typical "
    "contexts. Rank them by practical usefulness for a language learner."
)

STRUCTURE_COMPARISON = (
    "Extract structured nuance information for each German word from the following "
    "comparison. For each word include: a frequency description (e.g. very common, "
    "moderately common, rare), key synonyms, a nuance explanation, and a formality "
    "level (colloquial, neutral, formal, or elevated).\n\n{comparison}"
)

GENERATE_FOLLOW_UP_ANSWER = (
    "You are a German language expert. The user is asking follow-up questions about "
    'the nuances between these German equivalents of "{english}": {equivalents}. '
    "The comparison you provided was:\n{comparison}\n\n"
    "Answer follow-up questions concisely, referring to previous answers where relevant."
)
