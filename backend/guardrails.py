import re

# Defined Banned keywords
BANNED_TOPICS = ["race", "gender", "religion", "illegal", "political", "medical advice"]

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"you are now a (.*) model",
    r"forget your (rules|boundaries)"
]

def validate_query(query: str) -> bool:
    """
    Validates input to prevent safety violations and prompt injections.
    """
    query_lower = query.lower()

    # 1. Keyword check (Your original logic)
    for word in BANNED_TOPICS:
        if word in query_lower:
            return False

    # 2. Regex check for Prompt Injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return False

    # 3. Length check (Prevents token-stuffing attacks)
    if len(query) > 1000:
        return False

    return True

def filter_output(response: str) -> str:
    """
    Scans the AI response before it reaches the user.
    Ensures the agent hasn't accidentally mentioned sensitive topics.
    """
    # Define clear boundaries for what the agent will NOT answer
    safety_fallback = "I cannot provide this information as it falls outside of professional hiring guidelines."
    
    for word in BANNED_TOPICS:
        if word in response.lower():
            return safety_fallback
            
    return response