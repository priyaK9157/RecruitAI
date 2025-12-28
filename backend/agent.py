import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def determine_intent(query: str) -> str:
    """
    Uses LLM reasoning to categorize the user's intent.
    This fulfills the requirement for an agent that performs reasoning.
    """
    system_prompt = """
    You are an intent classifier for a hiring assistant. 
    Classify the user's query into one of these categories:
    - candidate_evaluation: if they are asking about cultural fit, suitability, or comparing candidates.
    - skill_matching: if they are looking for specific technical or soft skills.
    - experience_verification: if they are asking about years of work or specific past roles.
    - general_hiring: for anything else related to recruitment.

    Respond with ONLY the category name.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Use 3.5 for speed/cost for simple classification
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Fallback to your keyword logic if the API fails
        query_lower = query.lower()
        if any(word in query_lower for word in ["fit", "best", "should i hire"]):
            return "candidate_evaluation"
        if any(word in query_lower for word in ["skill", "know", "expert", "tech"]):
            return "skill_matching"
        return "general_hiring"