import json

from walmart_genai.core.llm import get_client, get_model
from walmart_genai.core.schema import GenAIResponse

SYSTEM = """You are a Walmart retail assistant.
Return ONLY valid JSON that matches this schema:
{
  "answer": string,
  "citations": [{"source": string, "quote": string}],
  "intent": "product_info|delivery|return|store_info|other",
  "entities": {"product_id": string|null, "store_id": string|null},
  "confidence": number between 0 and 1,
  "next_action": "respond|ask_clarification|handoff_human"
}
If you don't have enough evidence, set next_action="ask_clarification" and explain what you need.
"""


def answer_question(question: str) -> GenAIResponse:
    client = get_client()
    model = get_model()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
    )

    text = resp.output_text.strip()
    data = json.loads(text)
    return GenAIResponse.model_validate(data)
