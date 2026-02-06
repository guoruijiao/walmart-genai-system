from fastapi import FastAPI, Query
from pydantic import BaseModel

from walmart_genai.core.generate import answer_question
from walmart_genai.core.schema import GenAIResponse

app = FastAPI(title="Walmart GenAI System")


class QueryRequest(BaseModel):
    question: str


@app.post("/query", response_model=GenAIResponse)
def query(
    req: QueryRequest,
    mode: str = Query("structured", pattern="^(base|structured)$"),
) -> GenAIResponse:
    return answer_question(req.question, mode=mode)
