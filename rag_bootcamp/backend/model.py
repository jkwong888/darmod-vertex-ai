
from pydantic import BaseModel


class LLMRequest(BaseModel):
    query: str

class LLMResponse(BaseModel):
    response: str
