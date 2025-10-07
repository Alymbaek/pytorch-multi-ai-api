from pydantic import BaseModel


class AgNews(BaseModel):
    text: str