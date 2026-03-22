from pydantic import BaseModel, Field
from typing import Optional, List


class UserContentText(BaseModel):
    type: str = Field(default="text")
    text: str = Field(default="")


class UserContentImage(BaseModel):
    type: str = Field(default="image")
    image_url: Optional[str] = Field(default="")
    path: Optional[str] = Field(default="")


class UserContent(BaseModel):
    text: Optional[UserContentText] = Field(default=None)
    image: Optional[UserContentImage] = Field(default=None)


class UserMessage(BaseModel):
    role: str = Field(default="user")
    content: UserContent = Field(default=UserContent())
