from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from api.schemas.common import ThreadId

from core.config import ALLOWED_MODELS




MAX_CHAT_MESSAGE_LENGTH = 20_000



class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    message: str | None = Field(
        default=None,
        max_length=MAX_CHAT_MESSAGE_LENGTH,
    )

    thread_id: ThreadId

    model: str | None = None

    speech_language: Literal[
        "pl-PL",
        "en-US",
    ] = "pl-PL"

    resume: bool | None = None

    @model_validator(mode="after")
    def validate_request_mode(
        self,
    ):
        is_resume = (
            "resume"
            in self.model_fields_set
        )

        if is_resume:
            if self.resume is None:
                raise ValueError(
                    "Resume must be a boolean."
                )

            if self.message is not None:
                raise ValueError(
                    "Resume request cannot contain a message."
                )

        else:
            if (
                self.message is None
                or not self.message.strip()
            ):
                raise ValueError(
                    "Message is required."
                )

        if (
            self.model is not None
            and self.model not in ALLOWED_MODELS
        ):
            raise ValueError(
                "Unsupported model."
            )

        return self