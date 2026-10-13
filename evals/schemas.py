from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


class EvalMessage(BaseModel):
    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1
    )


class RagEvalCase(BaseModel):
    id: str = Field(
        min_length=1
    )

    category: Literal[
        "factual",
        "query_rewrite",
        "no_answer",
    ]

    question: str = Field(
        min_length=1
    )

    history: list[EvalMessage] = Field(
        default_factory=list
    )

    expected_route: Literal[
        "generate",
        "no_answer",
    ]

    reference_answer: str | None = None

    expected_source: str | None = None

    @model_validator(
        mode="after"
    )
    def validate_expected_output(
        self,
    ):
        if self.expected_route == "generate":

            if not self.reference_answer:
                raise ValueError(
                    "generate cases require "
                    "reference_answer."
                )

            if not self.expected_source:
                raise ValueError(
                    "generate cases require "
                    "expected_source."
                )

        if self.expected_route == "no_answer":

            if self.reference_answer is not None:
                raise ValueError(
                    "no_answer cases must not "
                    "define reference_answer."
                )

            if self.expected_source is not None:
                raise ValueError(
                    "no_answer cases must not "
                    "define expected_source."
                )

        if (
            self.category == "query_rewrite"
            and not self.history
        ):
            raise ValueError(
                "query_rewrite cases require "
                "conversation history."
            )

        return self