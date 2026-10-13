from pydantic import (
    BaseModel,
    Field,
)

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from core.config import settings




def route_correctness(
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    """
    Check whether the RAG pipeline selected
    the expected final route.
    """

    return (
        outputs.get("route")
        == reference_outputs.get(
            "expected_route"
        )
    )




def source_correctness(
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    """
    Check whether the expected document source
    is present in the returned RAG sources.
    """

    expected_source = (
        reference_outputs.get(
            "expected_source"
        )
    )

    sources = outputs.get(
        "sources",
        [],
    )

    if expected_source is None:
        return len(sources) == 0

    return any(
        source.get("source")
        == expected_source
        for source in sources
    )




def no_answer_correctness(
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    """
    Check whether no-answer examples are handled
    through the no_answer route with no sources.
    """

    expected_route = (
        reference_outputs.get(
            "expected_route"
        )
    )

    if expected_route != "no_answer":
        return True

    return (
        outputs.get("route")
        == "no_answer"
        and outputs.get("sources", [])
        == []
    )




class AnswerCorrectnessGrade(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    reasoning: str = Field(
        min_length=1
    )




def build_answer_correctness_evaluator(
    judge_llm=None,
):
    """
    Build an LLM-as-a-judge evaluator that compares
    the generated answer with the reference answer.
    """

    if judge_llm is None:
        judge_llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    structured_judge = (
        judge_llm.with_structured_output(
            AnswerCorrectnessGrade
        )
    )


    def answer_correctness(
        inputs: dict,
        outputs: dict,
        reference_outputs: dict,
    ) -> dict:

        reference_answer = (
            reference_outputs.get(
                "reference_answer"
            )
        )

        # This evaluator is not applicable
        # to no-answer examples.
        if reference_answer is None:
            return {
                "key": "answer_correctness",
                "score": None,
                "comment": (
                    "Not applicable to "
                    "no-answer example."
                ),
            }

        actual_answer = outputs.get(
            "answer",
            "",
        )

        if not isinstance(
            actual_answer,
            str,
        ) or not actual_answer.strip():

            return {
                "key": "answer_correctness",
                "score": 0.0,
                "comment": (
                    "The application returned "
                    "no answer."
                ),
            }

        question = inputs.get(
            "question",
            "",
        )

        messages = [
            SystemMessage(
                content=(
                    "You are evaluating the factual "
                    "correctness of a RAG answer.\n\n"
                    "Compare the generated answer with "
                    "the reference answer.\n\n"
                    "Score from 0.0 to 1.0:\n"
                    "- 1.0 = fully correct and consistent\n"
                    "- 0.5 = partially correct\n"
                    "- 0.0 = incorrect or contradictory\n\n"
                    "Do not require exact wording. "
                    "Evaluate semantic and factual "
                    "correctness only."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Reference answer:\n"
                    f"{reference_answer}\n\n"
                    f"Generated answer:\n"
                    f"{actual_answer}"
                )
            ),
        ]

        grade = structured_judge.invoke(
            messages
        )

        return {
            "key": "answer_correctness",
            "score": grade.score,
            "comment": grade.reasoning,
        }

    return answer_correctness