from pydantic import (
    BaseModel,
    Field,
)

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)




class RerankItem(BaseModel):
    """
    Relevance and answerability assessment
    assigned to one RAG chunk.
    """

    chunk_id: int

    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    answerable: bool



class RerankResult(BaseModel):
    """
    Structured result returned by the LLM reranker.
    """

    items: list[RerankItem]



class AnswerabilityResult(BaseModel):
    answerable: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reasoning: str





def rerank_candidates(
    query: str,
    candidates: list[dict],
    llm,
    top_k: int,
) -> list[dict]:
    """
    Rerank RAG candidates using an LLM.

    Candidates are initially ordered by pgvector
    cosine distance. The LLM performs a second-stage
    relevance ranking against the rewritten query.
    """

    if top_k < 1:
        raise ValueError(
            "top_k must be greater than 0."
        )

    if not candidates:
        return []

    candidate_by_id = {
        candidate["chunk_id"]: candidate
        for candidate in candidates
    }

    candidate_blocks = []

    for candidate in candidates:
        candidate_blocks.append(
            (
                f"CHUNK_ID: "
                f"{candidate['chunk_id']}\n"
                f"SOURCE: "
                f"{candidate['source']}\n"
                f"CONTENT:\n"
                f"{candidate['content']}"
            )
        )

    candidates_text = "\n\n---\n\n".join(
        candidate_blocks
    )

    prompt = f"""
Evaluate each document chunk for retrieval relevance
and answerability with respect to the user's query.

Search query:
{query}

Document chunks:

{candidates_text}

Rules:
- Score every chunk from 0.0 to 1.0.
- The score measures how useful the chunk is for
  answering the exact query, not only whether it is
  about a similar topic.
- Set answerable=true only when the chunk contains
  explicit information that answers the query or a
  meaningful part of a compound query.
- A chunk about a different named entity is not
  answerable even when the topic is similar.
  Example: information about project Atlas does not
  answer a question about project Orion or Apollo.
- Do not infer facts that are not explicitly supported.
- A deadline does not prove when a project started.
- A planned deadline does not prove whether a project
  is delayed.
- Mentioning a person does not provide their private
  contact details.
- For a compound question, answerable may be true when
  the chunk explicitly answers at least one substantive
  part of the question.
- 1.0 means the chunk contains direct, highly useful
  evidence for answering the exact query.
- 0.0 means the chunk cannot help answer the query.
- Do not answer the user's question.
- Use only the provided chunk IDs.
"""

    structured_llm = (
        llm.with_structured_output(
            RerankResult
        )
    )

    try:
        result = structured_llm.invoke([
            SystemMessage(
                content=(
                    "You are a strict RAG reranking and "
                    "answerability classification system."
                )
            ),
            HumanMessage(
                content=prompt
            ),
        ])

    except Exception:
        # Safe fallback:
        # keep original pgvector ranking.
        return candidates[:top_k]

    ranked_items = sorted(
        result.items,
        key=lambda item: item.score,
        reverse=True,
    )

    reranked = []
    seen_chunk_ids = set()

    for item in ranked_items:
        candidate = candidate_by_id.get(
            item.chunk_id
        )

        if candidate is None:
            continue

        if item.chunk_id in seen_chunk_ids:
            continue

        reranked_candidate = dict(
            candidate
        )

        reranked_candidate[
            "rerank_score"
        ] = float(item.score)

        reranked_candidate[
            "rerank_answerable"
        ] = item.answerable

        reranked.append(
            reranked_candidate
        )

        seen_chunk_ids.add(
            item.chunk_id
        )

    # If the model omitted some chunks,
    # preserve the original pgvector ordering
    # as a fallback.
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]

        if chunk_id in seen_chunk_ids:
            continue

        reranked.append(
            dict(candidate)
        )

    return reranked[:top_k]





def filter_relevant_candidates(
    candidates: list[dict],
    threshold: float,
) -> list[dict]:
    """
    Keep only candidates whose rerank score meets
    the configured relevance threshold.

    If reranking failed and no candidate contains
    a rerank score, preserve the original candidates
    as a safe availability fallback.
    """

    if not candidates:
        return []

    scored_candidates = [
        candidate
        for candidate in candidates
        if candidate.get(
            "rerank_score"
        ) is not None
    ]

    # Reranker failure fallback:
    # candidates came from pgvector without LLM scores.
    if not scored_candidates:
        return candidates

    return [
        candidate
        for candidate in scored_candidates
        if (
            candidate["rerank_score"]
            >= threshold
            and candidate.get(
                "rerank_answerable",
                True,
            )
        )
    ]




def assess_query_answerability(
    query: str,
    candidates: list[dict],
    llm,
) -> bool:
    """
    Decide whether the selected RAG evidence
    explicitly supports answering the query.
    """

    if not candidates:
        return False

    evidence = "\n\n---\n\n".join(
        candidate["content"]
        for candidate in candidates
    )

    prompt = f"""
Determine whether the provided document evidence
contains enough explicit information to answer at
least one substantive part of the user's exact query.

User query:
{query}

Document evidence:

{evidence}

Strict rules:
- Return answerable=true only when the evidence
  explicitly supports an answer.
- Do not infer, guess, or assume missing facts.
- Similar subject matter is not sufficient.
- Problems, risks, or technical issues do not prove
  that a project is delayed.
- A deadline does not prove whether a project is
  delayed.
- A pilot date does not prove when the whole project
  started.
- Evidence about one named entity cannot answer a
  question about another named entity.
- For a compound question, answerable=true if at least
  one substantive part can be explicitly answered.
"""

    structured_llm = (
        llm.with_structured_output(
            AnswerabilityResult
        )
    )

    try:
        result = structured_llm.invoke([
            SystemMessage(
                content=(
                    "You are a strict RAG "
                    "answerability verifier."
                )
            ),
            HumanMessage(
                content=prompt
            ),
        ])

    except Exception:
        # Preserve current availability behaviour
        # if the verifier itself fails.
        return True

    return result.answerable