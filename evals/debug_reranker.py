from langchain_openai import ChatOpenAI

from core.config import settings

from evals.seed_corpus import (
    EVAL_THREAD_ID,
    get_eval_user,
)

from rag import retrieve_candidates_from_rag

from reranker import (
    assess_query_answerability,
    filter_relevant_candidates,
    rerank_candidates,
)


QUERY = "Czy projekt Atlas jest opóźniony?"


def main():
    user = get_eval_user()

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    candidates = retrieve_candidates_from_rag(
        query=QUERY,
        thread_id=EVAL_THREAD_ID,
        user_id=user.id,
        k=settings.rag_retrieval_top_k,
    )

    reranked = rerank_candidates(
        query=QUERY,
        candidates=candidates,
        llm=llm,
        top_k=settings.rag_final_top_k,
    )

    relevant = filter_relevant_candidates(
        candidates=reranked,
        threshold=settings.rag_relevance_threshold,
    )

    globally_answerable = assess_query_answerability(
        query=QUERY,
        candidates=relevant,
        llm=llm,
    )

    print("\nQUERY:")
    print(QUERY)

    print("\nRERANKED:")

    for candidate in reranked:
        print("-" * 70)
        print(
            "chunk_id:",
            candidate["chunk_id"],
        )
        print(
            "distance:",
            candidate["distance"],
        )
        print(
            "rerank_score:",
            candidate.get(
                "rerank_score"
            ),
        )
        print(
            "answerable:",
            candidate.get(
                "rerank_answerable"
            ),
        )
        print(
            "content:",
            candidate["content"],
        )

    print("\nRELEVANT AFTER FILTER:")
    print(
        [
            candidate["chunk_id"]
            for candidate in relevant
        ]
    )

    print("\nGLOBAL ANSWERABILITY:")
    print(globally_answerable)


if __name__ == "__main__":
    main()