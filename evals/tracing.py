from core.config import settings


RAG_PIPELINE_VERSION = "advanced-rag-v1"





def build_eval_tags(
    model_name: str,
) -> list[str]:
    """
    Build stable LangSmith tags for RAG evaluations.
    """

    return [
        "robbotgpt",
        "rag",
        "offline-evaluation",
        f"pipeline:{RAG_PIPELINE_VERSION}",
        f"model:{model_name}",
    ]





def build_eval_metadata(
    model_name: str,
) -> dict:
    """
    Build LangSmith metadata describing
    the evaluated RAG configuration.
    """

    return {
        "application": "RobBotGPT",
        "component": "advanced_rag",
        "environment": settings.app_env,
        "pipeline_version": (
            RAG_PIPELINE_VERSION
        ),
        "model": model_name,
        "retrieval_top_k": (
            settings.rag_retrieval_top_k
        ),
        "final_top_k": (
            settings.rag_final_top_k
        ),
        "relevance_threshold": (
            settings.rag_relevance_threshold
        ),
        "context_max_chars": (
            settings.rag_context_max_chars
        ),
    }





def build_eval_trace_config(
    model_name: str,
) -> dict:
    """
    Build RunnableConfig-compatible tracing data.
    """

    return {
        "tags": build_eval_tags(
            model_name
        ),
        "metadata": build_eval_metadata(
            model_name
        ),
    }