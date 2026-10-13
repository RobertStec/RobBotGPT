from langsmith import Client

from core.config import settings


def get_langsmith_client() -> Client:
    """
    Create a configured LangSmith client.
    """

    if settings.langsmith_api_key is None:
        raise RuntimeError(
            "LANGSMITH_API_KEY is required "
            "for LangSmith evaluation."
        )

    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=(
            settings.langsmith_api_key
            .get_secret_value()
        ),
        workspace_id=(
            settings.langsmith_workspace_id
        ),
    )