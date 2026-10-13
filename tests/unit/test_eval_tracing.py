from evals.tracing import (
    RAG_PIPELINE_VERSION,
    build_eval_metadata,
    build_eval_tags,
    build_eval_trace_config,
)






def test_build_eval_tags_contains_required_tags():
    tags = build_eval_tags(
        "gpt-4o-mini"
    )

    assert "robbotgpt" in tags
    assert "rag" in tags
    assert "offline-evaluation" in tags

    assert (
        f"pipeline:{RAG_PIPELINE_VERSION}"
        in tags
    )

    assert (
        "model:gpt-4o-mini"
        in tags
    )






def test_build_eval_metadata_contains_rag_configuration():
    metadata = build_eval_metadata(
        "gpt-4o-mini"
    )

    assert (
        metadata["application"]
        == "RobBotGPT"
    )

    assert (
        metadata["component"]
        == "advanced_rag"
    )

    assert (
        metadata["pipeline_version"]
        == RAG_PIPELINE_VERSION
    )

    assert (
        metadata["model"]
        == "gpt-4o-mini"
    )

    assert (
        metadata["retrieval_top_k"]
        > 0
    )

    assert (
        metadata["final_top_k"]
        > 0
    )

    assert (
        0.0
        <= metadata[
            "relevance_threshold"
        ]
        <= 1.0
    )

    assert (
        metadata["context_max_chars"]
        > 0
    )





def test_build_eval_trace_config_has_tags_and_metadata():
    config = build_eval_trace_config(
        "gpt-4o-mini"
    )

    assert set(
        config.keys()
    ) == {
        "tags",
        "metadata",
    }

    assert isinstance(
        config["tags"],
        list,
    )

    assert isinstance(
        config["metadata"],
        dict,
    )