def test_expected_routes_are_registered(
    integration_app,
):
    schema = integration_app.app.openapi()

    paths = set(
        schema["paths"].keys()
    )

    assert "/conversations" in paths

    assert (
        "/conversations/{thread_id}"
        in paths
    )

    assert (
        "/history/{thread_id}"
        in paths
    )

    assert "/upload" in paths

    assert "/chat/stream" in paths