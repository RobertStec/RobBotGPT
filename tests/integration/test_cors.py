PRODUCTION_ORIGIN = (
    "https://robbotgpt.aispectra.pl"
)




def test_allowed_origin_preflight(
    api_client,
):
    response = api_client.options(
        "/chat/stream",
        headers={
            "Origin": PRODUCTION_ORIGIN,
            (
                "Access-Control-"
                "Request-Method"
            ): "POST",
            (
                "Access-Control-"
                "Request-Headers"
            ): "Content-Type",
        },
    )

    assert response.status_code == 200

    assert (
        response.headers[
            "access-control-allow-origin"
        ]
        == PRODUCTION_ORIGIN
    )

    assert (
        response.headers[
            "access-control-allow-credentials"
        ]
        == "true"
    )




def test_disallowed_origin_preflight(
    api_client,
):
    response = api_client.options(
        "/chat/stream",
        headers={
            "Origin": "https://evil.example",
            (
                "Access-Control-"
                "Request-Method"
            ): "POST",
        },
    )

    assert response.status_code == 400

    assert (
        "access-control-allow-origin"
        not in response.headers
    )




def test_disallowed_cors_method(
    api_client,
):
    response = api_client.options(
        "/chat/stream",
        headers={
            "Origin": PRODUCTION_ORIGIN,
            (
                "Access-Control-"
                "Request-Method"
            ): "PUT",
        },
    )

    assert response.status_code == 400




def test_allowed_origin_simple_request(
    api_client,
):
    response = api_client.get(
        "/health",
        headers={
            "Origin": PRODUCTION_ORIGIN,
        },
    )

    assert response.status_code == 200

    assert (
        response.headers[
            "access-control-allow-origin"
        ]
        == PRODUCTION_ORIGIN
    )




def test_cors_error_has_security_headers(
    api_client,
):
    response = api_client.options(
        "/chat/stream",
        headers={
            "Origin": "https://evil.example",
            (
                "Access-Control-"
                "Request-Method"
            ): "POST",
        },
    )

    assert response.status_code == 400

    assert (
        response.headers[
            "x-content-type-options"
        ]
        == "nosniff"
    )

    assert (
        response.headers[
            "x-frame-options"
        ]
        == "DENY"
    )