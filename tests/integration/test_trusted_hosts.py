def test_default_test_host_is_allowed(
    api_client,
):
    response = api_client.get(
        "/health"
    )

    assert response.status_code == 200


def test_production_host_is_allowed(
    api_client,
):
    response = api_client.get(
        "/health",
        headers={
            "Host": (
                "robbotgpt.aispectra.pl"
            )
        },
    )

    assert response.status_code == 200


def test_localhost_with_port_is_allowed(
    api_client,
):
    response = api_client.get(
        "/health",
        headers={
            "Host": "localhost:8080"
        },
    )

    assert response.status_code == 200


def test_untrusted_host_is_rejected(
    api_client,
):
    response = api_client.get(
        "/health",
        headers={
            "Host": "evil.example"
        },
    )

    assert response.status_code == 400

    assert (
        response.text
        == "Invalid host header"
    )


def test_untrusted_host_response_has_security_headers(
    api_client,
):
    response = api_client.get(
        "/health",
        headers={
            "Host": "evil.example"
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