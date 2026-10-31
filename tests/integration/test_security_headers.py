EXPECTED_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": (
        "camera=(), "
        "geolocation=(), "
        "payment=(), "
        "usb=()"
    ),
    "content-security-policy": (
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "form-action 'self'"
    ),
}


def _assert_security_headers(
    response,
):
    for (
        name,
        expected_value,
    ) in EXPECTED_SECURITY_HEADERS.items():
        assert (
            response.headers[name]
            == expected_value
        )


def test_security_headers_are_added_to_success_response(
    api_client,
):
    response = api_client.get(
        "/health"
    )

    assert response.status_code == 200

    _assert_security_headers(
        response
    )


def test_security_headers_are_added_to_error_response(
    api_client,
):
    response = api_client.get(
        "/auth/me"
    )

    assert response.status_code == 401

    _assert_security_headers(
        response
    )


def test_security_headers_are_added_to_validation_error(
    api_client,
):
    response = api_client.post(
        "/auth/login",
        json={
            "email": "invalid"
        },
    )

    assert response.status_code == 422

    _assert_security_headers(
        response
    )