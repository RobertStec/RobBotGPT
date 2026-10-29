import database

import pytest

import core.rate_limit as rate_limit

from core.exceptions import (
    RateLimitExceededError,
)

from core.security import (
    hash_session_token,
    verify_password,
)




def test_register_user_success(
    api_client,
):
    response = api_client.post(
        "/auth/register",
        json={
            "email": "user@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["success"] is True

    assert data["user"]["email"] == (
        "user@example.com"
    )

    assert isinstance(
        data["user"]["id"],
        int,
    )

    assert "password" not in data["user"]
    assert "password_hash" not in data["user"]




def test_register_user_persists_hashed_password(
    api_client,
):
    password = "StrongPassword123!"

    response = api_client.post(
        "/auth/register",
        json={
            "email": "stored@example.com",
            "password": password,
        },
    )

    assert response.status_code == 201

    user = database.get_user_by_email(
        "stored@example.com"
    )

    assert user is not None

    assert user.password_hash != password

    assert verify_password(
        password,
        user.password_hash,
    ) is True




def test_register_user_normalizes_email(
    api_client,
):
    response = api_client.post(
        "/auth/register",
        json={
            "email": "USER@Example.COM",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 201

    assert (
        response.json()["user"]["email"]
        == "user@example.com"
    )

    user = database.get_user_by_email(
        "user@example.com"
    )

    assert user is not None




def test_register_user_rejects_duplicate_email(
    api_client,
):
    payload = {
        "email": "duplicate@example.com",
        "password": "StrongPassword123!",
    }

    first_response = api_client.post(
        "/auth/register",
        json=payload,
    )

    second_response = api_client.post(
        "/auth/register",
        json=payload,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409

    assert second_response.json() == {
        "success": False,
        "message": (
            "A user with this email already exists."
        ),
        "error_code": "user_already_exists",
    }




def test_register_user_rejects_invalid_email(
    api_client,
):
    response = api_client.post(
        "/auth/register",
        json={
            "email": "not-an-email",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 422




def test_register_user_rejects_short_password(
    api_client,
):
    response = api_client.post(
        "/auth/register",
        json={
            "email": "user@example.com",
            "password": "short",
        },
    )

    assert response.status_code == 422




def test_login_user_success(
    api_client,
):
    password = "StrongPassword123!"

    register_response = api_client.post(
        "/auth/register",
        json={
            "email": "login@example.com",
            "password": password,
        },
    )

    assert register_response.status_code == 201

    response = api_client.post(
        "/auth/login",
        json={
            "email": "login@example.com",
            "password": password,
        },
    )

    assert response.status_code == 200

    assert response.json()["success"] is True

    assert (
        response.json()["user"]["email"]
        == "login@example.com"
    )

    assert (
        api_client.cookies.get(
            "robbotgpt_session"
        )
        is not None
    )




def test_login_creates_hashed_session_in_database(
    api_client,
):
    password = "StrongPassword123!"

    api_client.post(
        "/auth/register",
        json={
            "email": "session@example.com",
            "password": password,
        },
    )

    response = api_client.post(
        "/auth/login",
        json={
            "email": "session@example.com",
            "password": password,
        },
    )

    assert response.status_code == 200

    raw_token = api_client.cookies.get(
        "robbotgpt_session"
    )

    assert raw_token is not None

    token_hash = hash_session_token(
        raw_token
    )

    auth_session = (
        database.get_auth_session_by_token_hash(
            token_hash
        )
    )

    assert auth_session is not None




def test_login_rejects_wrong_password(
    api_client,
):
    api_client.post(
        "/auth/register",
        json={
            "email": "wrong@example.com",
            "password": "StrongPassword123!",
        },
    )

    response = api_client.post(
        "/auth/login",
        json={
            "email": "wrong@example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401

    assert response.json() == {
        "success": False,
        "message": "Invalid email or password.",
        "error_code": "invalid_credentials",
    }




def test_login_rejects_unknown_user(
    api_client,
):
    response = api_client.post(
        "/auth/login",
        json={
            "email": "missing@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 401

    assert response.json()["error_code"] == (
        "invalid_credentials"
    )




def test_auth_me_returns_logged_in_user(
    api_client,
):
    password = "StrongPassword123!"

    api_client.post(
        "/auth/register",
        json={
            "email": "me@example.com",
            "password": password,
        },
    )

    api_client.post(
        "/auth/login",
        json={
            "email": "me@example.com",
            "password": password,
        },
    )

    response = api_client.get(
        "/auth/me"
    )

    assert response.status_code == 200

    assert (
        response.json()["email"]
        == "me@example.com"
    )





def test_auth_me_requires_authentication(
    api_client,
):
    response = api_client.get(
        "/auth/me"
    )

    assert response.status_code == 401

    assert response.json() == {
        "success": False,
        "message": "Authentication required.",
        "error_code": "authentication_required",
    }





def test_logout_removes_session(
    api_client,
):
    password = "StrongPassword123!"

    api_client.post(
        "/auth/register",
        json={
            "email": "logout@example.com",
            "password": password,
        },
    )

    api_client.post(
        "/auth/login",
        json={
            "email": "logout@example.com",
            "password": password,
        },
    )

    assert (
        api_client.cookies.get(
            "robbotgpt_session"
        )
        is not None
    )

    response = api_client.post(
        "/auth/logout"
    )

    assert response.status_code == 200

    assert response.json() == {
        "success": True
    }

    assert (
        api_client.cookies.get(
            "robbotgpt_session"
        )
        is None
    )

    me_response = api_client.get(
        "/auth/me"
    )

    assert me_response.status_code == 401





def test_logout_without_session_is_idempotent(
    api_client,
):
    response = api_client.post(
        "/auth/logout"
    )

    assert response.status_code == 200

    assert response.json() == {
        "success": True
    }




def test_register_applies_ip_rate_limit(
    api_client,
    mocker,
):
    mock_enforce = mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new_callable=mocker.AsyncMock,
    )

    response = api_client.post(
        "/auth/register",
        headers={
            "X-Real-IP": "203.0.113.10"
        },
        json={
            "email": "limited-register@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 201

    mock_enforce.assert_awaited_once_with(
        scope="auth_register",
        identifier="ip:203.0.113.10",
        limit=5,
        window_seconds=3600,
    )




def test_login_applies_ip_rate_limit(
    api_client,
    mocker,
):
    password = "StrongPassword123!"

    api_client.post(
        "/auth/register",
        json={
            "email": "limited-login@example.com",
            "password": password,
        },
    )

    mock_enforce = mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new_callable=mocker.AsyncMock,
    )

    response = api_client.post(
        "/auth/login",
        headers={
            "X-Real-IP": "203.0.113.20"
        },
        json={
            "email": "limited-login@example.com",
            "password": password,
        },
    )

    assert response.status_code == 200

    mock_enforce.assert_awaited_once_with(
        scope="auth_login",
        identifier="ip:203.0.113.20",
        limit=10,
        window_seconds=300,
    )




def test_register_returns_429_when_rate_limit_exceeded(
    api_client,
    mocker,
):
    mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new=mocker.AsyncMock(
            side_effect=RateLimitExceededError(
                retry_after=120
            )
        ),
    )

    response = api_client.post(
        "/auth/register",
        json={
            "email": "blocked-register@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 429

    assert (
        response.headers["retry-after"]
        == "120"
    )

    assert response.json()[
        "error_code"
    ] == "rate_limit_exceeded"





def test_login_returns_429_when_rate_limit_exceeded(
    api_client,
    mocker,
):
    mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new=mocker.AsyncMock(
            side_effect=RateLimitExceededError(
                retry_after=45
            )
        ),
    )

    response = api_client.post(
        "/auth/login",
        json={
            "email": "blocked@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 429

    assert (
        response.headers["retry-after"]
        == "45"
    )

    assert response.json()[
        "error_code"
    ] == "rate_limit_exceeded"




def test_auth_me_applies_user_rate_limit(
    api_client,
    mocker,
):
    password = "StrongPassword123!"

    api_client.post(
        "/auth/register",
        json={
            "email": "me-rate-limit@example.com",
            "password": password,
        },
    )

    login_response = api_client.post(
        "/auth/login",
        json={
            "email": "me-rate-limit@example.com",
            "password": password,
        },
    )

    user = login_response.json()["user"]

    mock_enforce = mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new_callable=mocker.AsyncMock,
    )

    response = api_client.get(
        "/auth/me"
    )

    assert response.status_code == 200

    mock_enforce.assert_awaited_once_with(
        scope="auth_me",
        identifier=f"user:{user['id']}",
        limit=120,
        window_seconds=60,
    )




def test_auth_me_returns_429_when_rate_limit_exceeded(
    api_client,
    mocker,
):
    password = "StrongPassword123!"

    api_client.post(
        "/auth/register",
        json={
            "email": "me-blocked@example.com",
            "password": password,
        },
    )

    api_client.post(
        "/auth/login",
        json={
            "email": "me-blocked@example.com",
            "password": password,
        },
    )

    mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new=mocker.AsyncMock(
            side_effect=RateLimitExceededError(
                retry_after=15
            )
        ),
    )

    response = api_client.get(
        "/auth/me"
    )

    assert response.status_code == 429

    assert response.headers[
        "retry-after"
    ] == "15"

    assert response.json()[
        "error_code"
    ] == "rate_limit_exceeded"




def test_logout_applies_ip_rate_limit(
    api_client,
    mocker,
):
    mock_enforce = mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new_callable=mocker.AsyncMock,
    )

    response = api_client.post(
        "/auth/logout",
        headers={
            "X-Real-IP": "203.0.113.30"
        },
    )

    assert response.status_code == 200

    mock_enforce.assert_awaited_once_with(
        scope="auth_logout",
        identifier="ip:203.0.113.30",
        limit=30,
        window_seconds=60,
    )



def test_logout_returns_429_when_rate_limit_exceeded(
    api_client,
    mocker,
):
    mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new=mocker.AsyncMock(
            side_effect=RateLimitExceededError(
                retry_after=12
            )
        ),
    )

    response = api_client.post(
        "/auth/logout"
    )

    assert response.status_code == 429

    assert response.headers[
        "retry-after"
    ] == "12"

    assert response.json()[
        "error_code"
    ] == "rate_limit_exceeded"