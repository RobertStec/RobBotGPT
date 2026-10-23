import pytest

from sqlalchemy.exc import IntegrityError

from core.exceptions import (
    AuthenticationRequiredError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
)

import services.auth_service as auth_service




def test_register_user_normalizes_email_and_hashes_password(
    mocker,
):
    mocker.patch.object(
        auth_service,
        "get_user_by_email",
        return_value=None,
    )

    mock_hash_password = mocker.patch.object(
        auth_service,
        "hash_password",
        return_value="hashed-password",
    )

    mock_user = mocker.Mock(
        id=1,
        email="user@example.com",
    )

    mock_create_user = mocker.patch.object(
        auth_service,
        "create_user",
        return_value=mock_user,
    )

    result = auth_service.register_user(
        email="  USER@Example.COM  ",
        password="StrongPassword123!",
    )

    assert result is mock_user

    auth_service.get_user_by_email.assert_called_once_with(
        "user@example.com"
    )

    mock_hash_password.assert_called_once_with(
        "StrongPassword123!"
    )

    mock_create_user.assert_called_once_with(
        email="user@example.com",
        password_hash="hashed-password",
    )




def test_register_user_rejects_existing_email(
    mocker,
):
    existing_user = mocker.Mock()

    mocker.patch.object(
        auth_service,
        "get_user_by_email",
        return_value=existing_user,
    )

    mock_hash_password = mocker.patch.object(
        auth_service,
        "hash_password",
    )

    mock_create_user = mocker.patch.object(
        auth_service,
        "create_user",
    )

    with pytest.raises(
        UserAlreadyExistsError
    ):
        auth_service.register_user(
            email="user@example.com",
            password="StrongPassword123!",
        )

    mock_hash_password.assert_not_called()
    mock_create_user.assert_not_called()




def test_register_user_converts_integrity_error(
    mocker,
):
    mocker.patch.object(
        auth_service,
        "get_user_by_email",
        return_value=None,
    )

    mocker.patch.object(
        auth_service,
        "hash_password",
        return_value="hashed-password",
    )

    mocker.patch.object(
        auth_service,
        "create_user",
        side_effect=IntegrityError(
            "INSERT",
            {},
            Exception("duplicate"),
        ),
    )

    with pytest.raises(
        UserAlreadyExistsError
    ):
        auth_service.register_user(
            email="user@example.com",
            password="StrongPassword123!",
        )




def test_login_user_creates_session(
    mocker,
):
    user = mocker.Mock(
        id=1,
        email="user@example.com",
        password_hash="stored-hash",
        is_active=True,
    )

    mocker.patch.object(
        auth_service,
        "get_user_by_email",
        return_value=user,
    )

    mock_verify = mocker.patch.object(
        auth_service,
        "verify_password",
        return_value=True,
    )

    mocker.patch.object(
        auth_service,
        "generate_session_token",
        return_value="raw-session-token",
    )

    mocker.patch.object(
        auth_service,
        "hash_session_token",
        return_value="a" * 64,
    )

    mock_create_session = mocker.patch.object(
        auth_service,
        "create_auth_session",
    )

    result_user, token = (
        auth_service.login_user(
            email=" USER@Example.COM ",
            password="StrongPassword123!",
        )
    )

    assert result_user is user
    assert token == "raw-session-token"

    auth_service.get_user_by_email.assert_called_once_with(
        "user@example.com"
    )

    mock_verify.assert_called_once_with(
        "StrongPassword123!",
        "stored-hash",
    )

    mock_create_session.assert_called_once()

    call_kwargs = (
        mock_create_session.call_args.kwargs
    )

    assert call_kwargs["user_id"] == 1
    assert call_kwargs["token_hash"] == "a" * 64




def test_login_user_rejects_unknown_user(
    mocker,
):
    mocker.patch.object(
        auth_service,
        "get_user_by_email",
        return_value=None,
    )

    with pytest.raises(
        InvalidCredentialsError
    ):
        auth_service.login_user(
            email="missing@example.com",
            password="StrongPassword123!",
        )



def test_login_user_rejects_wrong_password(
    mocker,
):
    user = mocker.Mock(
        password_hash="stored-hash",
        is_active=True,
    )

    mocker.patch.object(
        auth_service,
        "get_user_by_email",
        return_value=user,
    )

    mocker.patch.object(
        auth_service,
        "verify_password",
        return_value=False,
    )

    with pytest.raises(
        InvalidCredentialsError
    ):
        auth_service.login_user(
            email="user@example.com",
            password="WrongPassword123!",
        )




def test_get_user_from_session_token_returns_user(
    mocker,
):
    session = mocker.Mock(
        user_id=1,
        expires_at=(
            auth_service.datetime.utcnow()
            + auth_service.timedelta(hours=1)
        ),
    )

    user = mocker.Mock(
        id=1,
        is_active=True,
    )

    mocker.patch.object(
        auth_service,
        "hash_session_token",
        return_value="a" * 64,
    )

    mocker.patch.object(
        auth_service,
        "get_auth_session_by_token_hash",
        return_value=session,
    )

    mocker.patch.object(
        auth_service,
        "get_user_by_id",
        return_value=user,
    )

    result = (
        auth_service
        .get_user_from_session_token(
            "raw-token"
        )
    )

    assert result is user




def test_get_user_from_session_token_rejects_missing_session(
    mocker,
):
    mocker.patch.object(
        auth_service,
        "hash_session_token",
        return_value="a" * 64,
    )

    mocker.patch.object(
        auth_service,
        "get_auth_session_by_token_hash",
        return_value=None,
    )

    with pytest.raises(
        AuthenticationRequiredError
    ):
        auth_service.get_user_from_session_token(
            "invalid-token"
        )




def test_get_user_from_session_token_rejects_expired_session(
    mocker,
):
    session = mocker.Mock(
        expires_at=(
            auth_service.datetime.utcnow()
            - auth_service.timedelta(hours=1)
        ),
    )

    mocker.patch.object(
        auth_service,
        "hash_session_token",
        return_value="b" * 64,
    )

    mocker.patch.object(
        auth_service,
        "get_auth_session_by_token_hash",
        return_value=session,
    )

    mock_delete = mocker.patch.object(
        auth_service,
        "delete_auth_session",
    )

    with pytest.raises(
        AuthenticationRequiredError
    ):
        auth_service.get_user_from_session_token(
            "expired-token"
        )

    mock_delete.assert_called_once_with(
        "b" * 64
    )




def test_logout_user_deletes_hashed_session(
    mocker,
):
    mocker.patch.object(
        auth_service,
        "hash_session_token",
        return_value="c" * 64,
    )

    mock_delete = mocker.patch.object(
        auth_service,
        "delete_auth_session",
    )

    auth_service.logout_user(
        "raw-token"
    )

    mock_delete.assert_called_once_with(
        "c" * 64
    )