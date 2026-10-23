from sqlalchemy.exc import IntegrityError

from datetime import (
    datetime,
    timedelta,
)

from core.config import settings

from core.exceptions import (
    AuthenticationRequiredError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
)
from core.security import (
    generate_session_token,
    hash_password,
    hash_session_token,
    verify_password,
)

from database import (
    create_auth_session,
    create_user,
    get_user_by_email,
    delete_auth_session,
    get_auth_session_by_token_hash,
    get_user_by_id,
)




def register_user(
    email: str,
    password: str,
):
    normalized_email = (
        email.strip().lower()
    )

    existing_user = get_user_by_email(
        normalized_email
    )

    if existing_user is not None:
        raise UserAlreadyExistsError()

    password_hash = hash_password(
        password
    )

    try:
        return create_user(
            email=normalized_email,
            password_hash=password_hash,
        )

    except IntegrityError as exc:
        raise UserAlreadyExistsError(
            internal_message=str(exc)
        ) from exc




def login_user(
    email: str,
    password: str,
):
    normalized_email = (
        email.strip().lower()
    )

    user = get_user_by_email(
        normalized_email
    )

    if (
        user is None
        or not user.is_active
        or not verify_password(
            password,
            user.password_hash,
        )
    ):
        raise InvalidCredentialsError()

    token = generate_session_token()

    token_hash = hash_session_token(
        token
    )

    expires_at = (
        datetime.utcnow()
        + timedelta(
            hours=settings.auth_session_ttl_hours
        )
    )

    create_auth_session(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )

    return user, token




def get_user_from_session_token(
    token: str,
):
    token_hash = hash_session_token(
        token
    )

    auth_session = (
        get_auth_session_by_token_hash(
            token_hash
        )
    )

    if auth_session is None:
        raise AuthenticationRequiredError()

    if (
        auth_session.expires_at
        <= datetime.utcnow()
    ):
        delete_auth_session(
            token_hash
        )

        raise AuthenticationRequiredError()

    user = get_user_by_id(
        auth_session.user_id
    )

    if (
        user is None
        or not user.is_active
    ):
        delete_auth_session(
            token_hash
        )

        raise AuthenticationRequiredError()

    return user




def logout_user(
    token: str,
) -> None:
    token_hash = hash_session_token(
        token
    )

    delete_auth_session(
        token_hash
    )