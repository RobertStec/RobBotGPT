from fastapi import Request

from core.config import settings
from core.exceptions import (
    AuthenticationRequiredError,
)

from services.auth_service import (
    get_user_from_session_token,
)




def get_current_user(
    request: Request,
):
    token = request.cookies.get(
        settings.auth_cookie_name
    )

    if not token:
        raise AuthenticationRequiredError()

    return get_user_from_session_token(
        token
    )