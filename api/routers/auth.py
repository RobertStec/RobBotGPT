from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response,
    status,
)

from api.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    UserResponse,
)

from api.dependencies.auth import (
    get_current_user,
)

from core.config import settings

import core.rate_limit as rate_limit

from services.auth_service import (
    login_user,
    register_user,
    logout_user,
)





router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    request: Request,
):
    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="auth_register",
        identifier=identifier,
        limit=5,
        window_seconds=3600,
    )

    user = register_user(
        email=str(payload.email),
        password=payload.password,
    )

    return RegisterResponse(
        success=True,
        user=UserResponse(
            id=user.id,
            email=user.email,
        ),
    )


@router.post(
    "/login",
    response_model=LoginResponse,
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
):
    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="auth_login",
        identifier=identifier,
        limit=10,
        window_seconds=300,
    )

    user, token = login_user(
        email=str(payload.email),
        password=payload.password,
    )

    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=(
            settings.auth_session_ttl_hours
            * 60
            * 60
        ),
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        path="/",
    )

    return LoginResponse(
        success=True,
        user=UserResponse(
            id=user.id,
            email=user.email,
        ),
    )



@router.get(
    "/me",
    response_model=UserResponse,
)

async def me(
    request: Request,
    current_user=Depends(
        get_current_user
    ),
):
    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request,
            user_id=current_user.id,
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="auth_me",
        identifier=identifier,
        limit=120,
        window_seconds=60,
    )

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
    )

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
    )




@router.post(
    "/logout",
)
async def logout(
    request: Request,
    response: Response,
):
    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="auth_logout",
        identifier=identifier,
        limit=30,
        window_seconds=60,
    )

    token = request.cookies.get(
        settings.auth_cookie_name
    )

    if token:
        logout_user(
            token
        )

    response.delete_cookie(
        key=settings.auth_cookie_name,
        path="/",
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite=settings.auth_cookie_samesite,
    )

    return {
        "success": True
    }