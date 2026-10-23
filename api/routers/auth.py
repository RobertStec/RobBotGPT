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
):
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
    response: Response,
):
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
    current_user=Depends(
        get_current_user
    ),
):
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