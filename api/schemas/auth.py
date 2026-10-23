from pydantic import (
    BaseModel,
    EmailStr,
    Field,
)


class RegisterRequest(BaseModel):
    email: EmailStr

    password: str = Field(
        min_length=12,
        max_length=128,
    )


class UserResponse(BaseModel):
    id: int
    email: EmailStr


class RegisterResponse(BaseModel):
    success: bool
    user: UserResponse


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    success: bool
    user: UserResponse