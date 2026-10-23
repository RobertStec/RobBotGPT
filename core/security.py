import hashlib
import secrets

from pwdlib import PasswordHash


_password_hash = PasswordHash.recommended()



def hash_password(password: str) -> str:
    """
    Hash a plaintext password using the recommended
    password hashing algorithm.
    """

    return _password_hash.hash(password)



def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    """
    Verify a plaintext password against a stored hash.
    """

    return _password_hash.verify(
        password,
        password_hash,
    )



def generate_session_token() -> str:
    """
    Generate a cryptographically secure random
    session token.
    """

    return secrets.token_urlsafe(32)



def hash_session_token(
    token: str,
) -> str:
    """
    Hash a session token before storing it
    in the database.
    """

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()