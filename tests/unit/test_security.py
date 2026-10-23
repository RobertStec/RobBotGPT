from core.security import (
    generate_session_token,
    hash_password,
    hash_session_token,
    verify_password,
)




def test_hash_password_does_not_store_plaintext():
    password = "StrongPassword123!"

    password_hash = hash_password(
        password
    )

    assert password_hash != password




def test_verify_password_accepts_correct_password():
    password = "StrongPassword123!"

    password_hash = hash_password(
        password
    )

    assert verify_password(
        password,
        password_hash,
    ) is True




def test_verify_password_rejects_wrong_password():
    password_hash = hash_password(
        "StrongPassword123!"
    )

    assert verify_password(
        "WrongPassword123!",
        password_hash,
    ) is False




def test_generate_session_token_returns_unique_tokens():
    token_one = generate_session_token()
    token_two = generate_session_token()

    assert token_one
    assert token_two
    assert token_one != token_two




def test_hash_session_token_is_deterministic():
    token = "test-session-token"

    first_hash = hash_session_token(
        token
    )

    second_hash = hash_session_token(
        token
    )

    assert first_hash == second_hash




def test_hash_session_token_returns_sha256_hex():
    token_hash = hash_session_token(
        "test-session-token"
    )

    assert len(token_hash) == 64