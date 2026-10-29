import pytest

import core.rate_limit as rate_limit


def test_get_redis_client_reuses_cached_client(
    monkeypatch,
    mocker,
):
    fake_client = mocker.Mock()

    mock_from_url = mocker.patch(
        "core.rate_limit.redis.from_url",
        return_value=fake_client,
    )

    monkeypatch.setattr(
        rate_limit,
        "_redis_client",
        None,
    )

    monkeypatch.setattr(
        rate_limit.settings,
        "redis_url",
        "redis://test-redis:6379/1",
    )

    first = (
        rate_limit.get_redis_client()
    )

    second = (
        rate_limit.get_redis_client()
    )

    assert first is fake_client
    assert second is fake_client

    mock_from_url.assert_called_once_with(
        "redis://test-redis:6379/1",
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )


@pytest.mark.asyncio
async def test_clear_rate_limit_resource_cache(
    monkeypatch,
    mocker,
):
    fake_client = mocker.Mock()

    fake_client.aclose = (
        mocker.AsyncMock()
    )

    monkeypatch.setattr(
        rate_limit,
        "_redis_client",
        fake_client,
    )

    await (
        rate_limit
        .clear_rate_limit_resource_cache()
    )

    fake_client.aclose.assert_awaited_once_with()

    assert (
        rate_limit._redis_client
        is None
    )


@pytest.mark.asyncio
async def test_check_rate_limit_allows_request(
    mocker,
):
    client = mocker.Mock()

    client.eval = mocker.AsyncMock(
        return_value=[
            1,
            60,
        ]
    )

    mocker.patch(
        "core.rate_limit.get_redis_client",
        return_value=client,
    )

    result = await rate_limit.check_rate_limit(
        scope="chat",
        identifier="user:101",
        limit=20,
        window_seconds=60,
    )

    assert result.allowed is True
    assert result.limit == 20
    assert result.remaining == 19
    assert result.retry_after == 0


@pytest.mark.asyncio
async def test_check_rate_limit_blocks_request(
    mocker,
):
    client = mocker.Mock()

    client.eval = mocker.AsyncMock(
        return_value=[
            21,
            42,
        ]
    )

    mocker.patch(
        "core.rate_limit.get_redis_client",
        return_value=client,
    )

    result = await rate_limit.check_rate_limit(
        scope="chat",
        identifier="user:101",
        limit=20,
        window_seconds=60,
    )

    assert result.allowed is False
    assert result.limit == 20
    assert result.remaining == 0
    assert result.retry_after == 42




@pytest.mark.asyncio
async def test_check_rate_limit_uses_expected_key(
    mocker,
):
    client = mocker.Mock()

    client.eval = mocker.AsyncMock(
        return_value=[
            1,
            300,
        ]
    )

    mocker.patch(
        "core.rate_limit.get_redis_client",
        return_value=client,
    )

    await rate_limit.check_rate_limit(
        scope="login",
        identifier="127.0.0.1",
        limit=10,
        window_seconds=300,
    )

    call_args = (
        client.eval.await_args.args
    )

    assert call_args[1] == 1

    assert (
        call_args[2]
        == (
            "rate_limit:"
            "login:"
            "127.0.0.1"
        )
    )

    assert call_args[3] == 300



@pytest.mark.asyncio
async def test_initialize_rate_limit_resources(
    mocker,
):
    fake_client = mocker.Mock()

    fake_client.ping = (
        mocker.AsyncMock(
            return_value=True
        )
    )

    mocker.patch(
        "core.rate_limit.get_redis_client",
        return_value=fake_client,
    )

    await (
        rate_limit
        .initialize_rate_limit_resources()
    )

    fake_client.ping.assert_awaited_once_with()




@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "kwargs",
        "message",
    ),
    [
        (
            {
                "scope": "",
                "identifier": "user:1",
                "limit": 10,
                "window_seconds": 60,
            },
            "scope cannot be empty",
        ),
        (
            {
                "scope": "chat",
                "identifier": "",
                "limit": 10,
                "window_seconds": 60,
            },
            "identifier cannot be empty",
        ),
        (
            {
                "scope": "chat",
                "identifier": "user:1",
                "limit": 0,
                "window_seconds": 60,
            },
            "limit must be greater than 0",
        ),
        (
            {
                "scope": "chat",
                "identifier": "user:1",
                "limit": 10,
                "window_seconds": 0,
            },
            (
                "window_seconds must be "
                "greater than 0"
            ),
        ),
    ],
)


async def test_check_rate_limit_rejects_invalid_values(
    kwargs,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        await rate_limit.check_rate_limit(
            **kwargs
        )




def test_get_client_ip_prefers_x_real_ip(
    mocker,
):
    request = mocker.Mock()

    request.headers = {
        "x-real-ip": "203.0.113.10"
    }

    request.client.host = "127.0.0.1"

    assert (
        rate_limit.get_client_ip(request)
        == "203.0.113.10"
    )




def test_get_client_ip_falls_back_to_request_client(
    mocker,
):
    request = mocker.Mock()

    request.headers = {}
    request.client.host = "192.0.2.10"

    assert (
        rate_limit.get_client_ip(request)
        == "192.0.2.10"
    )




def test_build_rate_limit_identifier_uses_user_id(
    mocker,
):
    request = mocker.Mock()

    result = (
        rate_limit
        .build_rate_limit_identifier(
            request=request,
            user_id=101,
        )
    )

    assert result == "user:101"




def test_build_rate_limit_identifier_uses_ip_when_anonymous(
    mocker,
):
    request = mocker.Mock()

    request.headers = {
        "x-real-ip": "203.0.113.25"
    }

    result = (
        rate_limit
        .build_rate_limit_identifier(
            request=request,
        )
    )

    assert (
        result
        == "ip:203.0.113.25"
    )




@pytest.mark.asyncio
async def test_enforce_rate_limit_returns_allowed_result(
    mocker,
):
    expected = rate_limit.RateLimitResult(
        allowed=True,
        limit=10,
        remaining=9,
        retry_after=0,
    )

    mocker.patch(
        "core.rate_limit.check_rate_limit",
        new=mocker.AsyncMock(
            return_value=expected
        ),
    )

    result = await rate_limit.enforce_rate_limit(
        scope="login",
        identifier="ip:127.0.0.1",
        limit=10,
        window_seconds=300,
    )

    assert result == expected




@pytest.mark.asyncio
async def test_enforce_rate_limit_raises_when_blocked(
    mocker,
):
    blocked = rate_limit.RateLimitResult(
        allowed=False,
        limit=10,
        remaining=0,
        retry_after=37,
    )

    mocker.patch(
        "core.rate_limit.check_rate_limit",
        new=mocker.AsyncMock(
            return_value=blocked
        ),
    )

    with pytest.raises(
        rate_limit.RateLimitExceededError
    ) as exc_info:
        await rate_limit.enforce_rate_limit(
            scope="login",
            identifier="ip:127.0.0.1",
            limit=10,
            window_seconds=300,
        )

    assert (
        exc_info.value.retry_after
        == 37
    )