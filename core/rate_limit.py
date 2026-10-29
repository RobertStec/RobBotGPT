import logging
from dataclasses import dataclass

import redis.asyncio as redis

from core.config import settings

from fastapi import Request

from core.exceptions import (
    RateLimitExceededError,
)



logger = logging.getLogger(__name__)


# =========================================================
# Redis resource
# =========================================================

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    Return the shared asynchronous Redis client.

    The connection pool is created lazily and reused
    across requests handled by this process.
    """

    global _redis_client

    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )

    return _redis_client



async def initialize_rate_limit_resources() -> None:
    """
    Verify that Redis is available during
    application startup.
    """

    client = get_redis_client()

    await client.ping()



async def clear_rate_limit_resource_cache() -> None:
    """
    Close the cached Redis client and release
    its connection pool.
    """

    global _redis_client

    if _redis_client is None:
        return

    await _redis_client.aclose()

    _redis_client = None


# =========================================================
# Rate limit result
# =========================================================

@dataclass(
    frozen=True
)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int



def get_client_ip(
    request: Request,
) -> str:
    """
    Resolve the client IP address.

    In production Nginx should overwrite
    X-Real-IP with the real remote address.
    """

    real_ip = request.headers.get(
        "x-real-ip"
    )

    if real_ip:
        real_ip = real_ip.strip()

        if real_ip:
            return real_ip

    if request.client is not None:
        return request.client.host

    return "unknown"



def build_rate_limit_identifier(
    *,
    request: Request,
    user_id: int | None = None,
) -> str:
    """
    Use stable user ownership when authenticated,
    otherwise fall back to the client IP.
    """

    if user_id is not None:
        if (
            not isinstance(user_id, int)
            or isinstance(user_id, bool)
            or user_id < 1
        ):
            raise ValueError(
                "user_id must be a positive integer."
            )

        return f"user:{user_id}"

    return (
        f"ip:{get_client_ip(request)}"
    )




async def enforce_rate_limit(
    *,
    scope: str,
    identifier: str,
    limit: int,
    window_seconds: int,
) -> RateLimitResult:
    """
    Check a rate limit and raise a safe
    HTTP-facing exception when exceeded.
    """

    result = await check_rate_limit(
        scope=scope,
        identifier=identifier,
        limit=limit,
        window_seconds=window_seconds,
    )

    if not result.allowed:
        raise RateLimitExceededError(
            retry_after=result.retry_after
        )

    return result


# =========================================================
# Atomic Redis operation
# =========================================================

_RATE_LIMIT_SCRIPT = """
local current = redis.call("INCR", KEYS[1])

if current == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end

local ttl = redis.call("TTL", KEYS[1])

if ttl < 0 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
    ttl = tonumber(ARGV[1])
end

return {current, ttl}
"""


async def check_rate_limit(
    *,
    scope: str,
    identifier: str,
    limit: int,
    window_seconds: int,
) -> RateLimitResult:
    """
    Increment and evaluate one fixed-window rate limit.

    The Redis Lua script makes INCR + EXPIRE atomic,
    preventing concurrent requests from creating
    inconsistent counters.
    """

    scope = scope.strip()
    identifier = identifier.strip()

    if not scope:
        raise ValueError(
            "scope cannot be empty."
        )

    if not identifier:
        raise ValueError(
            "identifier cannot be empty."
        )

    if limit < 1:
        raise ValueError(
            "limit must be greater than 0."
        )

    if window_seconds < 1:
        raise ValueError(
            "window_seconds must be greater than 0."
        )

    key = (
        f"rate_limit:"
        f"{scope}:"
        f"{identifier}"
    )

    client = get_redis_client()

    current, ttl = await client.eval(
        _RATE_LIMIT_SCRIPT,
        1,
        key,
        window_seconds,
    )

    current = int(current)
    ttl = max(
        int(ttl),
        1,
    )

    allowed = (
        current <= limit
    )

    remaining = max(
        limit - current,
        0,
    )

    retry_after = (
        0
        if allowed
        else ttl
    )

    return RateLimitResult(
        allowed=allowed,
        limit=limit,
        remaining=remaining,
        retry_after=retry_after,
    )