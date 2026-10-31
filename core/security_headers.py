from starlette.datastructures import (
    MutableHeaders,
)


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": (
        "camera=(), "
        "geolocation=(), "
        "payment=(), "
        "usb=()"
    ),
    "Content-Security-Policy": (
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "form-action 'self'"
    ),
}


class SecurityHeadersMiddleware:
    def __init__(
        self,
        app,
    ):
        self.app = app

    async def __call__(
        self,
        scope,
        receive,
        send,
    ):
        if scope["type"] != "http":
            await self.app(
                scope,
                receive,
                send,
            )
            return

        async def send_with_headers(
            message,
        ):
            if (
                message["type"]
                == "http.response.start"
            ):
                headers = MutableHeaders(
                    scope=message
                )

                for (
                    name,
                    value,
                ) in SECURITY_HEADERS.items():
                    headers[name] = value

            await send(message)

        await self.app(
            scope,
            receive,
            send_with_headers,
        )