from typing import Annotated

from pydantic import StringConstraints


MAX_THREAD_ID_LENGTH = 128

THREAD_ID_PATTERN = (
    r"^[A-Za-z0-9._:-]+$"
)


ThreadId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_THREAD_ID_LENGTH,
        pattern=THREAD_ID_PATTERN,
    ),
]