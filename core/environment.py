import os
from pathlib import Path

import certifi
from dotenv import load_dotenv


def configure_environment() -> None:
    load_dotenv()

    certificate_path = Path(
        certifi.where()
    ).resolve()

    if not certificate_path.is_file():
        raise RuntimeError(
            "Certifi certificate bundle "
            f"does not exist: {certificate_path}"
        )

    for variable_name in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    ):
        current_value = os.environ.get(
            variable_name
        )

        if (
            not current_value
            or not Path(
                current_value
            ).is_file()
        ):
            os.environ[
                variable_name
            ] = str(
                certificate_path
            )