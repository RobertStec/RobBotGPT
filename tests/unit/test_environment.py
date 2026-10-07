import os
from pathlib import Path

from core.environment import (
    configure_environment,
)


def test_configure_environment_loads_dotenv_and_sets_certificates(
    mocker,
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(
        "SSL_CERT_FILE",
        raising=False,
    )

    monkeypatch.delenv(
        "REQUESTS_CA_BUNDLE",
        raising=False,
    )

    certificate_file = (
        tmp_path
        / "ca.pem"
    )

    certificate_file.write_text(
        "fake certificate",
        encoding="utf-8",
    )

    mock_load_dotenv = mocker.patch(
        "core.environment.load_dotenv"
    )

    mocker.patch(
        "core.environment.certifi.where",
        return_value=str(
            certificate_file
        ),
    )

    configure_environment()

    mock_load_dotenv.assert_called_once_with()

    assert (
        os.environ[
            "SSL_CERT_FILE"
        ]
        == str(
            certificate_file.resolve()
        )
    )

    assert (
        os.environ[
            "REQUESTS_CA_BUNDLE"
        ]
        == str(
            certificate_file.resolve()
        )
    )


def test_configure_environment_does_not_override_valid_existing_certificate_settings(
    mocker,
    monkeypatch,
    tmp_path,
):
    custom_ssl = (
        tmp_path
        / "custom-ca.pem"
    )

    custom_requests = (
        tmp_path
        / "custom-requests-ca.pem"
    )

    default_ca = (
        tmp_path
        / "default-ca.pem"
    )

    custom_ssl.write_text(
        "ssl",
        encoding="utf-8",
    )

    custom_requests.write_text(
        "requests",
        encoding="utf-8",
    )

    default_ca.write_text(
        "default",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "SSL_CERT_FILE",
        str(custom_ssl),
    )

    monkeypatch.setenv(
        "REQUESTS_CA_BUNDLE",
        str(custom_requests),
    )

    mocker.patch(
        "core.environment.load_dotenv"
    )

    mocker.patch(
        "core.environment.certifi.where",
        return_value=str(default_ca),
    )

    configure_environment()

    assert (
        os.environ[
            "SSL_CERT_FILE"
        ]
        == str(custom_ssl)
    )

    assert (
        os.environ[
            "REQUESTS_CA_BUNDLE"
        ]
        == str(custom_requests)
    )


def test_configure_environment_replaces_invalid_certificate_paths(
    monkeypatch,
):
    monkeypatch.setenv(
        "SSL_CERT_FILE",
        "missing-cert.pem",
    )

    monkeypatch.setenv(
        "REQUESTS_CA_BUNDLE",
        "missing-cert.pem",
    )

    configure_environment()

    assert Path(
        os.environ["SSL_CERT_FILE"]
    ).is_file()

    assert Path(
        os.environ[
            "REQUESTS_CA_BUNDLE"
        ]
    ).is_file()