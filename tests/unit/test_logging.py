import core.logging as app_logging


def test_configure_logging_uses_configured_level(
    mocker,
    monkeypatch,
):
    monkeypatch.setattr(
        app_logging.settings,
        "log_level",
        "DEBUG",
    )

    mock_dict_config = mocker.patch(
        "core.logging.dictConfig"
    )

    app_logging.configure_logging()

    mock_dict_config.assert_called_once()

    config = (
        mock_dict_config
        .call_args
        .args[0]
    )

    assert (
        config["root"]["level"]
        == "DEBUG"
    )

    assert (
        config["handlers"]["console"]["level"]
        == "DEBUG"
    )