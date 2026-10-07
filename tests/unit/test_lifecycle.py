import pytest

import core.lifecycle as lifecycle


def test_initialize_resources(
    tmp_path,
    monkeypatch,
    mocker,
):
    data_dir = (
        tmp_path
        / "data"
    )

    upload_dir = (
        tmp_path
        / "uploads"
    )

    monkeypatch.setattr(
        lifecycle.settings,
        "data_dir",
        data_dir,
    )

    monkeypatch.setattr(
        lifecycle.settings,
        "upload_dir",
        upload_dir,
    )

    mock_init_db = mocker.patch(
        "core.lifecycle.init_db"
    )

    mock_configure_logging = mocker.patch(
        "core.lifecycle.configure_logging"
    )


    lifecycle.initialize_resources()

    mock_configure_logging.assert_called_once_with()


    assert data_dir.exists()
    assert data_dir.is_dir()

    assert upload_dir.exists()
    assert upload_dir.is_dir()

    mock_init_db.assert_called_once_with()




# Test shutdownu

def test_shutdown_resources_cleans_resources_in_order(
    mocker,
):
    calls = []

    mocker.patch(
        "core.lifecycle.clear_agent_cache",
        side_effect=lambda: calls.append(
            "agent"
        ),
    )

    mocker.patch(
        "core.lifecycle.clear_checkpoint_resource_cache",
        side_effect=lambda: calls.append(
            "checkpoints"
        ),
    )

    mocker.patch(
        "core.lifecycle.clear_rag_resource_cache",
        side_effect=lambda: calls.append(
            "rag"
        ),
    )

    mocker.patch(
        "core.lifecycle.clear_database_resource_cache",
        side_effect=lambda: calls.append(
            "database"
        ),
    )

    lifecycle.shutdown_resources()

    assert calls == [
        "agent",
        "checkpoints",
        "rag",
        "database",
    ]




# Test FastAPI lifespan

@pytest.mark.asyncio
async def test_lifespan_initializes_and_shuts_down(
    mocker,
):
    mock_initialize = mocker.patch(
        "core.lifecycle.initialize_resources"
    )

    mock_shutdown = mocker.patch(
        "core.lifecycle.shutdown_resources"
    )

    fake_app = mocker.Mock()

    async with lifecycle.lifespan(
        fake_app
    ):
        mock_initialize.assert_called_once_with()

        mock_shutdown.assert_not_called()

    mock_shutdown.assert_called_once_with()




# Test shutdown również po wyjątku

@pytest.mark.asyncio
async def test_lifespan_shuts_down_after_exception(
    mocker,
):
    mocker.patch(
        "core.lifecycle.initialize_resources"
    )

    mock_shutdown = mocker.patch(
        "core.lifecycle.shutdown_resources"
    )

    fake_app = mocker.Mock()

    with pytest.raises(
        RuntimeError,
        match="application error",
    ):
        async with lifecycle.lifespan(
            fake_app
        ):
            raise RuntimeError(
                "application error"
            )

    mock_shutdown.assert_called_once_with()
