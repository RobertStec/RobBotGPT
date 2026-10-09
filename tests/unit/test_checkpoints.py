import pytest
import checkpoints




def test_create_checkpoint_resource(
    mocker,
):
    database_url = (
        "postgresql+psycopg://"
        "user:password@localhost:5433/test_db"
    )

    mock_pool = mocker.Mock()

    mock_pool_class = mocker.patch(
        "checkpoints.ConnectionPool",
        return_value=mock_pool,
    )

    mock_checkpointer = mocker.Mock()

    mock_saver = mocker.patch(
        "checkpoints.PostgresSaver",
        return_value=mock_checkpointer,
    )

    resource = (
        checkpoints.create_checkpoint_resource(
            database_url
        )
    )

    mock_pool_class.assert_called_once()

    call_kwargs = (
        mock_pool_class.call_args.kwargs
    )

    assert (
        call_kwargs["conninfo"]
        ==
        "postgresql://"
        "user:password@localhost:5433/test_db"
    )

    assert call_kwargs["min_size"] == 1
    assert call_kwargs["max_size"] == 5
    assert call_kwargs["open"] is True

    assert (
        call_kwargs["kwargs"]["autocommit"]
        is True
    )

    assert (
        call_kwargs["kwargs"][
            "prepare_threshold"
        ]
        == 0
    )

    assert (
        call_kwargs["kwargs"][
            "row_factory"
        ]
        is checkpoints.dict_row
    )

    mock_saver.assert_called_once_with(
        mock_pool
    )

    mock_checkpointer.setup.assert_called_once_with()

    assert resource.pool is mock_pool

    assert (
        resource.checkpointer
        is mock_checkpointer
    )




def test_get_checkpointer_caches_resource(
    mocker,
):
    checkpoints.clear_checkpoint_resource_cache()

    database_url = (
        "postgresql://"
        "user:pass@localhost/test_db"
    )

    mock_resource = mocker.Mock()

    mock_create = mocker.patch(
        "checkpoints.create_checkpoint_resource",
        return_value=mock_resource,
    )

    first = checkpoints.get_checkpoint_resource(
        database_url
    )

    second = checkpoints.get_checkpoint_resource(
        database_url
    )

    assert first is mock_resource
    assert second is mock_resource

    mock_create.assert_called_once_with(
        database_url
    )

    checkpoints.clear_checkpoint_resource_cache()




def test_checkpoint_cache_is_separated_by_database_url(
    mocker,
):
    checkpoints.clear_checkpoint_resource_cache()

    first_resource = mocker.Mock()
    second_resource = mocker.Mock()

    mocker.patch(
        "checkpoints.create_checkpoint_resource",
        side_effect=[
            first_resource,
            second_resource,
        ],
    )

    first = checkpoints.get_checkpoint_resource(
        "postgresql://user:pass@localhost/db1"
    )

    second = checkpoints.get_checkpoint_resource(
        "postgresql://user:pass@localhost/db2"
    )

    assert first is first_resource
    assert second is second_resource

    checkpoints.clear_checkpoint_resource_cache()




def test_delete_thread_checkpoints(
    mocker,
):
    mock_checkpointer = mocker.Mock()

    mocker.patch(
        "checkpoints.get_checkpointer",
        return_value=mock_checkpointer,
    )

    checkpoints.delete_thread_checkpoints(
        "thread-123"
    )

    mock_checkpointer.delete_thread.assert_called_once_with(
        "thread-123"
    )





def test_clear_checkpoint_resource_cache_closes_pools(
    mocker,
):
    checkpoints.clear_checkpoint_resource_cache()

    mock_pool = mocker.Mock()
    mock_checkpointer = mocker.Mock()

    resource = checkpoints.CheckpointResource(
        pool=mock_pool,
        checkpointer=mock_checkpointer,
    )

    mocker.patch(
        "checkpoints.create_checkpoint_resource",
        return_value=resource,
    )

    checkpoints.get_checkpoint_resource(
        "postgresql://user:pass@localhost/test_db"
    )

    checkpoints.clear_checkpoint_resource_cache()

    mock_pool.close.assert_called_once_with()





def test_create_checkpoint_resource_closes_pool_when_setup_fails(
    mocker,
):
    mock_pool = mocker.Mock()

    mocker.patch(
        "checkpoints.ConnectionPool",
        return_value=mock_pool,
    )

    mock_checkpointer = mocker.Mock()

    mock_checkpointer.setup.side_effect = (
        RuntimeError("setup failed")
    )

    mocker.patch(
        "checkpoints.PostgresSaver",
        return_value=mock_checkpointer,
    )

    with pytest.raises(
        RuntimeError,
        match="setup failed",
    ):
        checkpoints.create_checkpoint_resource(
            "postgresql://user:pass@localhost/test_db"
        )

    mock_pool.close.assert_called_once_with()