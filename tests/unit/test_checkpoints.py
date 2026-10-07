import checkpoints


def test_create_checkpoint_resource(
    tmp_path,
    mocker,
):
    checkpoint_path = (
        tmp_path
        / "checkpoints.sqlite"
    )

    mock_connection = mocker.Mock()

    mock_connect = mocker.patch(
        "checkpoints.sqlite3.connect",
        return_value=mock_connection,
    )

    mock_checkpointer = mocker.Mock()

    mock_saver = mocker.patch(
        "checkpoints.SqliteSaver",
        return_value=mock_checkpointer,
    )

    resource = (
        checkpoints.create_checkpoint_resource(
            checkpoint_path
        )
    )

    mock_connect.assert_called_once_with(
        str(checkpoint_path.resolve()),
        check_same_thread=False,
    )

    mock_saver.assert_called_once_with(
        mock_connection
    )

    assert (
        resource.connection
        is mock_connection
    )

    assert (
        resource.checkpointer
        is mock_checkpointer
    )




def test_get_checkpointer_caches_resource(
    tmp_path,
    mocker,
):
    checkpoints.clear_checkpoint_resource_cache()

    checkpoint_path = (
        tmp_path
        / "cached.sqlite"
    )

    mock_resource = mocker.Mock()

    mock_create = mocker.patch(
        "checkpoints.create_checkpoint_resource",
        return_value=mock_resource,
    )

    first = checkpoints.get_checkpoint_resource(
        checkpoint_path
    )

    second = checkpoints.get_checkpoint_resource(
        checkpoint_path
    )

    assert first is mock_resource
    assert second is mock_resource

    mock_create.assert_called_once()

    checkpoints.clear_checkpoint_resource_cache()




def test_checkpoint_cache_is_separated_by_path(
    tmp_path,
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
        tmp_path / "first.sqlite"
    )

    second = checkpoints.get_checkpoint_resource(
        tmp_path / "second.sqlite"
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





def test_clear_checkpoint_resource_cache_closes_connections(
    tmp_path,
    mocker,
):
    checkpoints.clear_checkpoint_resource_cache()

    mock_connection = mocker.Mock()
    mock_checkpointer = mocker.Mock()

    resource = checkpoints.CheckpointResource(
        connection=mock_connection,
        checkpointer=mock_checkpointer,
    )

    mocker.patch(
        "checkpoints.create_checkpoint_resource",
        return_value=resource,
    )

    checkpoints.get_checkpoint_resource(
        tmp_path / "cleanup.sqlite"
    )

    checkpoints.clear_checkpoint_resource_cache()

    mock_connection.close.assert_called_once()