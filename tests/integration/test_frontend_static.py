def test_home_page_loads(
    api_client,
):
    response = api_client.get("/")

    assert response.status_code == 200

    assert (
        "/static/css/app.css"
        in response.text
    )

    assert (
        "/static/js/app.js"
        in response.text
    )


def test_css_static_file_is_available(
    api_client,
):
    response = api_client.get(
        "/static/css/app.css"
    )

    assert response.status_code == 200

    assert response.text.strip()


def test_javascript_static_file_is_available(
    api_client,
):
    response = api_client.get(
        "/static/js/app.js"
    )

    assert response.status_code == 200

    assert response.text.strip()