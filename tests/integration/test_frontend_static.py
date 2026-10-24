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



def test_home_page_contains_auth_ui(
    api_client,
):
    response = api_client.get("/")

    assert response.status_code == 200

    html = response.text

    assert 'id="authScreen"' in html
    assert 'id="authForm"' in html
    assert 'id="authLoginTab"' in html
    assert 'id="authRegisterTab"' in html
    assert 'id="logoutBtn"' in html
    assert "mobile-logout-btn" in html




def test_javascript_contains_auth_endpoints(
    api_client,
):
    response = api_client.get(
        "/static/js/app.js"
    )

    assert response.status_code == 200

    javascript = response.text

    assert '"/auth/me"' in javascript
    assert '"/auth/logout"' in javascript
    assert "`/auth/${authMode}`" in javascript




def test_javascript_initializes_auth_before_chat(
    api_client,
):
    response = api_client.get(
        "/static/js/app.js"
    )

    assert response.status_code == 200

    javascript = response.text

    assert (
        "async function initializeApp()"
        in javascript
    )

    assert (
        "await getCurrentUser()"
        in javascript
    )

    assert (
        "initializeApp();"
        in javascript
    )




def test_javascript_uses_user_specific_thread_storage(
    api_client,
):
    response = api_client.get(
        "/static/js/app.js"
    )

    assert response.status_code == 200

    javascript = response.text

    assert (
        "thread_id_user_${currentUser.id}"
        in javascript
    )

    assert (
        '"thread_id"'
        in javascript
    )




def test_css_contains_mobile_auth_layout(
    api_client,
):
    response = api_client.get(
        "/static/css/app.css"
    )

    assert response.status_code == 200

    css = response.text

    assert ".auth-screen" in css
    assert ".sidebar-user" in css
    assert ".mobile-logout-btn" in css

    assert (
        "@media (max-width: 768px)"
        in css
    )

    assert (
        "grid-template-columns:"
        in css
    )