"""Flask serving the built SPA: index for its routes, hashed assets, the CSP."""

import app as app_module

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
    "form-action 'self'"
)


def test_index_serves_the_spa_with_a_strict_csp(client, web_dist):
    response = client.get("/")
    assert response.status_code == 200
    assert b'<div id="root">' in response.data
    assert response.headers["Content-Security-Policy"] == CSP
    assert response.headers["Cache-Control"] == "no-cache"


def test_viewer_route_serves_the_same_index(client, web_dist):
    response = client.get("/p/any-name")
    assert response.status_code == 200
    assert b'<div id="root">' in response.data


def test_about_route_serves_the_same_index(client, web_dist):
    response = client.get("/about")
    assert response.status_code == 200
    assert b'<div id="root">' in response.data


def test_add_route_is_gone(client, web_dist):
    assert client.get("/add").status_code == 404


def test_assets_are_served_and_cached_for_good(client, web_dist):
    response = client.get("/assets/app-abc123.js")
    assert response.status_code == 200
    assert "immutable" in response.headers["Cache-Control"]


def test_asset_traversal_is_refused(client, web_dist):
    assert client.get("/assets/..%2Findex.html").status_code == 404
    assert client.get("/assets/missing.js").status_code == 404


def test_index_without_build_is_503(client, tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "WEB_DIST", tmp_path / "nothing")
    response = client.get("/")
    assert response.status_code == 503
    assert "task web:build" in response.get_json()["error"]
    assert client.get("/config").status_code == 200


def test_no_jsdelivr_anywhere(client, web_dist):
    assert "jsdelivr" not in client.get("/").headers["Content-Security-Policy"]
