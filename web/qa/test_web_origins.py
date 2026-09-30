"""显式局域网隧道来源必须精确匹配，不放开任意 Host/Origin。"""

import pytest
from fastapi.testclient import TestClient

from clsweb.app import create_app


def test_lan_tunnel_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("CLS_WEB_ORIGINS", "http://192.168.0.42:5173")
    with TestClient(create_app(tmp_path)) as client:
        headers = {"Host": "192.168.0.42:5173", "Origin": "http://192.168.0.42:5173"}
        assert client.get("/api/health", headers=headers).status_code == 200
        settings = client.get("/api/settings", headers=headers).json()
        assert client.post("/api/settings", headers=headers, json=settings).status_code == 200
        headers["Origin"] = "http://192.168.0.42:5174"
        assert client.post("/api/settings", headers=headers, json=settings).status_code == 403
        assert client.get("/api/health", headers={"Host": "192.168.0.43"}).status_code == 400


def test_default_rejects_unconfigured_lan(tmp_path, monkeypatch):
    monkeypatch.delenv("CLS_WEB_ORIGINS", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/health", headers={"Host": "192.168.0.42"}).status_code == 400


@pytest.mark.parametrize("origin", [
    "*", "http://0.0.0.0:5173", "http://8.8.8.8:5173", "http://192.168.0.42",
    "http://192.168.0.42:70000", "http://user@192.168.0.42:5173",
    "http://192.168.0.42:5173/", "http://192.168.0.42:5173,", "https://192.168.0.42:5173",
])
def test_invalid_web_origins_rejected(tmp_path, monkeypatch, origin):
    monkeypatch.setenv("CLS_WEB_ORIGINS", origin)
    with pytest.raises(ValueError, match="CLS_WEB_ORIGINS"):
        create_app(tmp_path)
