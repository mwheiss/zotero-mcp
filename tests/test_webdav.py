"""Tests for direct WebDAV attachment access."""

import io
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from conftest import skip_on_ci

from zotero_mcp import client, webdav


class _FailingZotero:
    def dump(self, *_args, **_kwargs):
        raise RuntimeError("not available")


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.auth = None
        self.trust_env = True
        self.requested = []

    def get(self, url, timeout=None, stream=False):
        self.requested.append((url, timeout, stream))
        return self._response

    def close(self):
        return None


def _build_zip_bytes(name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
        zf.writestr(name, content)
    return buf.getvalue()


@skip_on_ci
def test_download_attachment_from_webdav_extracts_expected_file(tmp_path, monkeypatch):
    session = _FakeSession(_FakeResponse(_build_zip_bytes("paper.pdf", b"%PDF-1.4 webdav test")))
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "https://dav.example.com/zotero")
    monkeypatch.setenv("ZOTERO_WEBDAV_USERNAME", "alice")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASSWORD", "secret")
    monkeypatch.setattr("requests.Session", lambda: session)

    file_path = webdav.download_attachment_from_webdav("ABCD1234", tmp_path, expected_filename="paper.pdf")

    assert file_path == tmp_path / "paper.pdf"
    assert file_path.read_bytes() == b"%PDF-1.4 webdav test"
    assert session.auth == ("alice", "secret")
    assert session.requested == [("https://dav.example.com/zotero/ABCD1234.zip", (10.0, 30.0), True)]


@skip_on_ci
def test_download_attachment_from_webdav_escapes_attachment_key(tmp_path, monkeypatch):
    session = _FakeSession(_FakeResponse(_build_zip_bytes("paper.pdf", b"%PDF-1.4 webdav test")))
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "https://dav.example.com/zotero")
    monkeypatch.setenv("ZOTERO_WEBDAV_USERNAME", "alice")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASSWORD", "secret")
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.download_attachment_from_webdav("AB/CD", tmp_path, expected_filename="paper.pdf")

    assert session.requested == [("https://dav.example.com/zotero/AB%2FCD.zip", (10.0, 30.0), True)]


@skip_on_ci
def test_download_attachment_file_falls_back_to_webdav(tmp_path, monkeypatch):
    webdav_path = tmp_path / "nested" / "paper.pdf"
    webdav_path.parent.mkdir()
    webdav_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "zotero_mcp.client.download_attachment_from_webdav",
        lambda attachment_key, destination_dir, expected_filename=None: webdav_path,
    )

    result = client.download_attachment_file(
        "ABCD1234",
        tmp_path,
        "paper.pdf",
        local_client=_FailingZotero(),
        web_client=_FailingZotero(),
    )

    assert result.path == webdav_path
    assert result.source == "WebDAV"
    assert result.errors == [
        "Local Zotero: not available",
    ]


@skip_on_ci
def test_extract_archive_rejects_backslash_path_traversal(tmp_path):
    zip_bytes = _build_zip_bytes("..\\evil.txt", b"oops")

    try:
        webdav._extract_archive(zip_bytes, tmp_path, expected_filename="evil.txt")
    except ValueError as exc:
        assert "Unsafe path" in str(exc)
    else:
        raise AssertionError("Expected _extract_archive() to reject backslash traversal paths")


def test_extract_archive_prevalidates_all_paths_before_writing(tmp_path):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("safe.txt", b"must not be written")
        archive.writestr("../escape.txt", b"unsafe")

    with pytest.raises(ValueError, match="Unsafe path"):
        webdav._extract_archive(buffer.getvalue(), tmp_path, expected_filename=None)

    assert not (tmp_path / "safe.txt").exists()


def test_extract_archive_cleans_up_file_directory_prefix_collision(tmp_path):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("foo", b"first file")
        archive.writestr("foo/bar.txt", b"cannot coexist")

    with pytest.raises(ValueError, match="file/directory path collision"):
        webdav._extract_archive(buffer.getvalue(), tmp_path, expected_filename=None)

    assert not (tmp_path / "foo").exists()


def test_extract_archive_cleanup_tolerates_preexisting_directory(tmp_path):
    (tmp_path / "existingdir").mkdir()
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("safe.txt", b"must be cleaned")
        archive.writestr("existingdir", b"cannot replace a directory")

    with pytest.raises(IsADirectoryError):
        webdav._extract_archive(buffer.getvalue(), tmp_path, expected_filename=None)

    assert not (tmp_path / "safe.txt").exists()
    assert (tmp_path / "existingdir").is_dir()


def test_extract_archive_enforces_uncompressed_size_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_WEBDAV_MAX_UNCOMPRESSED_BYTES", "8")
    zip_bytes = _build_zip_bytes("paper.txt", b"nine-byte!")

    with pytest.raises(ValueError, match="uncompressed-size limit"):
        webdav._extract_archive(zip_bytes, tmp_path, expected_filename="paper.txt")


def test_extract_archive_enforces_compression_ratio_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_WEBDAV_MAX_COMPRESSION_RATIO", "2")
    zip_bytes = _build_zip_bytes("paper.txt", b"A" * 4096)

    with pytest.raises(ValueError, match="compression-ratio limit"):
        webdav._extract_archive(zip_bytes, tmp_path, expected_filename="paper.txt")


def test_extract_archive_enforces_member_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_WEBDAV_MAX_ARCHIVE_MEMBERS", "1")
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("one.txt", b"one")
        archive.writestr("two.txt", b"two")

    with pytest.raises(ValueError, match="member-count limit"):
        webdav._extract_archive(buffer.getvalue(), tmp_path, expected_filename=None)


def test_extract_archive_rejects_symbolic_links(tmp_path):
    buffer = io.BytesIO()
    link = ZipInfo("paper.txt")
    link.create_system = 3
    link.external_attr = 0o120777 << 16
    with ZipFile(buffer, "w") as archive:
        archive.writestr(link, "elsewhere")

    with pytest.raises(ValueError, match="symbolic link"):
        webdav._extract_archive(buffer.getvalue(), tmp_path, expected_filename=None)


def test_download_attachment_from_webdav_enforces_streamed_size_limit(
    tmp_path, monkeypatch
):
    payload = _build_zip_bytes("paper.pdf", b"PDF")
    session = _FakeSession(_FakeResponse(payload))
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "https://dav.example.com/zotero")
    monkeypatch.setenv("ZOTERO_WEBDAV_USERNAME", "alice")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASSWORD", "secret")
    monkeypatch.setenv("ZOTERO_MCP_WEBDAV_MAX_DOWNLOAD_BYTES", "8")
    monkeypatch.setattr("requests.Session", lambda: session)

    with pytest.raises(ValueError, match="download-size limit"):
        webdav.download_attachment_from_webdav("ABCD1234", tmp_path)


# ---------------------------------------------------------------------------
# upload_attachment_to_webdav
# ---------------------------------------------------------------------------


class _RecordingPutSession:
    """Captures PUT calls and returns a configurable response."""

    def __init__(self, status_code: int = 201):
        self.status_code = status_code
        self.auth = None
        self.trust_env = True
        self.calls = []  # list of (url, data, headers)
        self.timeouts = []  # parallel list of the timeout passed to each PUT

    def put(self, url, data=None, headers=None, timeout=None):
        self.calls.append((url, data, dict(headers or {})))
        self.timeouts.append(timeout)

        class _Resp:
            def __init__(self_inner, sc):
                self_inner.status_code = sc

            def raise_for_status(self_inner):
                if self_inner.status_code >= 400:
                    raise RuntimeError(f"http {self_inner.status_code}")

        return _Resp(self.status_code)

    def close(self):
        return None


def _setup_webdav_env(monkeypatch):
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "https://dav.example.com/zotero")
    monkeypatch.setenv("ZOTERO_WEBDAV_USERNAME", "alice")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASSWORD", "secret")


@skip_on_ci
def test_upload_attachment_puts_zip_then_prop(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"%PDF-1.4 hello")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: session)

    md5_hex, mtime_ms = webdav.upload_attachment_to_webdav("ABCD1234", src)

    # exactly two PUTs, .zip before .prop (Zotero treats orphan .prop as corruption)
    assert len(session.calls) == 2
    zip_url, zip_data, zip_headers = session.calls[0]
    prop_url, prop_data, prop_headers = session.calls[1]
    assert zip_url == "https://dav.example.com/zotero/ABCD1234.zip"
    assert prop_url == "https://dav.example.com/zotero/ABCD1234.prop"
    assert zip_headers.get("Content-Type") == "application/zip"
    assert prop_headers.get("Content-Type", "").startswith("text/xml")
    assert session.auth == ("alice", "secret")

    # return values match what we'll write to the Zotero attachment item
    import hashlib
    expected_md5 = hashlib.md5(b"%PDF-1.4 hello").hexdigest()  # noqa: S324
    assert md5_hex == expected_md5
    assert mtime_ms == int(src.stat().st_mtime * 1000)

    # prop XML carries the same md5 and mtime
    prop_text = prop_data.decode("utf-8")
    assert f"<hash>{expected_md5}</hash>" in prop_text
    assert f"<mtime>{mtime_ms}</mtime>" in prop_text


@skip_on_ci
def test_upload_attachment_zip_contains_file_under_basename(tmp_path, monkeypatch):
    src = tmp_path / "subdir" / "paper.pdf"
    src.parent.mkdir()
    src.write_bytes(b"%PDF body")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src)

    import io
    _zip_url, zip_data, _ = session.calls[0]
    with ZipFile(io.BytesIO(zip_data)) as zf:
        names = zf.namelist()
        # zip stores the BASENAME — not the full path — so Zotero clients
        # can extract it next to other attachments without leaking paths.
        assert names == ["paper.pdf"]
        assert zf.read("paper.pdf") == b"%PDF body"


@skip_on_ci
def test_upload_attachment_raises_when_not_configured(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    monkeypatch.delenv("ZOTERO_WEBDAV_URL", raising=False)
    monkeypatch.delenv("ZOTERO_WEBDAV_USERNAME", raising=False)
    monkeypatch.delenv("ZOTERO_WEBDAV_PASSWORD", raising=False)

    try:
        webdav.upload_attachment_to_webdav("ABCD1234", src)
    except webdav.WebDAVNotConfiguredError as exc:
        assert "ZOTERO_WEBDAV_URL" in str(exc)
    else:
        raise AssertionError("Expected WebDAVNotConfiguredError when env vars are missing")


@skip_on_ci
def test_upload_attachment_escapes_key_in_urls(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("AB/CD", src)

    assert session.calls[0][0] == "https://dav.example.com/zotero/AB%2FCD.zip"
    assert session.calls[1][0] == "https://dav.example.com/zotero/AB%2FCD.prop"


@skip_on_ci
def test_upload_default_timeout_is_60s(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.delenv("ZOTERO_WEBDAV_TIMEOUT", raising=False)
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src)

    # Both PUTs use the historic default — (10s connect, 60s read).
    assert session.timeouts == [(10.0, 60.0), (10.0, 60.0)]


@skip_on_ci
def test_upload_env_override_respected(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setenv("ZOTERO_WEBDAV_TIMEOUT", "300")
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src)

    assert session.timeouts == [(10.0, 300.0), (10.0, 300.0)]


@skip_on_ci
def test_upload_explicit_timeout_wins_over_env(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setenv("ZOTERO_WEBDAV_TIMEOUT", "300")
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src, timeout=45.0)

    # Explicit caller arg takes precedence over the env var.
    assert session.timeouts == [(10.0, 45.0), (10.0, 45.0)]
