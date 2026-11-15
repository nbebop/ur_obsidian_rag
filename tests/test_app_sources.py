from pathlib import Path
from urllib.parse import quote

from obsidian_rag.app import _format_sources
from obsidian_rag.query import Source


def test_sources_show_full_path_as_obsidian_link():
    path = Path("/vault/My Notes/café.md")
    out = _format_sources([Source(note="My Notes/café.md", path=path, score=0.5, snippet="hi")])
    assert str(path) in out
    assert f"(obsidian://open?path={quote(str(path), safe='')})" in out
    assert "%20" in out and "%C3%A9" in out


def test_no_sources_is_empty():
    assert _format_sources([]) == ""
