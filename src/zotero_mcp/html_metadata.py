"""Extract bibliographic metadata embedded in a publisher page's head."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_HIGHWIRE = {
    "citation_title": "title",
    "citation_journal_title": "publication",
    "citation_conference_title": "publication",
    "citation_inbook_title": "book_title",
    "citation_book_title": "book_title",
    "citation_publisher": "publisher",
    "citation_volume": "volume",
    "citation_issue": "issue",
    "citation_firstpage": "first_page",
    "citation_lastpage": "last_page",
    "citation_doi": "doi",
    "citation_issn": "issn",
    "citation_isbn": "isbn",
    "citation_language": "language",
    "citation_abstract": "abstract",
    "citation_pdf_url": "pdf_url",
    "citation_dissertation_institution": "institution",
    "citation_technical_report_institution": "institution",
}
_DUBLIN_CORE = {
    "dc.title": "title",
    "dc.publisher": "publisher",
    "dc.identifier": "doi",
    "dc.language": "language",
    "dc.description": "abstract",
    "dc.source": "publication",
    "dcterms.abstract": "abstract",
    "dcterms.issued": "date",
}
_DATE_TAGS = (
    "citation_publication_date",
    "citation_date",
    "citation_online_date",
    "citation_cover_date",
    "citation_year",
)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


@dataclass
class EmbeddedMetadata:
    title: str = ""
    authors: list[tuple[str, str]] = field(default_factory=list)
    publication: str = ""
    book_title: str = ""
    publisher: str = ""
    volume: str = ""
    issue: str = ""
    first_page: str = ""
    last_page: str = ""
    date: str = ""
    doi: str = ""
    issn: str = ""
    isbn: str = ""
    language: str = ""
    abstract: str = ""
    pdf_url: str = ""
    institution: str = ""

    @property
    def pages(self) -> str:
        if self.first_page and self.last_page:
            return self.first_page if self.first_page == self.last_page else f"{self.first_page}-{self.last_page}"
        return self.first_page or self.last_page

    def is_usable(self) -> bool:
        return bool(self.title or self.doi)

    def looks_like_article(self) -> bool:
        return bool(self.publication or self.volume or self.issue or self.first_page or self.issn)

    def looks_like_chapter(self) -> bool:
        return bool(self.book_title or self.isbn) and not self.publication


class _HeadMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.metas: list[tuple[str, str]] = []
        self.done = False

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if tag == "body":
            self.done = True
            return
        if tag != "meta":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        name = values.get("name") or values.get("property") or values.get("http-equiv")
        content = values.get("content", "")
        if name and content:
            self.metas.append((name.strip().lower(), content.strip()))

    def handle_endtag(self, tag):
        if tag == "head":
            self.done = True


def parse_meta_tags(html: str) -> dict[str, list[str]]:
    parser = _HeadMetaParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    output: dict[str, list[str]] = {}
    for name, content in parser.metas:
        output.setdefault(name, []).append(content)
    return output


def _clean_doi(value: str) -> str:
    match = _DOI_RE.search(value or "")
    return match.group(1).rstrip(".,;)") if match else ""


def _split_name(value: str) -> tuple[str, str]:
    value = " ".join(value.split())
    if "," in value:
        last, _, first = value.partition(",")
        return first.strip(), last.strip()
    parts = value.split()
    return ("", parts[0]) if len(parts) == 1 else (" ".join(parts[:-1]), parts[-1]) if parts else ("", "")


def extract_embedded_metadata(html: str) -> EmbeddedMetadata:
    tags = parse_meta_tags(html)
    meta = EmbeddedMetadata()

    def first(name: str) -> str:
        return (tags.get(name) or [""])[0]

    for name, attr in _HIGHWIRE.items():
        if value := first(name):
            setattr(meta, attr, value)
    for name, attr in _DUBLIN_CORE.items():
        if getattr(meta, attr):
            continue
        value = first(name)
        if attr == "doi":
            value = _clean_doi(value)
        if value:
            setattr(meta, attr, value)
    if not meta.title:
        meta.title = first("og:title") or first("twitter:title")
    for name in _DATE_TAGS:
        if value := first(name):
            meta.date = value.replace("/", "-")
            break
    meta.doi = _clean_doi(meta.doi)
    raw_authors = []
    for name in ("citation_author", "citation_authors", "dc.creator", "dc.contributor"):
        raw_authors.extend(tags.get(name, []))
    seen = set()
    for raw in raw_authors:
        for piece in raw.split(";"):
            first_name, last_name = _split_name(piece)
            key = (first_name.casefold(), last_name.casefold())
            if last_name and key not in seen:
                seen.add(key)
                meta.authors.append((first_name, last_name))
    return meta
