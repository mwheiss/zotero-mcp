"""Resolve Zotero base fields to each item type's actual API field."""

from __future__ import annotations

# Zotero's schema uses these type-specific fields for otherwise generic base
# fields. Keeping this table local makes reads and writes correct offline; the
# active item's template remains authoritative for field validity.
_ALIASES: dict[str, dict[str, str]] = {
    "artwork": {"medium": "artworkMedium"},
    "audioRecording": {"medium": "audioRecordingFormat", "publisher": "label"},
    "bill": {"number": "billNumber", "pages": "codePages", "volume": "codeVolume", "authority": "legislativeBody"},
    "blogPost": {"publicationTitle": "blogTitle", "type": "websiteType"},
    "book": {"medium": "format"},
    "bookSection": {"publicationTitle": "bookTitle", "medium": "format"},
    "case": {"title": "caseName", "authority": "court", "date": "dateDecided", "number": "docketNumber", "pages": "firstPage", "volume": "reporterVolume"},
    "computerProgram": {"publisher": "company"},
    "conferencePaper": {"publicationTitle": "proceedingsTitle"},
    "dataset": {"medium": "format", "number": "identifier", "publisher": "repository", "place": "repositoryLocation"},
    "dictionaryEntry": {"publicationTitle": "dictionaryTitle"},
    "email": {"title": "subject"},
    "encyclopediaArticle": {"publicationTitle": "encyclopediaTitle"},
    "film": {"publisher": "distributor", "type": "genre", "medium": "videoRecordingFormat"},
    "forumPost": {"publicationTitle": "forumTitle", "type": "postType"},
    "hearing": {"number": "documentNumber", "authority": "legislativeBody"},
    "interview": {"medium": "interviewMedium"},
    "letter": {"type": "letterType"},
    "manuscript": {"publisher": "institution", "type": "manuscriptType"},
    "map": {"type": "mapType"},
    "patent": {"date": "issueDate", "authority": "issuingAuthority", "status": "legalStatus", "number": "patentNumber", "originalDate": "priorityDate"},
    "podcast": {"medium": "audioFileType", "number": "episodeNumber"},
    "preprint": {"number": "archiveID", "type": "genre", "publisher": "repository"},
    "presentation": {"type": "presentationType", "publicationTitle": "sessionTitle"},
    "radioBroadcast": {"medium": "audioRecordingFormat", "number": "episodeNumber", "publisher": "network", "publicationTitle": "programTitle"},
    "report": {"publisher": "institution", "number": "reportNumber", "type": "reportType"},
    "standard": {"authority": "organization"},
    "statute": {"date": "dateEnacted", "title": "nameOfAct", "number": "publicLawNumber"},
    "thesis": {"type": "thesisType", "publisher": "university"},
    "tvBroadcast": {"number": "episodeNumber", "publisher": "network", "publicationTitle": "programTitle", "medium": "videoRecordingFormat"},
    "videoRecording": {"publisher": "studio", "medium": "videoRecordingFormat"},
    "webpage": {"publicationTitle": "websiteTitle", "type": "websiteType"},
}


def resolve_field(item_type: str, field: str) -> str:
    """Map a generic Zotero base field to its type-specific key."""
    return _ALIASES.get(item_type, {}).get(field, field)


def base_field_of(item_type: str, field: str) -> str:
    """Return the generic base field for a type-specific key."""
    for base, actual in _ALIASES.get(item_type, {}).items():
        if actual == field:
            return base
    return field
