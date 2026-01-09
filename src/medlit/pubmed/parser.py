"""XML parsing utilities for PubMed responses."""

from datetime import date
from xml.etree import ElementTree as ET

import structlog

from medlit.models import Article, Author

logger = structlog.get_logger(__name__)


def parse_search_results(xml_content: str) -> list[str]:
    """Parse esearch XML response to extract PMIDs.

    Args:
        xml_content: XML response from esearch

    Returns:
        List of PMIDs
    """
    try:
        root = ET.fromstring(xml_content)
        id_list = root.find(".//IdList")

        if id_list is None:
            return []

        pmids = [id_elem.text for id_elem in id_list.findall("Id") if id_elem.text]
        return pmids

    except ET.ParseError as e:
        logger.error("Failed to parse search results", error=str(e))
        return []


def parse_articles(xml_content: str) -> list[Article]:
    """Parse efetch XML response to extract article details.

    Args:
        xml_content: XML response from efetch

    Returns:
        List of Article objects
    """
    articles = []

    try:
        root = ET.fromstring(xml_content)

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article = _parse_single_article(article_elem)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning("Failed to parse article", error=str(e))
                continue

    except ET.ParseError as e:
        logger.error("Failed to parse articles XML", error=str(e))

    return articles


def _parse_single_article(article_elem: ET.Element) -> Article | None:
    """Parse a single PubmedArticle element.

    Args:
        article_elem: PubmedArticle XML element

    Returns:
        Article object or None if parsing fails
    """
    medline = article_elem.find(".//MedlineCitation")
    if medline is None:
        return None

    # Get PMID
    pmid_elem = medline.find(".//PMID")
    if pmid_elem is None or not pmid_elem.text:
        return None
    pmid = pmid_elem.text

    # Get article info
    article_info = medline.find(".//Article")
    if article_info is None:
        return None

    # Title
    title_elem = article_info.find(".//ArticleTitle")
    title = _get_text_content(title_elem) if title_elem is not None else ""

    # Abstract
    abstract = _parse_abstract(article_info)

    # Authors
    authors = _parse_authors(article_info)

    # Journal info
    journal_elem = article_info.find(".//Journal")
    journal = ""
    journal_abbrev = ""
    if journal_elem is not None:
        journal_title = journal_elem.find(".//Title")
        journal = journal_title.text if journal_title is not None and journal_title.text else ""

        iso_abbrev = journal_elem.find(".//ISOAbbreviation")
        journal_abbrev = iso_abbrev.text if iso_abbrev is not None and iso_abbrev.text else ""

    # Publication date
    pub_date, year = _parse_publication_date(article_info)

    # DOI
    doi = _parse_doi(article_elem)

    # Article type
    article_type = _parse_article_type(medline)

    # MeSH terms
    mesh_terms = _parse_mesh_terms(medline)

    # Keywords
    keywords = _parse_keywords(medline)

    return Article(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        journal_abbrev=journal_abbrev,
        publication_date=pub_date,
        year=year,
        doi=doi,
        article_type=article_type,
        mesh_terms=mesh_terms,
        keywords=keywords,
    )


def _get_text_content(elem: ET.Element) -> str:
    """Get all text content from an element, including nested elements."""
    return "".join(elem.itertext()).strip()


def _parse_abstract(article_info: ET.Element) -> str:
    """Parse abstract, handling structured abstracts."""
    abstract_elem = article_info.find(".//Abstract")
    if abstract_elem is None:
        return ""

    parts = []
    for text_elem in abstract_elem.findall(".//AbstractText"):
        label = text_elem.get("Label", "")
        text = _get_text_content(text_elem)

        if label and text:
            parts.append(f"{label}: {text}")
        elif text:
            parts.append(text)

    return "\n\n".join(parts)


def _parse_authors(article_info: ET.Element) -> list[Author]:
    """Parse author list."""
    authors = []
    author_list = article_info.find(".//AuthorList")

    if author_list is None:
        return authors

    for author_elem in author_list.findall("Author"):
        last_name_elem = author_elem.find("LastName")
        if last_name_elem is None or not last_name_elem.text:
            continue

        first_name_elem = author_elem.find("ForeName")
        initials_elem = author_elem.find("Initials")
        affiliation_elem = author_elem.find(".//Affiliation")

        authors.append(
            Author(
                last_name=last_name_elem.text,
                first_name=first_name_elem.text if first_name_elem is not None else "",
                initials=initials_elem.text if initials_elem is not None else "",
                affiliation=affiliation_elem.text if affiliation_elem is not None else None,
            )
        )

    return authors


def _parse_publication_date(article_info: ET.Element) -> tuple[date | None, int | None]:
    """Parse publication date."""
    pub_date = None
    year = None

    # Try ArticleDate first (electronic publication)
    article_date = article_info.find(".//ArticleDate")
    if article_date is not None:
        pub_date, year = _extract_date_from_element(article_date)

    # Fall back to PubDate
    if year is None:
        journal = article_info.find(".//Journal")
        if journal is not None:
            pub_date_elem = journal.find(".//PubDate")
            if pub_date_elem is not None:
                pub_date, year = _extract_date_from_element(pub_date_elem)

    return pub_date, year


def _extract_date_from_element(date_elem: ET.Element) -> tuple[date | None, int | None]:
    """Extract date from a date element."""
    year_elem = date_elem.find("Year")
    month_elem = date_elem.find("Month")
    day_elem = date_elem.find("Day")

    year = None
    pub_date = None

    if year_elem is not None and year_elem.text:
        try:
            year = int(year_elem.text)

            month = 1
            if month_elem is not None and month_elem.text:
                month = _parse_month(month_elem.text)

            day = 1
            if day_elem is not None and day_elem.text:
                try:
                    day = int(day_elem.text)
                except ValueError:
                    day = 1

            pub_date = date(year, month, day)
        except (ValueError, TypeError):
            pass

    return pub_date, year


def _parse_month(month_str: str) -> int:
    """Parse month string (numeric or abbreviated)."""
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    try:
        return int(month_str)
    except ValueError:
        return month_map.get(month_str.lower()[:3], 1)


def _parse_doi(article_elem: ET.Element) -> str | None:
    """Parse DOI from article data."""
    # Check ELocationID
    for eloc in article_elem.findall(".//ELocationID"):
        if eloc.get("EIdType") == "doi" and eloc.text:
            return eloc.text

    # Check ArticleIdList
    for article_id in article_elem.findall(".//ArticleId"):
        if article_id.get("IdType") == "doi" and article_id.text:
            return article_id.text

    return None


def _parse_article_type(medline: ET.Element) -> str | None:
    """Parse primary article type."""
    pub_type_list = medline.find(".//PublicationTypeList")
    if pub_type_list is None:
        return None

    # Priority order for article types
    priority_types = [
        "Meta-Analysis",
        "Systematic Review",
        "Randomized Controlled Trial",
        "Clinical Trial",
        "Review",
        "Guideline",
        "Practice Guideline",
    ]

    found_types = []
    for pub_type in pub_type_list.findall("PublicationType"):
        if pub_type.text:
            found_types.append(pub_type.text)

    # Return highest priority type found
    for ptype in priority_types:
        if ptype in found_types:
            return ptype

    # Return first type if no priority types found
    return found_types[0] if found_types else None


def _parse_mesh_terms(medline: ET.Element) -> list[str]:
    """Parse MeSH terms."""
    mesh_terms = []
    mesh_list = medline.find(".//MeshHeadingList")

    if mesh_list is not None:
        for heading in mesh_list.findall("MeshHeading"):
            descriptor = heading.find("DescriptorName")
            if descriptor is not None and descriptor.text:
                mesh_terms.append(descriptor.text)

    return mesh_terms


def _parse_keywords(medline: ET.Element) -> list[str]:
    """Parse keywords."""
    keywords = []
    keyword_list = medline.find(".//KeywordList")

    if keyword_list is not None:
        for keyword in keyword_list.findall("Keyword"):
            if keyword.text:
                keywords.append(keyword.text)

    return keywords
