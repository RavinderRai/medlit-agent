"""Pytest configuration and fixtures."""

import os
from datetime import date

import pytest

# Set test environment
os.environ["ENVIRONMENT"] = "development"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture
def sample_question() -> str:
    """Sample medical question for testing."""
    return "Is low-dose aspirin recommended for preventing heart attacks in healthy adults?"


@pytest.fixture
def sample_pubmed_query() -> str:
    """Sample PubMed query string."""
    return "aspirin[MeSH] AND primary prevention[MeSH] AND cardiovascular diseases[MeSH]"


@pytest.fixture
def sample_pmids() -> list[str]:
    """Sample PMIDs for testing."""
    return ["38123456", "37654321", "36987654"]


@pytest.fixture
def sample_article_data() -> dict:
    """Sample article data dictionary."""
    return {
        "pmid": "38123456",
        "title": "Effects of Aspirin on Cardiovascular Prevention: A Meta-Analysis",
        "abstract": "Background: Aspirin has been used for cardiovascular prevention...",
        "authors": "Smith J, Jones M, et al.",
        "journal": "JAMA Internal Medicine",
        "year": 2023,
        "article_type": "Meta-Analysis",
    }


@pytest.fixture
def sample_xml_search_response() -> str:
    """Sample PubMed search XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <eSearchResult>
        <Count>3</Count>
        <RetMax>3</RetMax>
        <RetStart>0</RetStart>
        <IdList>
            <Id>38123456</Id>
            <Id>37654321</Id>
            <Id>36987654</Id>
        </IdList>
    </eSearchResult>
    """


@pytest.fixture
def sample_xml_article_response() -> str:
    """Sample PubMed article XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>38123456</PMID>
                <Article>
                    <ArticleTitle>Effects of Aspirin on Cardiovascular Prevention</ArticleTitle>
                    <Abstract>
                        <AbstractText>Background: This meta-analysis examines aspirin use...</AbstractText>
                    </Abstract>
                    <AuthorList>
                        <Author>
                            <LastName>Smith</LastName>
                            <ForeName>John</ForeName>
                            <Initials>J</Initials>
                        </Author>
                    </AuthorList>
                    <Journal>
                        <Title>JAMA Internal Medicine</Title>
                        <ISOAbbreviation>JAMA Intern Med</ISOAbbreviation>
                        <JournalIssue>
                            <PubDate>
                                <Year>2023</Year>
                                <Month>06</Month>
                            </PubDate>
                        </JournalIssue>
                    </Journal>
                </Article>
                <PublicationTypeList>
                    <PublicationType>Meta-Analysis</PublicationType>
                </PublicationTypeList>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """


@pytest.fixture
def date_range() -> tuple[date, date]:
    """Date range for testing."""
    today = date.today()
    min_date = date(today.year - 5, 1, 1)
    return min_date, today
