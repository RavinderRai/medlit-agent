"""Tests for PubMed XML parser."""

import pytest

from medlit.pubmed.parser import (
    parse_search_results,
    parse_articles,
)


class TestParseSearchResults:
    """Tests for search results parsing."""

    def test_parse_valid_results(self, sample_xml_search_response):
        """Test parsing valid search results."""
        pmids = parse_search_results(sample_xml_search_response)
        assert len(pmids) == 3
        assert "38123456" in pmids
        assert "37654321" in pmids

    def test_parse_empty_results(self):
        """Test parsing empty results."""
        xml = """<?xml version="1.0"?>
        <eSearchResult>
            <Count>0</Count>
            <IdList></IdList>
        </eSearchResult>
        """
        pmids = parse_search_results(xml)
        assert len(pmids) == 0

    def test_parse_invalid_xml(self):
        """Test parsing invalid XML."""
        pmids = parse_search_results("not valid xml")
        assert len(pmids) == 0


class TestParseArticles:
    """Tests for article parsing."""

    def test_parse_valid_article(self, sample_xml_article_response):
        """Test parsing valid article."""
        articles = parse_articles(sample_xml_article_response)
        assert len(articles) == 1

        article = articles[0]
        assert article.pmid == "38123456"
        assert "Aspirin" in article.title
        assert article.year == 2023
        assert article.article_type == "Meta-Analysis"

    def test_parse_article_authors(self, sample_xml_article_response):
        """Test parsing article authors."""
        articles = parse_articles(sample_xml_article_response)
        article = articles[0]

        assert len(article.authors) == 1
        assert article.authors[0].last_name == "Smith"
        assert article.authors[0].first_name == "John"

    def test_parse_empty_articles(self):
        """Test parsing empty article set."""
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet></PubmedArticleSet>
        """
        articles = parse_articles(xml)
        assert len(articles) == 0

    def test_parse_invalid_xml(self):
        """Test parsing invalid XML."""
        articles = parse_articles("not valid xml")
        assert len(articles) == 0
