"""Application constants."""

# PubMed API
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_ENDPOINT = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_ENDPOINT = f"{PUBMED_BASE_URL}/efetch.fcgi"

# Rate limits (requests per second)
PUBMED_RATE_LIMIT = 3  # Without API key
PUBMED_RATE_LIMIT_WITH_KEY = 10  # With API key

# Search defaults
DEFAULT_MAX_RESULTS = 8
DEFAULT_DATE_RANGE_YEARS = 5
DEFAULT_SEARCH_FILTERS = {
    "species": "humans",
    "language": "english",
}

# Article types to prioritize
PREFERRED_ARTICLE_TYPES = [
    "Meta-Analysis",
    "Systematic Review",
    "Randomized Controlled Trial",
    "Clinical Trial",
    "Review",
    "Guideline",
]

# MeSH term mappings for common queries
MESH_TERM_MAPPINGS = {
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "diabetes": "diabetes mellitus",
    "cancer": "neoplasms",
    "stroke": "cerebrovascular accident",
    "blood thinner": "anticoagulants",
}

# Response formatting
MAX_ABSTRACT_LENGTH = 2000
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}"

# Disclaimer
MEDICAL_DISCLAIMER = """
**Disclaimer**: This information is synthesized from published medical literature
and is intended for educational purposes only. It should not be used as a substitute
for professional medical advice, diagnosis, or treatment. Always consult with a
qualified healthcare provider for medical decisions.
""".strip()
