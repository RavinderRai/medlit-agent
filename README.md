# MedLit Agent

AI agent that searches medical literature and synthesizes evidence-based clinical guidance.

Built with [Google ADK](https://google.github.io/adk-docs/) and [LangSmith](https://smith.langchain.com/) for observability.

## Features

- Natural language medical question processing
- PubMed search with MeSH term optimization
- Evidence synthesis with source citations
- LangSmith tracing for full observability

## Quick Start

### Prerequisites

- Python 3.11+
- Google API key (Gemini)
- LangSmith API key (optional, for tracing)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/medlit-agent.git
cd medlit-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Usage

```bash
# Run the CLI
medlit ask "Is low-dose aspirin recommended for preventing heart attacks in healthy adults?"

# Or use the Python API
python -c "
from medlit.agent import MedLitAgent
agent = MedLitAgent()
response = agent.ask('What are the side effects of metformin?')
print(response)
"
```

## Project Structure

```
medlit-agent/
├── config/              # Settings and constants
├── src/medlit/
│   ├── agent/           # Google ADK agent and tools
│   ├── prompts/         # Prompt templates and registry
│   ├── pubmed/          # PubMed API client
│   ├── observability/   # LangSmith integration
│   ├── models/          # Pydantic schemas
│   └── utils/           # Rate limiting, caching, etc.
├── tests/               # Unit, integration, and eval tests
├── scripts/             # Utility scripts
└── docker/              # Container configuration
```

## Development

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/

# Format code
ruff format .
```

## Configuration

See `.env.example` for all configuration options.

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Gemini API key | Yes |
| `LANGSMITH_API_KEY` | LangSmith API key | No |
| `LANGSMITH_PROJECT` | LangSmith project name | No |
| `NCBI_API_KEY` | PubMed API key (increases rate limit) | No |
| `REDIS_URL` | Redis URL for caching | No |

## License

MIT
