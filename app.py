"""Chainlit UI for MedLit Agent."""

import re

import chainlit as cl

from medlit.agent import create_agent
from medlit.config.constants import MEDICAL_DISCLAIMER


def parse_answer_sections(answer: str) -> dict:
    """Parse the answer into main answer and collapsible sections.

    Returns:
        Dict with keys: main_answer, evidence_summary, limitations, sources
    """
    sections = {
        "main_answer": "",
        "evidence_summary": "",
        "limitations": "",
        "sources": "",
    }

    # Remove any disclaimer at the end
    text = answer
    if "Disclaimer:" in text:
        text = text.split("Disclaimer:")[0].strip()
    if "disclaimer:" in text:
        text = text.split("disclaimer:")[0].strip()

    # Clean up stray markdown markers and excessive whitespace
    # Remove standalone ** markers (often used as section separators by LLM)
    text = re.sub(r'^\*\*\s*$', '', text, flags=re.MULTILINE)
    # Remove standalone ### or ## markers
    text = re.sub(r'^#{2,3}\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple consecutive blank lines into at most 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Common section headers to look for
    evidence_patterns = [
        r"(?:##?\s*)?Summary of Key Evidence:?\s*",
        r"(?:##?\s*)?Key Evidence:?\s*",
        r"(?:##?\s*)?Evidence Summary:?\s*",
    ]

    limitation_patterns = [
        r"(?:##?\s*)?Important Limitations or Caveats:?\s*",
        r"(?:##?\s*)?Limitations:?\s*",
        r"(?:##?\s*)?Caveats:?\s*",
        r"(?:##?\s*)?Important Limitations:?\s*",
    ]

    source_patterns = [
        r"(?:##?\s*)?Sources:?\s*",
        r"(?:##?\s*)?Sources with PMIDs:?\s*",
        r"(?:##?\s*)?References:?\s*",
        r"(?:##?\s*)?Citations:?\s*",
    ]

    # Find section boundaries
    evidence_match = None
    limitation_match = None
    source_match = None

    for pattern in evidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            evidence_match = match
            break

    for pattern in limitation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            limitation_match = match
            break

    for pattern in source_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            source_match = match
            break

    # Extract sections based on matches
    positions = []
    if evidence_match:
        positions.append(("evidence", evidence_match.start(), evidence_match.end()))
    if limitation_match:
        positions.append(("limitation", limitation_match.start(), limitation_match.end()))
    if source_match:
        positions.append(("source", source_match.start(), source_match.end()))

    # Sort by position
    positions.sort(key=lambda x: x[1])

    if not positions:
        # No sections found, everything is main answer
        sections["main_answer"] = text.strip()
        return sections

    # Main answer is everything before first section
    sections["main_answer"] = text[:positions[0][1]].strip()

    # Extract each section
    for i, (section_type, _start, end) in enumerate(positions):
        # Find where this section ends (start of next section or end of text)
        section_end = positions[i + 1][1] if i + 1 < len(positions) else len(text)

        content = text[end:section_end].strip()

        # Additional cleanup for each section
        # Remove stray markdown and excessive whitespace
        content = re.sub(r'^\*\*\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^#{2,3}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()

        if section_type == "evidence":
            sections["evidence_summary"] = content
        elif section_type == "limitation":
            sections["limitations"] = content
        elif section_type == "source":
            sections["sources"] = content

    return sections


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Create agent and store in session
    agent = create_agent(enable_tracing=True)
    cl.user_session.set("agent", agent)

    # Welcome message
    await cl.Message(
        content="""# 🔬 MedLit Agent

Welcome! I help you find and understand evidence from published medical research.

**How to use:**
- Ask any medical question
- I'll search PubMed for relevant studies
- Then synthesize the evidence into a clear answer

**Example questions:**
- "Is aspirin effective for preventing heart attacks?"
- "What are the side effects of metformin?"
- "Does vitamin D supplementation help with depression?"

---
""",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process a user message."""
    agent = cl.user_session.get("agent")

    if not agent:
        agent = create_agent(enable_tracing=True)
        cl.user_session.set("agent", agent)

    question = message.content

    # Create a message to update with progress
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        # Step 1: Searching
        async with cl.Step(name="🔍 Searching PubMed", type="tool") as search_step:
            search_step.output = "Searching for relevant medical literature..."

        # Step 2: Processing
        async with cl.Step(name="📚 Fetching Articles", type="tool") as fetch_step:
            fetch_step.output = "Retrieving article details and abstracts..."

        # Step 3: Synthesizing
        async with cl.Step(name="🧪 Synthesizing Evidence", type="llm") as synth_step:
            synth_step.output = "Analyzing studies and generating synthesis..."

            # Actually run the agent
            response = await agent.ask(question)

        # Format the response
        if response.answer:
            # Parse sections from the answer
            sections = parse_answer_sections(response.answer)

            # Build formatted response with collapsible sections
            formatted_response = f"{sections['main_answer']}\n\n"

            # Add collapsible sections using HTML details/summary
            if sections['evidence_summary']:
                formatted_response += f"""<details>
<summary><strong>📊 Summary of Key Evidence</strong></summary>
<br>

{sections['evidence_summary']}

</details>

"""

            if sections['limitations']:
                formatted_response += f"""<details>
<summary><strong>⚠️ Limitations & Caveats</strong></summary>
<br>

{sections['limitations']}

</details>

"""

            if sections['sources']:
                formatted_response += f"""<details>
<summary><strong>📖 Sources</strong></summary>
<br>

{sections['sources']}

</details>

"""

            # Add metadata line
            formatted_response += "\n---\n"
            if response.evidence and response.evidence.quality:
                quality_emoji = {"high": "🟢", "moderate": "🟡", "low": "🟠", "unknown": "⚪"}
                emoji = quality_emoji.get(response.evidence.quality.value, "⚪")
                formatted_response += f"{emoji} Evidence: {response.evidence.quality.value.title()}"
            if response.citations:
                formatted_response += f" · 📖 {len(response.citations)} sources"

            formatted_response += f"\n\n<sub>⚠️ *{MEDICAL_DISCLAIMER}*</sub>"

            response_msg.content = formatted_response
            await response_msg.update()

        elif response.error_message:
            response_msg.content = f"❌ **Error:** {response.error_message}"
            await response_msg.update()

        else:
            response_msg.content = "I couldn't find relevant evidence for your question. Try rephrasing or asking about a different topic."
            await response_msg.update()

    except Exception as e:
        response_msg.content = f"❌ **Error:** {str(e)}"
        await response_msg.update()


@cl.on_stop
async def on_stop():
    """Handle chat stop."""
    pass
