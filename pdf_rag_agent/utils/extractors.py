"""
Structured extraction schemas and prompts for different extraction modes.
"""

from pydantic import BaseModel, Field


# ── Pydantic Schemas for Structured Extraction ──────────────

class PersonEntity(BaseModel):
    name: str = Field(description="Full name of the person")
    role: str = Field(description="Role/designation (e.g., Patient, Doctor, Nurse)")
    details: str = Field(default="", description="Any extra details about this person")


class DateEntity(BaseModel):
    date: str = Field(description="The date value")
    context: str = Field(description="What the date refers to")


class NumericFinding(BaseModel):
    parameter: str = Field(description="Name of the measurement/parameter")
    value: str = Field(description="The numeric value with units")
    reference_range: str = Field(default="", description="Normal reference range if available")
    status: str = Field(default="", description="Normal / Abnormal / High / Low")


class ExtractedEntities(BaseModel):
    persons: list[PersonEntity] = Field(default_factory=list, description="All persons mentioned")
    dates: list[DateEntity] = Field(default_factory=list, description="All dates mentioned")
    numeric_findings: list[NumericFinding] = Field(default_factory=list, description="All numeric values/test results")
    key_observations: list[str] = Field(default_factory=list, description="Important observations or diagnoses")
    medications: list[str] = Field(default_factory=list, description="Medications mentioned")
    organizations: list[str] = Field(default_factory=list, description="Hospitals, labs, organizations")


class TableData(BaseModel):
    title: str = Field(description="Title or heading of the table")
    headers: list[str] = Field(description="Column headers")
    rows: list[list[str]] = Field(description="Table rows as list of lists")


class SummaryOutput(BaseModel):
    title: str = Field(description="Document title or subject")
    summary: str = Field(description="Concise summary of the document")
    key_points: list[str] = Field(description="Bullet-point key findings")
    conclusion: str = Field(default="", description="Overall conclusion if any")


# ── Extraction Prompts ──────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """
You are a precision extraction engine. Extract ALL entities from the provided context.
Be thorough — do NOT miss any person, date, numeric value, medication, or organization.

Return the result as structured JSON matching the schema exactly.
If a field has no data, return an empty list.

Context:
{context}
"""

TABLE_EXTRACTION_PROMPT = """
You are a table extraction specialist. Identify ALL tabular data in the provided context.
Convert each table into structured format with headers and rows.
If data is presented in a list-like or semi-tabular format, convert it into a proper table.

Context:
{context}
"""

SUMMARY_PROMPT = """
You are a document summarization expert. Create a comprehensive yet concise summary.
Include the title/subject, a narrative summary, key bullet points, and an overall conclusion.

Context:
{context}
"""

CUSTOM_EXTRACTION_PROMPT = """
You are a precision data extraction engine. The user wants to extract specific information.

User's extraction request: {user_request}

Extract EXACTLY what the user asks for from the context below.
Be precise, quote exact values, and structure your response clearly.

Context:
{context}
"""

COMPARISON_PROMPT = """
You are an analytical comparison engine. Compare and contrast information found in the documents.

User's comparison request: {user_request}

Analyze the context below and provide a detailed comparison with:
- Similarities
- Differences
- Key observations
- Tabular comparison where appropriate

Context:
{context}
"""
