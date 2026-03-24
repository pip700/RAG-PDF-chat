"""
Export utilities for extracted data.
"""

import json
from datetime import datetime


def export_to_json(data: dict, pretty: bool = True) -> str:
    """Convert data dict to JSON string."""
    return json.dumps(data, indent=4 if pretty else None, ensure_ascii=False)


def generate_report_text(data: dict, query: str) -> str:
    """Generate a formatted text report."""
    lines = [
        "=" * 60,
        "PDF RAG AGENT — EXTRACTION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Query: {query}",
        "-" * 60,
        "",
    ]

    def render(obj, indent=0):
        pad = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}📌 {k}:")
                    render(v, indent + 1)
                else:
                    lines.append(f"{pad}• {k}: {v}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}[{i + 1}]")
                    render(item, indent + 1)
                else:
                    lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{obj}")

    render(data)
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
