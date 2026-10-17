"""Report generation: self-contained HTML and a lightweight PDF.

``reports.generator`` holds the ten-section template; the AI tools and the
Settings page both write through :func:`reports.generator.generate_report`.
"""

from reports import generator

__all__ = ["generator"]
