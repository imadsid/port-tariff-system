"""
ingestion/pdf_extractor.py
PDF Text Extractor
Extracts raw text from port tariff PDFs page-by-page using PyMuPDF,
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from monitoring import get_logger

log = get_logger(__name__)

SECTION_RE = re.compile(
    r"^(?:SECTION\s+\d+|\d+\.\d+(?:\.\d+)?)\s+[A-Z].*$",
    re.MULTILINE,
)


@dataclass
class PageContent:
    page_num: int
    raw_text: str
    section: Optional[str] = None
    subsection: Optional[str] = None


@dataclass
class ExtractedDocument:
    source_path: str
    total_pages: int
    pages: list[PageContent]
    full_text: str
    metadata: dict = field(default_factory=dict)


class PDFTextExtractor:
    """
    Extracts clean text from a PDF file, page by page.
    Detects section headings and annotates each page with its section context.
    """

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        log.info("Extracting PDF", path=str(pdf_path))
        doc = fitz.open(str(pdf_path))

        pages: list[PageContent] = []
        all_text_parts: list[str] = []
        current_section: Optional[str] = None
        current_subsection: Optional[str] = None

        for idx in range(len(doc)):
            raw = doc[idx].get_text("text")
            cleaned = self._clean(raw)

            # Detect new section headings on this page
            for match in SECTION_RE.finditer(cleaned):
                heading = match.group(0).strip()
                if re.match(r"SECTION\s+\d+", heading, re.IGNORECASE):
                    current_section = heading
                    current_subsection = None
                else:
                    current_subsection = heading

            pages.append(PageContent(
                page_num=idx + 1,
                raw_text=cleaned,
                section=current_section,
                subsection=current_subsection,
            ))
            all_text_parts.append(cleaned)

        doc.close()
        full_text = "\n\n".join(all_text_parts)

        log.info("PDF extracted", pages=len(pages), chars=len(full_text))
        return ExtractedDocument(
            source_path=str(pdf_path),
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
            metadata={"filename": pdf_path.name, "total_pages": len(pages)},
        )

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\.{4,}", " ", text)
        text = re.sub(r"_{4,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"Tariff Book[^\n]*\n", "", text)
        return text.strip()
