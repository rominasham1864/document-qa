"""Extract searchable text from DOCX files, independently of Django views."""

from zipfile import BadZipFile

from docx import Document as open_docx
from docx.opc.exceptions import OpcError
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError


class DocumentExtractionError(ValueError):
    """Raised when the uploaded DOCX cannot provide readable text."""


def extract_docx_text(file_obj) -> str:
    """Return paragraphs and table rows in their original document order.

    ``file_obj`` is a seekable binary stream (e.g. a Django UploadedFile).
    Its position is restored to the beginning so Django can save it afterward.
    """
    try:
        file_obj.seek(0)
        document = open_docx(file_obj)
        parts = []

        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text:
                    parts.append(text)
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [" ".join(cell.text.split()) for cell in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))

        extracted = "\n\n".join(parts)
        if not extracted:
            raise DocumentExtractionError(
                "The DOCX contains no extractable paragraph or table text."
            )
        return extracted
    except (OpcError, BadZipFile, KeyError, OSError, ValueError, XMLSyntaxError) as exc:
        if isinstance(exc, DocumentExtractionError):
            raise
        raise DocumentExtractionError(
            "The file is not a readable DOCX document."
        ) from exc
    finally:
        file_obj.seek(0)
