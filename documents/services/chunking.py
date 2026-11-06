"""Split extracted document text into LangChain chunks with source metadata."""

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Values are CHARACTER counts, not token counts.
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100


def split_document_text(
    *,
    document_id: int,
    title: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[LangChainDocument]:
    """Return ordered, overlapping chunks ready for vector indexing.

    No database writes or embedding calls occur here. Chunk IDs are stable
    while a document's content and chunking configuration stay unchanged.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "؟ ", ". ", "، ", " ", ""],
    )
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces = splitter.split_text(normalized_text)
    return [
        LangChainDocument(
            page_content=piece,
            metadata={
                "document_id": document_id,
                "title": title,
                "chunk_index": index,
                "chunk_id": f"{document_id}:{index}",
            },
        )
        for index, piece in enumerate(pieces)
    ]
