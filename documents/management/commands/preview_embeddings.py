"""Preview chunking and embeddings for a previously uploaded document."""

from django.core.management.base import BaseCommand, CommandError

from documents.models import Document
from documents.services.chunking import split_document_text
from documents.services.embeddings import E5Embeddings


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--document-id", type=int, required=True)

    def handle(self, *args, **options):
        try:
            document = Document.objects.get(pk=options["document_id"])
        except Document.DoesNotExist as exc:
            raise CommandError("Document not found in this database.") from exc

        if not document.extracted_text.strip():
            raise CommandError(
                "Document has no extracted text. Upload/reupload a valid DOCX first."
            )

        chunks = split_document_text(
            document_id=document.pk,
            title=document.title,
            text=document.extracted_text,
        )
        vectors = E5Embeddings().embed_documents(
            [chunk.page_content for chunk in chunks]
        )
        self.stdout.write(f"Document ID: {document.pk}")
        self.stdout.write(f"Chunks: {len(chunks)}")
        self.stdout.write(f"Embedding dimension: {len(vectors[0])}")
        self.stdout.write("First chunk IDs: " + ", ".join(
            chunk.metadata["chunk_id"] for chunk in chunks[:3]
        ))
        self.stdout.write("Nothing was written to the vector database (task 7).")
