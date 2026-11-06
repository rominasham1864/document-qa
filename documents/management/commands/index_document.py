"""Index one previously uploaded Document in persistent ChromaDB."""

from django.core.management.base import BaseCommand, CommandError

from documents.models import Document
from documents.services.vector_store import index_document


class Command(BaseCommand):
    help = "Create/update Chroma chunks for a Document with extracted_text."

    def add_arguments(self, parser):
        parser.add_argument("--document-id", type=int, required=True)

    def handle(self, *args, **options):
        document_id = options["document_id"]
        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist as exc:
            raise CommandError(f"Document {document_id} was not found in SQLite.") from exc

        try:
            count = index_document(document)
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(
            f"Indexed Document {document_id}: {count} chunk(s) saved in ChromaDB."
        ))
