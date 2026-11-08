"""Repair or create a Document's synchronized Chroma index."""

from django.core.management.base import BaseCommand, CommandError

from documents.models import Document
from documents.services.document_lifecycle import index_saved_document


class Command(BaseCommand):
    help = "Create/retry an index and update its SQLite READY/FAILED state."

    def add_arguments(self, parser):
        parser.add_argument("--document-id", type=int, required=True)

    def handle(self, *args, **options):
        document_id = options["document_id"]
        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist as exc:
            raise CommandError(f"Document {document_id} was not found in SQLite.") from exc
        try:
            count = index_saved_document(document)
        except Exception as exc:
            raise CommandError(
                f"Indexing document {document_id} failed ({type(exc).__name__}); "
                "status is FAILED. Check server logs, then retry."
            ) from exc
        self.stdout.write(self.style.SUCCESS(
            f"Document {document_id} READY: {count} chunk(s) indexed."
        ))
