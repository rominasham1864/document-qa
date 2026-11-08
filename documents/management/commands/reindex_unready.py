"""Rebuild old/failed/interrupted document indexes in small batches."""

from django.core.management.base import BaseCommand

from documents.models import Document
from documents.services.document_lifecycle import index_saved_document


class Command(BaseCommand):
    help = "Retry pending/failed/interrupted indexes without reuploading files."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=20)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **opts):
        if opts["limit"] < 1:
            self.stderr.write("--limit must be positive")
            return
        documents = list(Document.objects.exclude(
            index_status=Document.IndexStatus.READY
        ).order_by("pk")[:opts["limit"]])
        for document in documents:
            if opts["dry_run"]:
                self.stdout.write(f"Would reindex Document {document.pk}")
                continue
            try:
                count = index_saved_document(document)
            except Exception as exc:
                self.stderr.write(f"Document {document.pk} FAILED ({type(exc).__name__})")
            else:
                self.stdout.write(self.style.SUCCESS(
                    f"Document {document.pk} READY ({count} chunks)"
                ))
