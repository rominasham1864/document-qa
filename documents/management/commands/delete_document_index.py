"""Manually remove one Document's chunks from Chroma (SQLite stays intact)."""

from django.core.management.base import BaseCommand, CommandError

from documents.services.vector_store import delete_document_index


class Command(BaseCommand):
    help = "Delete Chroma vectors for one document ID; does not delete SQLite data."

    def add_arguments(self, parser):
        parser.add_argument("--document-id", type=int, required=True)

    def handle(self, *args, **options):
        try:
            removed = delete_document_index(options["document_id"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"Removed {removed} indexed chunk(s)."))
