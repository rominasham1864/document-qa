"""Search Chroma chunks using the E5 query embedding."""

from django.core.management.base import BaseCommand, CommandError

from documents.services.vector_store import search_chunks


class Command(BaseCommand):
    help = "Search indexed document chunks (no LLM answer generation)."

    def add_arguments(self, parser):
        parser.add_argument("--query", required=True)
        parser.add_argument("--k", type=int, default=4)
        parser.add_argument("--document-id", type=int, default=None)

    def handle(self, *args, **options):
        try:
            results = search_chunks(
                options["query"],
                k=options["k"],
                document_id=options["document_id"],
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        if not results:
            self.stdout.write("No indexed chunks matched the query.")
            return

        self.stdout.write(f"Retrieved {len(results)} chunk(s):")
        for rank, chunk in enumerate(results, start=1):
            meta = chunk.metadata
            preview = " ".join(chunk.page_content.split())[:250]
            self.stdout.write(
                f"{rank}. {meta.get('title', 'Untitled')} "
                f"(Document {meta.get('document_id')}, chunk {meta.get('chunk_id')})"
            )
            self.stdout.write(f"   {preview}")
