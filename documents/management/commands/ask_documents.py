"""CLI to test end-to-end RAG before the HTTP API is introduced in task 9."""

from django.core.management.base import BaseCommand, CommandError

from documents.services.rag import answer_question


class Command(BaseCommand):
    help = "Answer a question using indexed DOCX chunks and OpenRouter."

    def add_arguments(self, parser):
        parser.add_argument("--query", required=True, help="Question to answer")
        parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks (1-8)")
        parser.add_argument(
            "--document-id", type=int, default=None, help="Optional single-document filter"
        )

    def handle(self, *args, **options):
        try:
            result = answer_question(
                options["query"], k=options["k"], document_id=options["document_id"]
            )
        except (ValueError, RuntimeError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write("Answer:")
        self.stdout.write(result["answer"])
        self.stdout.write("\nSources:")
        if not result["sources"]:
            self.stdout.write("(none)")
        for source in result["sources"]:
            self.stdout.write(
                f"[{source['rank']}] {source['title']} "
                f"(Document {source['document_id']}, chunk {source['chunk_id']})"
            )
