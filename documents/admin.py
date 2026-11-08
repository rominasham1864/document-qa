"""Admin interface for uploaded documents."""

import logging

from django import forms
from django.contrib import admin, messages
from django.core.files.uploadedfile import UploadedFile

from .models import Document
from .services.docx_parser import DocumentExtractionError, extract_docx_text
from .services.document_lifecycle import index_saved_document, delete_document_and_index

logger = logging.getLogger(__name__)


class DocumentAdminForm(forms.ModelForm):
    """Validate new DOCX files and extract text before anything is saved."""

    class Meta:
        model = Document
        fields = ("title", "file")

    def clean_file(self):
        file = self.cleaned_data["file"]
        if isinstance(file, UploadedFile):
            try:
                self.new_extracted_text = extract_docx_text(file)
            except DocumentExtractionError as exc:
                raise forms.ValidationError(str(exc)) from exc
        return file


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Manage DOCX files and show their automatically extracted text."""

    form = DocumentAdminForm
    list_display = ("id", "title", "file", "created_at", "updated_at")
    list_display_links = ("id", "title")
    search_fields = ("title",)
    list_filter = ("created_at",)
    ordering = ("-created_at",)
    list_per_page = 25
    readonly_fields = ("extracted_text", "created_at", "updated_at")
    fields = ("title", "file", "extracted_text", "created_at", "updated_at")

    def save_model(self, request, obj, form, change):
        # Editing only the title must not reread the DOCX or erase existing text.
        previous_file_name = None
        if change:
            previous_file_name = Document.objects.only("file").get(pk=obj.pk).file.name
        if hasattr(form, "new_extracted_text"):
            obj.extracted_text = form.new_extracted_text
        super().save_model(request, obj, form, change)

        # Skip unrelated edits; metadata changes (title) also need reindexing.
        if not change or {"title", "file"} & set(form.changed_data):
            try:
                count = index_saved_document(obj, previous_file_name=previous_file_name)
            except Exception:
                logger.exception("Could not index saved document %s", obj.pk)
                self.message_user(
                    request,
                    f"Document {obj.pk} was saved but NOT indexed. "
                    f"Retry: python manage.py index_document --document-id {obj.pk}",
                    level=messages.WARNING,
                )
            else:
                self.message_user(
                    request, f"Document indexed: {count} chunk(s).", level=messages.INFO
                )

    def delete_model(self, request, obj):
        # Never remove the SQLite row if its vector index cannot be removed.
        delete_document_and_index(obj)

    def delete_queryset(self, request, queryset):
        # Django's bulk delete bypasses delete_model; handle each row explicitly.
        for document in list(queryset):
            delete_document_and_index(document)
