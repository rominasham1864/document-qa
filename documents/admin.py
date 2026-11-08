"""Admin interface for uploaded documents."""

import logging

from django import forms
from django.contrib import admin, messages
from django.core.files.uploadedfile import UploadedFile

from .models import Document
from .services.docx_parser import DocumentExtractionError, extract_docx_text
from .services.document_lifecycle import (
    index_saved_document, delete_document_and_index, get_index_snapshot,
)

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
    list_display = ("id", "title", "index_status", "file", "created_at", "updated_at")
    list_display_links = ("id", "title")
    search_fields = ("title",)
    list_filter = ("index_status", "created_at")
    ordering = ("-created_at",)
    list_per_page = 25
    readonly_fields = ("extracted_text", "index_status", "index_error", "created_at", "updated_at")
    fields = ("title", "file", "extracted_text", "index_status", "index_error", "created_at", "updated_at")

    def save_model(self, request, obj, form, change):
        changed = not change or bool({"title", "file"} & set(form.changed_data))
        old_filename = None
        previous = None
        if change and changed:
            old = Document.objects.get(pk=obj.pk)
            old_filename = old.file.name
            previous = get_index_snapshot(old)

        if hasattr(form, "new_extracted_text"):
            obj.extracted_text = form.new_extracted_text
        if changed:
            # Store data and invalidation together. Never serve a stale vector
            # after SQLite has already accepted a new document title or file.
            obj.index_status = Document.IndexStatus.PENDING
            obj.index_key = ""
            obj.index_error = ""
        super().save_model(request, obj, form, change)

        if changed:
            try:
                count = index_saved_document(
                    obj, previous_file_name=old_filename, previous_index=previous
                )
            except Exception:
                logger.exception("Could not index saved document %s", obj.pk)
                self.message_user(
                    request,
                    f"Document {obj.pk} was saved, but is NOT searchable. "
                    f"Retry: python manage.py index_document --document-id {obj.pk}",
                    level=messages.WARNING,
                )
            else:
                self.message_user(
                    request, f"Document is READY: {count} chunk(s).", level=messages.INFO
                )

    def delete_model(self, request, obj):
        # Never remove the SQLite row if its vector index cannot be removed.
        delete_document_and_index(obj)

    def delete_queryset(self, request, queryset):
        # Django's bulk delete bypasses delete_model; handle each row explicitly.
        for document in list(queryset):
            delete_document_and_index(document)
