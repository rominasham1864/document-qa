"""Admin interface for uploaded documents."""

from django import forms
from django.contrib import admin
from django.core.files.uploadedfile import UploadedFile

from .models import Document
from .services.docx_parser import DocumentExtractionError, extract_docx_text


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
        if hasattr(form, "new_extracted_text"):
            obj.extracted_text = form.new_extracted_text
        super().save_model(request, obj, form, change)
