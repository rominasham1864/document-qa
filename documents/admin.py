"""Admin interface for uploaded documents."""

from django.contrib import admin

from .models import Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Manage documents without editing automatically generated metadata."""

    list_display = ("id", "title", "file", "created_at", "updated_at")
    search_fields = ("title",)
    list_filter = ("created_at",)
    ordering = ("-created_at",)
    list_per_page = 25

    # The document parser will populate extracted_text in task 5.
    readonly_fields = ("extracted_text", "created_at", "updated_at")
    fields = ("title", "file", "extracted_text", "created_at", "updated_at")
