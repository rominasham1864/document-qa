"""Track whether the stored Chroma index matches each SQLite document.

Existing documents start pending, even if older Chroma vectors exist. Their
legacy vectors cannot pass the new READY/index_key retrieval guard; reindex
with the management command to make them searchable again.
"""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("documents", "0002_questionanswer")]

    operations = [
        migrations.AddField(
            model_name="document", name="index_status",
            field=models.CharField(
                choices=[("pending", "Pending"), ("processing", "Processing"),
                         ("ready", "Ready"), ("failed", "Failed")],
                default="pending", max_length=12,
            ),
        ),
        migrations.AddField(
            model_name="document", name="index_key",
            field=models.CharField(blank=True, default="", max_length=80),
        ),
        migrations.AddField(
            model_name="document", name="indexed_content_hash",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="document", name="indexed_title",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
        migrations.AddField(
            model_name="document", name="indexed_chunk_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="document", name="index_error",
            field=models.CharField(blank=True, default="", max_length=120),
        ),
        migrations.AddField(
            model_name="document", name="index_attempt",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
    ]
