"""Stamp `document_id` and `shared` onto documents indexed before scoping.

WHY THIS IS NOT OPTIONAL
------------------------
Retrieval matches the shared corpus on `shared: True`. A document indexed
before that field existed carries neither it nor `document_id`, so it matches
nothing and is INVISIBLE rather than shared - agents quietly stop citing
documents that are still sitting in the index.

That is the safe direction to be wrong in, and it is deliberate: the
alternative design matched everything and excluded what was hidden, which
fails the other way round - a document nobody stamped stays readable by every
bot, including the ones answering people outside the organization, and nothing
anywhere says so.

But it still has to be corrected, and this is what corrects it.

Cheap: metadata is updated in place, so nothing is re-embedded and no provider
is called. Idempotent, so running it twice is harmless and running it again
after a partial failure finishes the job.

    manage.py backfill_knowledge_scoping --dry-run
    manage.py backfill_knowledge_scoping
"""

from django.core.management.base import BaseCommand

from llm.models import KnowledgeDocument
from llm.scripts.tasks import get_rag


class Command(BaseCommand):
    help = "Write document_id and shared onto already-indexed knowledge vectors."

    def add_arguments(self, parser):
        parser.add_argument(
            "--organization",
            help="Only this organization id. Default: every organization.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be stamped and change nothing.",
        )

    def handle(self, *args, **options):
        documents = (
            KnowledgeDocument.objects.filter(status=KnowledgeDocument.STATUS_INDEXED)
            .exclude(vector_ids=[])
            .prefetch_related("agents")
            .order_by("organization_id", "created_at")
        )
        if options["organization"]:
            documents = documents.filter(organization_id=options["organization"])

        total = documents.count()
        if not total:
            self.stdout.write("Nothing indexed to stamp.")
            return

        self.stdout.write(f"{total} indexed document(s) to stamp.\n")

        rag = None if options["dry_run"] else get_rag()
        chunks_total = stamped_total = 0
        failures = []

        for document in documents:
            vector_ids = list(document.vector_ids or [])
            shared = not document.agents.all()
            chunks_total += len(vector_ids)

            label = (
                f"  {document.title[:48]:<48} {len(vector_ids):>4} chunks  "
                f"{'shared' if shared else 'restricted'}"
            )

            if rag is None:
                self.stdout.write(label)
                continue

            stamped = rag.stamp_scoping(
                vector_ids,
                organization_id=document.organization_id,
                document_id=document.pk,
                shared=shared,
            )
            stamped_total += stamped
            if stamped != len(vector_ids):
                failures.append((document.pk, stamped, len(vector_ids)))
                self.stdout.write(
                    self.style.WARNING(f"{label}  -> only {stamped} stamped")
                )
            else:
                self.stdout.write(label)

        if rag is None:
            self.stdout.write(
                f"\nDry run: {chunks_total} chunk(s) across {total} document(s) "
                f"would be stamped."
            )
            return

        self.stdout.write(f"\nStamped {stamped_total}/{chunks_total} chunk(s).")
        if failures:
            # Named rather than summarised: a document that is still unstamped
            # is a document its agents cannot read, and the operator needs to
            # know which one before deciding whether to re-run or re-index.
            self.stdout.write(
                self.style.WARNING(
                    f"{len(failures)} document(s) incomplete - safe to run again:"
                )
            )
            for document_id, stamped, expected in failures:
                self.stdout.write(f"  {document_id}: {stamped}/{expected}")
        else:
            self.stdout.write(self.style.SUCCESS("All documents stamped."))
