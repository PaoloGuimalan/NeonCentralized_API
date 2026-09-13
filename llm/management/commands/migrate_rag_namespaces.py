"""Move vectors out of the default Pinecone namespace and into per-organization ones.

WHY THIS IS A COMMAND AND NOT A DATA MIGRATION
----------------------------------------------
It touches Pinecone, not Postgres, so `manage.py migrate` is the wrong place:
a Django migration that made network calls to a third party would make every
deployment's schema step depend on that service being up.

It is also SAFE TO NOT RUN. `RAG['NAMESPACE_MODE']` defaults to "dual", which
reads both namespaces, so an unmigrated deployment keeps working - it just
pays an extra query. This command is what lets you stop paying it.

HOW IT MOVES THEM
-----------------
Pinecone has no "move between namespaces" operation, so each vector is read
(values and metadata), written into its organization's namespace under the
SAME id, and only then deleted from the default one. Same id matters:
`KnowledgeDocument.vector_ids` records what was written, and changing the ids
would orphan every document's handle on its own chunks.

Copy-then-delete, in that order, per batch. A crash half way leaves a vector
in BOTH namespaces, which is harmless - the same id in two places dedupes on
read and the next run finishes the job. The reverse order would lose data.

    manage.py migrate_rag_namespaces --dry-run     # count what would move
    manage.py migrate_rag_namespaces               # move it
"""

import time

from django.core.management.base import BaseCommand, CommandError

from llm.services.rag import LEGACY_NAMESPACE, CustomerServiceRAG

# Pinecone caps how many ids one fetch or delete may name.
BATCH = 100
# How many ids to pull from a single list() page.
PAGE = 500


class Command(BaseCommand):
    help = "Move legacy Pinecone vectors into per-organization namespaces."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would move without changing anything.",
        )
        parser.add_argument(
            "--organization",
            default="",
            help="Only move vectors belonging to this organization id.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        only = options["organization"]

        try:
            rag = CustomerServiceRAG()
        except Exception as ex:
            raise CommandError(f"Could not reach Pinecone: {ex}") from ex

        index = rag.index
        moved = 0
        skipped_no_org = 0
        scanned = 0
        started = time.time()

        self.stdout.write(
            f"Reading the legacy namespace of index {rag.index_name!r}…"
        )

        try:
            pages = index.list(namespace=LEGACY_NAMESPACE, limit=PAGE)
        except Exception as ex:
            raise CommandError(f"Could not list vectors: {ex}") from ex

        for page in pages:
            ids = [str(value) for value in page]
            if not ids:
                continue
            scanned += len(ids)

            for start in range(0, len(ids), BATCH):
                batch = ids[start : start + BATCH]
                fetched = index.fetch(ids=batch, namespace=LEGACY_NAMESPACE)
                vectors = getattr(fetched, "vectors", None) or {}

                by_namespace = {}
                for vector_id, vector in vectors.items():
                    metadata = _metadata(vector)
                    organization_id = str(metadata.get("organization_id") or "")
                    if not organization_id:
                        # Nothing says which tenant it belongs to, so there is
                        # no namespace it can honestly be moved into. Left
                        # where it is - "dual" mode still reads it - and
                        # counted, because a pile of these means something
                        # wrote vectors without tenancy metadata.
                        skipped_no_org += 1
                        continue
                    if only and organization_id != only:
                        continue

                    by_namespace.setdefault(organization_id, []).append(
                        {
                            "id": vector_id,
                            "values": _values(vector),
                            "metadata": metadata,
                        }
                    )

                for organization_id, payload in by_namespace.items():
                    if dry_run:
                        moved += len(payload)
                        continue

                    # Copy first. A crash between these two leaves the vector
                    # in both namespaces, which the next run tidies up; the
                    # other order would lose it.
                    index.upsert(vectors=payload, namespace=organization_id)
                    index.delete(
                        ids=[item["id"] for item in payload],
                        namespace=LEGACY_NAMESPACE,
                    )
                    moved += len(payload)

        elapsed = time.time() - started
        self.stdout.write("")
        self.stdout.write(f"  scanned            {scanned}")
        self.stdout.write(f"  {'would move' if dry_run else 'moved'}         {moved}")
        if skipped_no_org:
            self.stdout.write(
                self.style.WARNING(
                    f"  skipped            {skipped_no_org} (no organization_id metadata)"
                )
            )
        self.stdout.write(f"  took               {elapsed:.1f}s")
        self.stdout.write("")

        if dry_run:
            self.stdout.write("Dry run - nothing was changed.")
            return

        if moved and not skipped_no_org and not only:
            self.stdout.write(
                self.style.SUCCESS(
                    "Done. Set RAG['NAMESPACE_MODE'] = 'namespaced' (env "
                    "RAG_NAMESPACE_MODE=namespaced) to stop reading the legacy "
                    "namespace and drop one query per retrieval."
                )
            )
        elif skipped_no_org:
            self.stdout.write(
                self.style.WARNING(
                    "Some vectors could not be placed, so leave NAMESPACE_MODE "
                    "on 'dual' - switching now would hide them."
                )
            )
        else:
            self.stdout.write("Nothing left in the legacy namespace.")


def _metadata(vector):
    metadata = getattr(vector, "metadata", None)
    if metadata is None and isinstance(vector, dict):
        metadata = vector.get("metadata")
    return dict(metadata or {})


def _values(vector):
    values = getattr(vector, "values", None)
    if values is None and isinstance(vector, dict):
        values = vector.get("values")
    return list(values or [])
