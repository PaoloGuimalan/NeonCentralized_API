"""`manage.py runbots` - the bot supervisor.

Runs one process that leases whatever bots it can and holds an SSE connection
for each. Safe to run several copies: they divide the work by racing for leases
in Redis, and a crashed one's bots are picked up by a peer within a lease TTL.

    manage.py runbots              # run until stopped
    manage.py runbots --once       # one sweep, then exit (for checking config)

It does NOT generate replies. When a bot decides to answer, the work is queued
for a Celery worker - so this process stays cheap enough that hundreds of idle
bots fit in one small container, and a busy one does not slow the others down.
You need `celery -A neon worker` running too, or bots will decide to answer and
nothing will.
"""

import asyncio
import logging
import signal

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from chatterloop.supervisor import SWEEP_INTERVAL_SECONDS, Supervisor

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run the Chatterloop bot supervisor."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Sweep once and exit. Useful for checking configuration.",
        )
        parser.add_argument(
            "--sweep-interval",
            type=int,
            default=SWEEP_INTERVAL_SECONDS,
            help="Seconds between sweeps for newly runnable bots.",
        )

    def handle(self, *args, **options):
        if not getattr(settings, "DEVELOPER_SERVICE_BASE_URL", ""):
            # Checked up front rather than letting each bot fail its first
            # connection: without this the command starts, leases everything,
            # and then logs the same error once per bot forever.
            raise CommandError(
                "DEVELOPER_SERVICE_BASE_URL is not set, so there is nothing to "
                "subscribe to. Set it to your developer_service base URL."
            )

        try:
            from django_redis import get_redis_connection

            redis_client = get_redis_connection("default")
        except Exception as ex:
            # Refused rather than degraded. The lease IS the guarantee that one
            # bot has one consumer; running without it means every mention
            # answered once per supervisor, which is worse than not running.
            raise CommandError(
                f"Could not reach Redis, which the bot leases require: {ex}"
            ) from ex

        supervisor = Supervisor(
            redis_client,
            sweep_interval=options["sweep_interval"],
            once=options["once"],
        )

        asyncio.run(self._run(supervisor))

    async def _run(self, supervisor):
        loop = asyncio.get_running_loop()

        # SIGTERM is what a container orchestrator sends, and handling it is
        # what makes a deploy hand the leases back instead of leaving each bot
        # unheld until its TTL expires.
        for signame in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, supervisor.stop)
            except NotImplementedError:
                # Windows has no add_signal_handler for the proactor loop.
                # KeyboardInterrupt still unwinds through the finally in
                # Supervisor.run, which is what matters.
                pass

        try:
            await supervisor.run()
        except KeyboardInterrupt:
            await supervisor.shutdown()
