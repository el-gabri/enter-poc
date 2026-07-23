"""Structured logging setup.

Uses structlog so every log line is a key-value event. In development logs
render as colored console output; the JSON renderer can be enabled for
production (log aggregators like Datadog/CloudWatch parse it natively).
"""

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog + stdlib logging once at application startup."""
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(stream=sys.stdout, level=level.upper(), format="%(message)s")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structured logger."""
    return structlog.get_logger(name)
