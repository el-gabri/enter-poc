"""Application services (use cases).

Import from the concrete modules (``app.services.analysis``,
``app.services.composer``) - re-exports here would create an import cycle:
graph -> composer -> (package init) -> analysis -> graph.
"""
