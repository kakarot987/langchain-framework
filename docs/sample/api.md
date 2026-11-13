# Documentation Assistant Interface

The assistant is a CLI application, not an HTTP API server. Run `docqa ask "question"` to ask a grounded question using a configured chat model.
Run `docqa index` to display how many local Markdown files and chunks were loaded.
Run `docqa demo` for an offline retrieval and read-only tool demonstration.
Run `docqa evaluate` to measure retrieval hit-at-k using a labeled JSON fixture.

## Guardrails

Only Markdown files below the configured document directory are ingested.
Two read-only tools are exposed: `list_documents` and `read_document`.
`read_document` requires an exact known filename; it never accepts arbitrary filesystem paths.
The assistant limits executions to two tool calls per question and does not execute shell commands.
