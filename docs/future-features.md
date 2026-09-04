# Future features

## JSON CLI and packaged agent skill

Status: deferred. This is intended for local agents with shell access; remote
ChatGPT clients should continue using MCP.

The goal is a low-context alternative to advertising the full MCP tool schema.
A local agent would load a small Zotero skill only when needed and call
`zotero-cli --json`. Local-agent configurations should use either this route or
the full Zotero MCP, not both, because attaching both preserves the MCP context
cost and gives the agent duplicate interfaces.

Rough implementation blueprint:

1. Define one stable JSON envelope shared with MCP: `ok`, `status`, `data`,
   `warnings`, and `errors`. Reuse the existing result classification and
   structured-data helpers rather than parsing a second contract independently.
2. Add JSON rendering to read-only CLI commands first: search, metadata, notes,
   collections, semantic search/context, and database status. Keep current human
   output unchanged unless `--json` is passed.
3. Generate the CLI reference from `argparse`; fail CI when the checked-in
   reference differs from the parser. Package a short skill that teaches the
   find-keys-then-read workflow and links to that generated reference.
4. Add parity tests that run representative operations through CLI and MCP and
   compare their structured fields, identifiers, empty/error classification,
   library scope, and pagination behavior.
5. Add explicit installation/configuration commands for supported local agent
   harnesses. Never overwrite unrelated instructions; update only a marked
   Zotero block and require `--force` for conflicting files.
6. Defer mutation commands until their authorization model is explicit. Writes
   must retain previews, version checks, confirmation tokens, and an intentional
   policy for the write secret rather than becoming a shell-side bypass.

Acceptance criteria: no duplicated business logic, generated documentation is
current, read results agree with MCP, the skill is not loaded eagerly, and
enabling the CLI route lets a local agent omit the full Zotero MCP surface.
