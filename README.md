# Zotero MCP: Chat with your Research Library—Local or Web—in Claude, ChatGPT, and more.

<p align="center">
  <a href="https://www.zotero.org/">
    <img src="https://img.shields.io/badge/Zotero-CC2936?style=for-the-badge&logo=zotero&logoColor=white" alt="Zotero">
  </a>
  <a href="https://www.anthropic.com/claude">
    <img src="https://img.shields.io/badge/Claude-6849C3?style=for-the-badge&logo=anthropic&logoColor=white" alt="Claude">
  </a>
  <a href="https://chatgpt.com/">
    <img src="https://img.shields.io/badge/ChatGPT-74AA9C?style=for-the-badge&logo=openai&logoColor=white" alt="ChatGPT">
  </a>
  <a href="https://modelcontextprotocol.io/introduction">
    <img src="https://img.shields.io/badge/MCP-0175C2?style=for-the-badge&logoColor=white" alt="MCP">
  </a>
  <a href="https://pypi.org/project/zotero-mcp-server/">
    <img src="https://img.shields.io/pypi/v/zotero-mcp-server?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://discord.gg/BvgjbcBUqg">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</p>

**Zotero MCP** seamlessly connects your [Zotero](https://www.zotero.org/) research library with [ChatGPT](https://openai.com), [Claude](https://www.anthropic.com/claude), and other AI assistants (e.g., [Cherry Studio](https://cherry-ai.com/), [Chorus](https://chorus.sh), [Cursor](https://www.cursor.com/)) via the [Model Context Protocol](https://modelcontextprotocol.io/introduction). Review papers, get summaries, analyze citations, extract PDF annotations, and more!

---

## ✨ Features

### 🧠 AI-Powered Semantic Search
- **Vector-based similarity search** over your entire research library (requires `[semantic]` extra)
- **Multiple embedding models**: Default (free, local), OpenAI, Gemini, and Ollama
- **Intelligent results** with similarity scores and contextual matching
- **Auto-updating database** with configurable sync schedules

### 🔍 Search Your Library
- Find papers, articles, and books by title, author, or content
- Perform complex searches with multiple criteria
- Browse collections, tags, and recent additions
- Semantic search for conceptual and topic-based discovery

### 📚 Access Your Content
- Retrieve detailed metadata for any item (markdown or BibTeX export)
- Get full text content (when available)
- Look up items by BetterBibTeX citation key

### 📝 Work with Annotations
- Extract and search PDF annotations with page numbers
- Access Zotero's native annotations
- Create and update notes; update stored annotations (new PDF/EPUB highlight creation requires `[pdf]`)
- Extract PDF table of contents / outlines (requires `[pdf]` extra)

### ✏️ Write Operations
- **Add papers by DOI** with auto-fetched metadata and open-access PDF cascade (Unpaywall, arXiv, Semantic Scholar, PMC)
- **Add papers by URL** (arXiv, DOI links, generic webpages) or from local files
- Create and manage collections, update item metadata, batch-update tags
- Find and merge duplicate items with dry-run preview
- **Local-first writes**: Zotero desktop authorizes changes directly; no cloud API key required

### 📊 Scite Citation Intelligence
- **Citation tallies**: See how many papers support, contrast, or mention each item — the MCP version of the [Scite Zotero Plugin](https://github.com/scitedotai/scite-zotero-plugin)
- **Retraction alerts**: Scan your library for retracted or corrected papers
- No Scite account required — uses public API endpoints

### 🌐 Flexible Access Methods
- Local mode for offline reads and writes (Zotero 10+; no cloud API key needed)
- Web API for cloud library access
- Automatic Web API fallback for older Zotero versions when credentials exist

### ⌨️ Standalone CLI (`zotero-cli`)
- Search, browse, and edit your library directly from the terminal — no AI assistant required
- Ideal for scripting, automation, and quick lookups
- Short aliases (`s`, `g`, `ann`, `coll`) for interactive use

## 🚀 Quick Install

> **New to the command line?** Try the community-built [Zotero MCP Setup](https://github.com/ehawkin/zotero-mcp-setup) — includes a macOS GUI installer (DMG), one-click install scripts for Mac/Windows, and a step-by-step guide. No Terminal experience needed.

### Default Installation (core tools only)

The base install is lightweight — it includes search, metadata retrieval, stored-annotation access, notes, and non-PDF write operations. No ML/AI dependencies are pulled in.

#### Installing via uv (recommended)

```bash
uv tool install zotero-mcp-server
zotero-mcp setup  # Auto-configure (Claude Desktop supported)
```

#### Installing via pip

```bash
pip install zotero-mcp-server
zotero-mcp setup  # Auto-configure (Claude Desktop supported)
```

#### Installing via pipx

```bash
pipx install zotero-mcp-server
zotero-mcp setup  # Auto-configure (Claude Desktop supported)
```

### Optional Extras

Heavy ML/PDF dependencies are separated into optional extras so the base install stays fast and small:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `semantic` | Semantic search via ChromaDB, sentence-transformers, OpenAI/Gemini embeddings | `pip install "zotero-mcp-server[semantic]"` |
| `pdf` | PDF pages/outlines/layout and PDF/EPUB annotation authoring | `pip install "zotero-mcp-server[pdf]"` |
| `scite` | Compatibility alias; Scite uses the core HTTP dependency and is already available | `pip install "zotero-mcp-server[scite]"` |
| `all` | Everything above | `pip install "zotero-mcp-server[all]"` |

For example, with uv:
```bash
uv tool install "zotero-mcp-server[all]"    # Full install with all features
uv tool install "zotero-mcp-server[semantic]" # Just semantic search
```

If you only need basic library access, metadata/full-text retrieval, stored annotations, notes, and ordinary item/collection writes, the default install with no extras is sufficient.

#### Updating Your Installation

Keep zotero-mcp up to date with the smart update command:

```bash
# Check for updates
zotero-mcp update --check-only

# Update to latest version (preserves all configurations)
zotero-mcp update
```

## 🧠 Semantic Search

Zotero MCP now includes powerful AI-powered semantic search capabilities that let you find research based on concepts and meaning, not just keywords.

### Setup Semantic Search

During setup or separately, configure semantic search:

```bash
# Configure during initial setup (recommended)
zotero-mcp setup

# Or configure semantic search separately
zotero-mcp setup --semantic-config-only
```

**Available Embedding Models:**
- **Default (all-MiniLM-L6-v2)**: Free, runs locally, good for most use cases
- **OpenAI**: Better quality, requires API key (`text-embedding-3-small` or `text-embedding-3-large`)
- **Gemini**: Better quality, requires API key (`gemini-embedding-001`)
- **Ollama**: Runs locally via Ollama API (requires model name, e.g., 'qwen3-embedding')

**Using Ollama embeddings:**

Install and start Ollama, then pull an embedding model before running `zotero-mcp update-db`:

```bash
ollama serve

# Small model: fast and lightweight
ollama pull nomic-embed-text

# Medium model: better multilingual retrieval quality
ollama pull bge-m3
```

When prompted by `zotero-mcp setup --semantic-config-only`, choose **Ollama** and use either `nomic-embed-text` or `bge-m3` as the model name. If you change embedding models later, rebuild the index:

```bash
zotero-mcp update-db --force-rebuild
```

When you choose OpenAI, setup also asks whether database updates should use
OpenAI Batch API. Batch updates are cheaper for large libraries, but they are
asynchronous: submit the batch, wait for completion, then import the embeddings.

**Update Frequency Options:**
- **Manual**: Update only when you run `zotero-mcp update-db`
- **Auto on startup**: Update database every time the server starts
- **Daily**: Update once per day automatically
- **Every N days**: Set custom interval

### Using Semantic Search

After setup, initialize your search database:

```bash
# Build the semantic search database (fast, metadata-only)
zotero-mcp update-db

# Submit OpenAI embeddings through Batch API for this update
zotero-mcp update-db --openai-batch

# Check and import completed OpenAI Batch API embeddings
zotero-mcp openai-batch-status
zotero-mcp openai-batch-import

# Force realtime OpenAI embeddings even if Batch API is enabled in config
zotero-mcp update-db --no-openai-batch

# Use two realtime OpenAI-compatible encoder workers (ChromaDB writes stay sequential)
zotero-mcp update-db --embedding-concurrency 2

# Build with full-text extraction (slower, more comprehensive)
zotero-mcp update-db --fulltext

# Use your custom zotero.sqlite path
zotero-mcp update-db --fulltext --db-path "/Your_custom_path/zotero.sqlite"

# If the embedding model changed, clear and rebuild with realtime embeddings
zotero-mcp update-db --force-rebuild --force-clear --no-openai-batch

# Destructive alternative: clear the live index before a realtime rebuild
zotero-mcp update-db --force-rebuild --force-clear --no-openai-batch

# Check database status
zotero-mcp db-status
```

For an OpenAI-compatible proxy whose API model name is only an alias, declare
the real backend identity and optional retrieval instruction in `config.json`.
Changing `model_identity` then blocks search until an explicitly confirmed
force rebuild, preventing vectors from different models from being mixed:

```json
{
  "semantic_search": {
    "embedding_config": {
      "model_name": "api-alias",
      "model_identity": "Qwen3-Embedding-8B-Q8_0",
      "query_instruction": "Given a scientific literature search query, retrieve relevant passages that identify papers addressing the query"
    }
  }
}
```

With `--fulltext`, each update selects the best attachment currently
available for semantic retrieval. BetterIssa sources are preferred in this
order: `BetterIssa indexing text`, `BetterIssa semantic document`,
`BetterIssa Advanced OCR Markdown`, then `BetterIssa Reading View`. The
original PDF and legacy text sources remain fallbacks. `BetterIssa references`
is never embedded. A later artifact or an in-place BetterIssa upsert changes
the attachment fingerprint, so the affected item advances to the better source
on the next `update-db --fulltext` run without a force rebuild.

**Example Semantic Queries in your AI assistant:**
- *"Find research similar to machine learning concepts in neuroscience"*
- *"Papers that discuss climate change impacts on agriculture"*
- *"Research related to quantum computing applications"*
- *"Studies about social media influence on mental health"*
- *"Find papers conceptually similar to this abstract: [paste abstract]"*

The semantic search provides similarity scores and finds papers based on conceptual understanding, not just keyword matching.

## 🖥️ Setup & Usage

Full documentation is available at [Zotero MCP docs](https://stevenyuyy.com/zotero-mcp/).

**Requirements**
- Python 3.10+
- Zotero 7+ (for local API and attachment access)
- An MCP-compatible client (e.g., Claude Desktop, ChatGPT Developer Mode, Cherry Studio, Chorus)

**For ChatGPT setup: see the [Getting Started guide](./docs/getting-started.md).**

### Configure Zotero

The Zotero local API must be enabled for the MCP server to work.

In Zotero 9, the local API toggle is under Settings → Advanced → 'Allow other applications on this computer to communicate with Zotero'.

Here is a screenshot:

![Zotero local API](./docs/zotero-local-api.png)

### For Claude Desktop / Claude Code (MCP client)

#### Configuration
After installation, either:

1. **Auto-configure** (recommended):
   ```bash
   zotero-mcp setup
   ```

2. **Manual configuration**:
   For Claude Desktop, add this to `claude_desktop_config.json`.
   For Claude Code, add this to `~/.claude.json`:
   ```json
   {
     "mcpServers": {
       "zotero": {
         "command": "zotero-mcp",
         "env": {
           "ZOTERO_LOCAL": "true"
         }
       }
     }
   }
   ```

   `ZOTERO_LOCAL: "true"` is sufficient for reads and, with Zotero 10 or
   newer, writes. The first write opens Zotero's authorization dialog. Choose
   **Always Allow** to reuse the local authorization; otherwise Zotero grants a
   single-use key and prompts again for the next write.

   Cloud credentials are optional in local mode. When configured, they are
   used only as a compatibility fallback if the running Zotero does not expose
   local writes. For remote Web API setup, use the instructions below.

   You can authorize before the first MCP write from a terminal:

   ```bash
   zotero-mcp authorize-local-writes
   ```

   To expose a different Linux workstation's loopback-only Zotero API as a
   boot-persistent LAN endpoint, use the included
   [`systemd-socket-proxyd` installer](contrib/zotero-local-api-proxy/README.md).

   > **Important Note**: Environmental variables set in the shell you run `claude` in will override these values.

   > **Tip:** If Claude Desktop reports it can't find the `zotero-mcp` command, use the
   > absolute path instead (run `zotero-mcp setup-info` or `which zotero-mcp` to find it) —
   > GUI apps don't always inherit your shell `PATH`.

#### Usage

1. Start Zotero desktop (make sure local API is enabled in preferences)
2. Launch Claude Desktop / Claude Code
3. For Claude Desktop, access the Zotero-MCP tool through Claude Desktop's tools interface.
For Claude Code, run the `/mcp` command, and make sure the Zotero MCP server is connected.

Example prompts:
- "Search my library for papers on machine learning"
- "Find recent articles I've added about climate change"
- "Summarize the key findings from my paper on quantum computing"
- "Extract all PDF annotations from my paper on neural networks"
- "Search my notes and annotations for mentions of 'reinforcement learning'"
- "Show me papers tagged '#Arm' excluding those with '#Crypt' in my library"
- "Search for papers on operating system with tag '#Arm'"
- "Export the BibTeX citation for papers on machine learning"
- **"Find papers conceptually similar to deep learning in computer vision"** *(semantic search)*
- **"Research that relates to the intersection of AI and healthcare"** *(semantic search)*
- **"Papers that discuss topics similar to this abstract: [paste text]"** *(semantic search)*

### For Autohand Code

After installing Zotero MCP, add a local server with:

```bash
autohand mcp add zotero env ZOTERO_LOCAL=true zotero-mcp
```

Add `--scope project` after `add` to keep the server configuration in the current project. For remote Web API access, add the credentials described below to the `env` command. See [Autohand Code](https://github.com/autohandai/code-cli/) for current installation and CLI details.

### For Cherry Studio

#### Configuration
Go to Settings -> MCP Servers -> Edit MCP Configuration, and add the following:

```json
{
  "mcpServers": {
    "zotero": {
      "name": "zotero",
      "type": "stdio",
      "isActive": true,
      "command": "zotero-mcp",
      "args": [],
      "env": {
        "ZOTERO_LOCAL": "true"
      }
    }
  }
}
```
Then click "Save".

Cherry Studio also provides a visual configuration method for general settings and tools selection.

## 🔧 Advanced Configuration

### Using Web API Instead of Local API

For accessing your Zotero library via the web API (useful for remote setups):

```bash
zotero-mcp setup --no-local --api-key YOUR_API_KEY --library-id YOUR_LIBRARY_ID
```

### Environment Variables

**Zotero Connection:**
- `ZOTERO_LOCAL=true`: Use the local Zotero API (default: false)
- `ZOTERO_LOCAL_PORT`: Override the local Zotero API port (default: 23119)
- `ZOTERO_REMOTE_LOCAL_URL`: Prefer another Zotero Local API for writes, then fall back to server-local Zotero. Accepts LAN HTTP, HTTPS, or a loopback SSH tunnel URL ending in `/api`
- `ZOTERO_REMOTE_LOCAL_HOST_HEADER`: Host header sent through a remote-local proxy (default: `127.0.0.1:23119`; set `preserve` when a reverse proxy rewrites Host itself)
- `ZOTERO_REMOTE_LOCAL_API_KEY`: Optional remembered Local API key for the remote-local Zotero instance
- `ZOTERO_LOCAL_API_KEY`: Optional remembered Local API key used for local endpoints when a remote-specific key is not set
- `ZOTERO_MCP_LOCAL_AUTH_PATH`: Override the private remembered local-write authorization file
- `ZOTERO_API_KEY`: Your Zotero API key (for web API)
- `ZOTERO_LIBRARY_ID`: Your Zotero library ID (for web API)
- `ZOTERO_LIBRARY_TYPE`: The type of library (user or group, default: user)
- `ZOTERO_MCP_LOCK_TIMEOUT`: Maximum seconds to wait for another Zotero API request across threads or processes (default: 45; `0` waits indefinitely)
- `ZOTERO_MCP_API_LOCK_PATH`: Optional shared API lock-file path when MCP and CLI processes use different home/config directories
- `ZOTERO_MCP_UPDATE_LOCK_PATH`: Optional shared semantic-update lock path (default: `~/.config/zotero-mcp/update.lock`)
- `ZOTERO_MCP_WRITE_SECRET`: Required per-call admin secret for every MCP tool that mutates Zotero or the semantic index. The value is never exposed in tool descriptions
- `ZOTERO_MCP_PUBLIC_BASE_URL`: Public MCP URL prefix used to create short-lived signed attachment upload/download URLs
- `ZOTERO_MCP_ATTACHMENT_MAX_BYTES`: Maximum staged attachment upload size (default: 512 MiB)
- `ZOTERO_MCP_ATTACHMENT_INLINE_MAX_BYTES`: Maximum binary embedded as base64 in a tool result (default: 1 MiB; hard cap: 32 MiB)
- `ZOTERO_MCP_ATTACHMENT_RESOURCE_MAX_BYTES`: Maximum binary returned through an in-memory MCP resource (default: 8 MiB; hard cap: 64 MiB)
- `ZOTERO_MCP_ATTACHMENT_LOCK_TIMEOUT`: Maximum seconds to wait for the same attachment idempotency key (default: 45)
- `ZOTERO_MCP_ATTACHMENT_STATE_DIR`: Private staging/token state directory (default: `~/.cache/zotero-mcp/attachments`)
- `ZOTERO_WEBDAV_URL`: Optional WebDAV folder URL for direct attachment downloads in remote mode
- `ZOTERO_WEBDAV_USERNAME`: Optional WebDAV username
- `ZOTERO_WEBDAV_PASSWORD`: Optional WebDAV password
- `OPENALEX_API_KEY`: Optional free OpenAlex key for a larger daily citation-graph query budget

**Semantic Search:**
- `ZOTERO_EMBEDDING_MODEL`: Embedding model to use (default, openai, gemini, ollama)
- `OPENAI_API_KEY`: Your OpenAI API key (for OpenAI embeddings)
- `OPENAI_EMBEDDING_MODEL`: OpenAI model name (text-embedding-3-small, text-embedding-3-large)
- `OPENAI_BASE_URL`: Custom OpenAI endpoint URL (optional, for use with compatible APIs)
- OpenAI Batch API indexing is configured by `zotero-mcp setup` and can be overridden with
  `zotero-mcp update-db --openai-batch` or `--no-openai-batch`
- `GEMINI_API_KEY`: Your Gemini API key (for Gemini embeddings)
- `GEMINI_EMBEDDING_MODEL`: Gemini model name (gemini-embedding-001)
- `GEMINI_BASE_URL`: Custom Gemini endpoint URL (optional, for use with compatible APIs)
- `OLLAMA_EMBEDDING_MODEL`: Ollama embedding model name (qwen3-embedding by default)
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `ZOTERO_DB_PATH`: Custom `zotero.sqlite` path (optional). When unset, the
  database is located automatically: a data directory configured in Zotero's
  preferences (read from the profile's `prefs.js`) is tried first, then the
  default `~/Zotero` location.
- `ZOTERO_SEARCH_BACKEND=split|api|sqlite`: Search routing policy. `split` is
  the default: ordinary keyword and tag searches retain Zotero's live local-API
  semantics, while advanced and all-library searches use `zotero.sqlite`.
  `api` forces the live API and disables all-library search; `sqlite` forces
  every supported metadata search through SQLite. Unsupported single-library
  requests fall back to the API; global requests fail closed rather than
  silently searching only the active library.

### Command-Line Options

```bash
# Run the server directly
zotero-mcp serve

# Specify transport method
zotero-mcp serve --transport stdio|streamable-http|sse

# Select the exposed MCP surface (auto is the default)
zotero-mcp serve --tool-profile auto|research|full|admin|connector|all

# Setup and configuration
zotero-mcp setup --help                    # Get help on setup options
zotero-mcp setup --semantic-config-only    # Configure only semantic search
zotero-mcp setup-info                      # Show installation path and config info for MCP clients

# Updates and maintenance
zotero-mcp update                          # Update to latest version
zotero-mcp update --check-only             # Check for updates without installing
zotero-mcp update --force                  # Force update even if up to date

# Semantic search database management
zotero-mcp update-db                       # Index API title and abstract only (default)
zotero-mcp update-db --openai-batch        # Submit OpenAI embeddings through Batch API
zotero-mcp update-db --no-openai-batch     # Force realtime OpenAI embeddings for this run
zotero-mcp update-db --embedding-concurrency 2 # Run two realtime OpenAI-compatible encoder workers
zotero-mcp openai-batch-status             # Check latest OpenAI embedding batch status
zotero-mcp openai-batch-import             # Import completed OpenAI batch embeddings
zotero-mcp update-db --fulltext            # Add one preferred local attachment per item
zotero-mcp update-db --force-rebuild       # Force complete database rebuild
zotero-mcp update-db --force-rebuild --force-clear --no-openai-batch # Clear first (required for model changes)
zotero-mcp update-db --fulltext --force-rebuild  # Rebuild with local attachments
zotero-mcp update-db --fulltext --db-path "your_path/to/zotero.sqlite" # Customize your Zotero database path
zotero-mcp db-status                       # Show database status and info

# General
zotero-mcp version                         # Show current version
```

Updates prune removed Zotero items first and refresh changed bibliographic
metadata immediately. Ordinary incremental updates write replacement vectors
before removing obsolete chunks. A realtime `--force-rebuild` invalidates the
stored content hashes, computes each complete item's new vectors, then replaces
that item's old records. If interrupted, the next ordinary `update-db` reuses
finished items and continues the rebuild. `--force-clear` clears the collection
first and is required when the embedding model or vector dimensions changed; it
requires `--force-rebuild` and realtime embeddings.

## ⌨️ CLI Mode (`zotero-cli`)

`zotero-cli` is a standalone terminal interface to your Zotero library. It uses the same tools as the MCP server but without needing an AI assistant — useful for quick lookups, shell scripts, and automation.

Use `zotero-mcp` when your AI client supports MCP (Claude Desktop, ChatGPT). Use `zotero-cli` for shell scripts, cron jobs, or agentic pipelines with shell access (e.g. Claude Code) — CLI commands cost far fewer tokens than MCP tool schemas and compose naturally with Unix pipes.

Both share the same configuration set up by `zotero-mcp setup`.

### Quick reference

```bash
# Search
zotero-cli search "machine learning"           # keyword search
zotero-cli s "neural networks" --limit 5       # short alias, limit results
zotero-cli search --mode semantic "attention mechanisms"
zotero-cli search --mode tag "important,reviewed"

# Get item details
zotero-cli get metadata ABC123                 # markdown metadata
zotero-cli g metadata ABC123 --format bibtex  # BibTeX export
zotero-cli get fulltext ABC123                 # full text
zotero-cli get children ABC123                 # attachments and notes

# Edit item metadata
zotero-cli edit ABC123 --title "New Title"
zotero-cli edit ABC123 --add-tags "reviewed,important" --date "2024"

# Notes and annotations
zotero-cli notes list ABC123
zotero-cli notes create --item-key ABC123 --text "My note" --tags "idea"
zotero-cli notes create --item-key ABC123 --text -   # read from stdin
zotero-cli ann list ABC123                    # annotations (short alias)
zotero-cli ann search "highlight text"

# Add items
zotero-cli add doi 10.1038/s41586-021-03819-2
zotero-cli add url https://arxiv.org/abs/2301.00001
zotero-cli add file --filepath /path/to/paper.pdf --title "Override Title"
zotero-cli add isbn 9780262046305
zotero-cli add bibtex --file refs.bib                # or --bibtex '@article{...}'
zotero-cli add bibtex --bibtex - < refs.bib          # stdin via -
zotero-cli add csl-json --file refs.json             # or --json '...' / --json -

# --collections accepts keys, names, or parent/child paths — resolved and
# validated before the item is created (a typo fails the add, with suggestions,
# instead of leaving an unfiled item)
zotero-cli add doi 10.1038/s41586-021-03819-2 --collections "Reading List"
zotero-cli collections manage --item-keys ABC123 --add-to "_project/topic"

# Adds are idempotent by default (--if-exists file): if the item is already in
# the library it is reused — filed into any missing collections, given any
# missing tags — instead of duplicated. Re-running the same command is a no-op.
zotero-cli add doi 10.1038/s41586-021-03819-2 -c "Reading List"   # run it twice: converges
zotero-cli add doi 10.1038/s41586-021-03819-2 --if-exists skip       # never touch existing
zotero-cli add doi 10.1038/s41586-021-03819-2 --if-exists duplicate  # old behavior
zotero-cli add doi 10.1038/s41586-021-03819-2 -c "New Topic" --create-collections
# -c/--collection is repeatable and never comma-split (names with commas work);
# --collections remains the comma-separated form

# Collections and tags
zotero-cli coll list                          # list collections (short alias)
zotero-cli coll search "PhD Research"
zotero-cli tags list

# Semantic search database
zotero-cli db update
zotero-cli db update --fulltext --force-rebuild
zotero-cli db status

# Library and duplicates
zotero-cli library info
zotero-cli duplicates find
```

### Verbose mode

Add `-v` anywhere to see progress messages (e.g., which API calls are made):

```bash
zotero-cli -v search "CRISPR"
```

## 📑 PDF Annotation Extraction

Zotero MCP includes advanced PDF annotation extraction capabilities:

- **Direct PDF Processing**: Extract annotations directly from PDF files, even if they're not yet indexed by Zotero
- **Enhanced Search**: Search through PDF annotations and comments
- **Image Annotation Support**: Extract image annotations from PDFs
- **Seamless Integration**: Works alongside Zotero's native annotation system

For optimal annotation extraction, it is **highly recommended** to install the [Better BibTeX plugin](https://retorque.re/zotero-better-bibtex/installation/) for Zotero. The annotation-related functions have been primarily tested with this plugin and provide enhanced functionality when it's available.


The first time you use PDF annotation features, the necessary tools will be automatically downloaded.

## 🔗 Managing Related Items

Zotero MCP now supports managing relationships between items in your library. This is useful for linking related papers, tracking versions, or connecting preprints to their published versions.

### View Related Items
```
zotero_get_item_related(item_key="ABCD1234")
```

### Add a Relation
Create a bidirectional link between two items:
```
zotero_add_item_relation(
    item_key="ABCD1234",
    related_item_key="EFGH5678",
    relation_type="dc:relation"  # Optional, defaults to "dc:relation"
)
```

### Remove a Relation
```
zotero_remove_item_relation(
    item_key="ABCD1234",
    related_item_key="EFGH5678",
    remove_bidirectional=True  # Also remove the reverse relation (default: true)
)
```

**Relation Types:**
- `dc:relation` — General related items (default)
- `owl:sameAs` — Items that are the same work (e.g., preprint and published version)

## 📚 Available Tools

### 🧠 Semantic Search Tools
- `zotero_semantic_search`: AI-powered similarity search with embedding models
- `zotero_get_semantic_context`: Retrieve an exact matched passage and optional neighboring chunks
- `zotero_update_search_database`: Manually update the semantic search database
- `zotero_get_search_database_status`: Check database status and configuration
- `zotero_get_search_database_health`: Audit semantic storage for the active library
- `zotero_get_capabilities`: Show the active tool profile, library, credentials, and optional features

Tool profiles keep the advertised surface aligned with the deployment. `auto`
selects `full` in local mode or when a web API key is available, and `research` otherwise;
`connector` exposes only the standard `search`/`fetch` pair plus capabilities;
`admin` exposes semantic maintenance and health tools. `all` is intended for
diagnosis but still applies credential, local-path, and optional-dependency
capability gates.

### 🔍 Search Tools
- `zotero_search_items`: Search by keywords; supports collection subtrees and, with the SQLite backend, all libraries
- `zotero_advanced_search`: Perform bounded multi-criteria searches with correct date sorting, collection-subtree scope, and optional global SQLite scope
- `zotero_get_collections`: List collections
- `zotero_get_collection_items`: Page through items in a collection or collection subtree using `limit` and `offset`
- `zotero_get_tags`: List all tags
- `zotero_get_recent`: Get recently added items
- `zotero_search_by_tag`: Search using custom tag filters, collection subtrees, or all local libraries
- `zotero_search_by_citation_key`: Look up an item by an exact Better BibTeX citation key

### 🗂️ Library and Feed Tools
- `zotero_list_libraries`: List My Library, accessible groups, and local feeds
- `zotero_switch_library`: Change the active library for the current MCP session
- `zotero_list_feeds`: List local Zotero RSS feeds
- `zotero_get_feed_items`: Read items from one local RSS feed

### 📚 Content Tools
- `zotero_get_item_metadata`: Get detailed metadata (supports `format="markdown"`, `format="json"` for complete raw Zotero metadata, and `format="bibtex"`)
- `zotero_get_item_fulltext`: Get full text content
- `zotero_get_document_text`: Get the canonical BetterIssa-first model-facing document text with source provenance
- `zotero_list_attachments`: List every attachment and its binary capabilities
- `zotero_get_attachment`: Return an exact binary resource and short-lived streaming download URL
- `zotero_prepare_attachment_upload`: Create a checksummed staged binary upload
- `zotero_prepare_attachment_change`: Preview and authorize a version-bound destructive attachment change
- `zotero_put_attachment`: Create or replace an imported attachment from a staged upload
- `zotero_update_attachment`: Update attachment metadata, with confirmation for reparenting
- `zotero_set_attachment_trashed`: Trash or restore an attachment; permanent deletion is unavailable
- `zotero_get_item_children`: Get attachments and notes
- `zotero_get_items_children`: Fetch direct children for multiple parent items
- `zotero_get_attachment_path`: Return local attachment paths only when explicitly enabled
- `zotero_read_pdf_pages`: Read a bounded page range from a selected PDF
- `zotero_get_pdf_outline`: Extract a PDF's embedded outline/bookmarks

Inline attachment data is intentionally conservative: base64 expands a binary
by roughly one third, then both the MCP server and client hold encoded and
decoded copies. The 1 MiB default is for small convenience payloads, not an
attachment-size limit. Use the signed streaming URL for larger files, or raise
the bounded inline/resource settings above when both server and client have
adequate memory and message-size limits.

### 📝 Annotation & Notes Tools
- `zotero_get_annotations`: Get annotations (including direct PDF extraction)
- `zotero_get_notes`: Retrieve notes from your Zotero library
- `zotero_search_notes`: Search in notes and annotations (including PDF-extracted)
- `zotero_create_note`: Create a new note for an item (beta feature)
- `zotero_update_note`: Replace or append to an existing note
- `zotero_delete_note`: Move a note to Zotero Trash
- `zotero_create_annotation`: Create a PDF or EPUB text highlight
- `zotero_create_area_annotation`: Create a rectangular PDF image/area annotation
- `zotero_update_annotation`: Update annotation text, comment, color, or tags
- `zotero_delete_annotation`: Move an annotation to Zotero Trash
- `zotero_get_page_layout`: Detect figure/table regions on a PDF page (with captions and normalized coordinates) for accurate area annotation placement

### 📊 Scite Citation Intelligence Tools
Scite support uses the core HTTP dependency and is available in the default installation; the historical `[scite]` extra remains as a compatibility alias.
- `scite_enrich_item`: Get Scite citation tallies and retraction alerts for a paper
- `scite_enrich_search`: Search your Zotero library with Scite-enriched results (tallies + alerts inline)
- `scite_check_retractions`: Scan items for retractions and editorial notices

### 📦 Item & Collection Management Tools
- `zotero_add_by_doi`: Add one or multiple DOIs with Crossref metadata and open-access PDF attachment
- `zotero_add_by_url`: Add one or multiple URLs; publisher pages contribute Highwire/Dublin Core citation metadata
- `zotero_add_by_isbn`: Add one or multiple ISBNs through the Open Library + Google Books cascade
- `zotero_add_by_bibtex`: Add one or more items from BibTeX (inline or .bib file)
- `zotero_add_by_csl_json`: Add one or more items from CSL JSON (inline or file)
- `zotero_add_from_file`: Import a local PDF, EPUB, DjVu, DOC/DOCX, ODT, or RTF file (PDFs get automatic DOI extraction)

All add tools take a `collections` parameter accepting collection keys, names, or `parent/child` paths — resolved and validated before the item is created, so unknown or ambiguous specs fail with suggestions instead of producing an unfiled item. They also take `if_exists` (`"reuse"` — default — returns an identifier match unchanged; `"merge"` adds missing collections and tags to the match; `"duplicate"` explicitly creates another item) and `create_missing_collections` (create unknown collection specs, including path chains, instead of failing). Legacy `skip`/`file` values remain aliases for `reuse`/`merge`. Attachment-aware import tools use `attach_mode="auto|none|linked_url|required"`; an unsatisfied `required` request is reported as partial because Zotero metadata creation cannot be rolled back reliably.
- `zotero_create_collection`: Create a new collection (folder/project) in your library
- `zotero_search_collections`: Search for collections by name to find their keys
- `zotero_manage_collections`: Add or remove items from collections (accepts keys, names, or `parent/child` paths)
- `zotero_update_item`: Update metadata for an existing item (title, tags, abstract, date, etc.)
- `zotero_delete_item`: Move a non-note item to Zotero Trash
- `zotero_delete_collection`: Permanently delete a collection after an explicit confirmation preview
- `zotero_batch_update_tags`: Add/remove tags across a bounded item selection
- `zotero_batch_update_extra`: Upsert/remove structured lines in multiple Extra fields
- `zotero_find_duplicates`: Find duplicate items by title/DOI, or produce a read-only exact-DOI merge plan with deterministic keeper recommendations
- `zotero_merge_duplicates`: Merge duplicates only from a fresh, one-use, version- and child-inventory-bound dry-run plan

Deferred work, including the proposed JSON CLI and packaged local-agent skill,
is tracked in [docs/future-features.md](docs/future-features.md).

### 🧭 Agentic Research Tools
- `zotero_find_related_papers`: Traverse an OpenAlex citation neighborhood
- `zotero_library_coverage`: Audit confirmed, missing, and unknown PDF coverage
- `zotero_synthesize_annotations`: Build a per-paper digest of notes and highlights
- `zotero_export_bibliography`: Render bibliography, citation, or BibTeX output through Zotero

### 🔗 Related Items Tools
- `zotero_get_item_related`: Get all related items for a specific Zotero item
- `zotero_add_item_relation`: Add a related item relationship (creates bidirectional link)
- `zotero_remove_item_relation`: Remove a related item relationship

### 🔌 ChatGPT Connector Compatibility Tools
- `search`: Return typed `{results:[{id,title,url}]}` connector search data
- `fetch`: Return typed `{id,title,text,url,metadata}` connector document data

## 🧪 Testing

### Unit Tests
```bash
uv run pytest tests/
```

### Integration Test Plan
A representative live integration plan is included at `docs/integration-test-plan.md`. It's designed to be given to an MCP client against a disposable/test Zotero library. It covers the highest-risk read, write, PDF-cascade, attachment-mode, Better BibTeX, and multi-step workflows; the automated suite remains the exhaustive tool-contract regression check.

## 🔍 Troubleshooting

### General Issues
- **No results found**: Ensure Zotero is running and the local API is enabled. You need to toggle on `Allow other applications on this computer to communicate with Zotero` in Zotero preferences.
- **Can't connect to library**: Check your API key and library ID if using web API
- **Full text not available**: Make sure you're using Zotero 7+ for local full-text access
- **Local library limitations**: Some functionality (tagging, library modifications) may not work with local JS API. Consider using web library setup for full functionality. (See the [docs](docs/getting-started.md#local-library-limitations) for more info.)
- **Installation/search option switching issues**: Inspect with `zotero-mcp db-status` or `zotero_get_search_database_health` first. A confirmed `zotero-mcp update-db --force-rebuild` is the last resort because it re-embeds the whole library.

### Semantic Search Issues
- **"Missing required environment variables" when running update-db**: Run `zotero-mcp setup` to configure your environment, or the CLI will automatically load settings from your MCP client config (e.g., Claude Desktop)
- **ChromaDB / stale embedding model errors**: If you changed embedding models and see 404 errors (e.g., `text-embedding-004 is not found`), run `zotero-mcp update-db --force-rebuild --force-clear --no-openai-batch` to recreate the collection with your current model. If that doesn't work, inspect it with `zotero-mcp db-health` before removing any files.
- **Database update takes long**: Full-text attachment extraction is opt-in with `--fulltext`. Omit it to index title and abstract only, or use `--limit` for testing: `zotero-mcp update-db --limit 100`
- **Semantic search returns no results**: Ensure the database is initialized with `zotero-mcp update-db` and check status with `zotero-mcp db-status`
- **Limited search quality**: Use `zotero-mcp update-db --fulltext` to add the preferred local BetterIssa/PDF attachment to each item's title and abstract.
- **OpenAI/Gemini API errors**: Verify your API keys are correctly set and have sufficient credits/quota

### Update Issues
- **Update command fails**: Check your internet connection and try `zotero-mcp update --force`
- **Configuration lost after update**: The update process preserves configs automatically, but check `~/.config/zotero-mcp/` for backup files

## ☕ Support

If you find Zotero MCP useful, consider buying me a coffee!

<a href="https://buymeacoffee.com/stevenyuyy">
  <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">
</a>

## 📄 License

MIT
