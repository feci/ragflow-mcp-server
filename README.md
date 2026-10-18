# ragflow-mcp-server

An MCP server for managing [RAGFlow](https://github.com/infiniflow/ragflow) knowledge bases
through the RAGFlow HTTP API: create datasets, upload documents, run parsing, edit chunks
and metadata, and search.

RAGFlow ships its own MCP endpoint, but it only offers `ragflow_list_datasets`,
`ragflow_list_chats` and `ragflow_retrieval`. Agents could search the knowledge base and
could not add anything to it. This server fills that gap. Tested against RAGFlow
1.0.0-rc2 (Go API).

The Python package and the command are called `ragflow-kb-mcp`.

## Tools

| Area | Tools |
|---|---|
| Datasets | `ragflow_list_datasets`, `ragflow_get_dataset`, `ragflow_create_dataset`, `ragflow_update_dataset`, `ragflow_delete_datasets` |
| Documents | `ragflow_list_documents`, `ragflow_upload_document`, `ragflow_update_document`, `ragflow_download_document`, `ragflow_delete_documents` |
| Parsing | `ragflow_parse_documents`, `ragflow_stop_parsing` |
| Chunks | `ragflow_list_chunks`, `ragflow_add_chunk`, `ragflow_update_chunk`, `ragflow_delete_chunks` |
| Metadata | `ragflow_metadata_summary`, `ragflow_update_metadata` |
| Search | `ragflow_retrieval` |

Tools carry MCP annotations: read-only tools are marked `readOnlyHint`, and the delete
tools are marked `destructiveHint`, so clients can ask before running them.

### Uploading

`ragflow_upload_document` takes exactly one source:

- `text`: UTF-8 content such as Markdown notes. Needs a `filename` with an extension.
- `content_base64`: file bytes. The bytes pass through the model's context, so use this
  only for small files.
- `url`: RAGFlow downloads the file or page itself.
- `file_path`: a file on the machine that runs this server. This is off until the operator
  lists allowed directories in `RAGFLOW_UPLOAD_ROOTS`. Paths are resolved, including
  symlinks, before the check.

The file extension decides which RAGFlow parser runs. Parsing starts right after the
upload unless `parse` is false. Set `wait_seconds` (up to 600) to wait for the chunks.
An existing document with the same name makes the upload fail unless you pass
`replace_existing`. For `url` uploads, include the extension in `filename`: RAGFlow adds one
when it is missing, and the duplicate check compares the name you passed. In that case the new file is uploaded first and the old one is
deleted afterwards.

`ragflow_update_metadata` without `document_ids` or `metadata_condition` would change every
document in the dataset, so it asks for `all_documents: true` first.

## Configuration

| Variable | Required | Meaning |
|---|---|---|
| `RAGFLOW_BASE_URL` | yes | e.g. `https://ragflow.example.com` (`/api/v1` is added if missing) |
| `RAGFLOW_API_KEY` | yes | RAGFlow API key |
| `RAGFLOW_DEFAULT_EMBEDDING_MODEL` | no | embedding model for new datasets, `model_name@factory` |
| `RAGFLOW_DISABLED_TOOLS` | no | comma-separated tool names to leave out, e.g. the delete tools |
| `RAGFLOW_UPLOAD_ROOTS` | no | directories `file_path` may read from, separated by `:` (`;` on Windows) |
| `RAGFLOW_TIMEOUT` | no | HTTP timeout in seconds, default 120 |

## Running

The server speaks MCP over stdio.

```bash
uvx --from git+https://github.com/feci/ragflow-mcp-server ragflow-kb-mcp
```

Claude Code:

```bash
claude mcp add ragflow-kb \
  -e RAGFLOW_BASE_URL=https://ragflow.example.com \
  -e RAGFLOW_API_KEY=... \
  -e RAGFLOW_UPLOAD_ROOTS="$HOME/Documents" \
  -- uvx --from git+https://github.com/feci/ragflow-mcp-server ragflow-kb-mcp
```

If RAGFlow's built-in MCP endpoint is also connected, you can avoid duplicate tool names
with `RAGFLOW_DISABLED_TOOLS=ragflow_list_datasets,ragflow_retrieval`.

## RAGFlow 1.0 notes

These differences from the published HTTP reference were checked against a live 1.0.0-rc2
instance and its source:

- Dataset creation needs `parse_type: 1` together with `parser_id` (e.g. `general`), and
  rejects `description` and `chunk_method`. The server sets the description with a follow-up
  update.
- Parsing state is reported in `ingestion_status` (`UNSTART`, `SCHEDULED`, `RUNNING`,
  `COMPLETED`, `FAILED`, `STOPPED`) together with `progress`.
- Listing documents with a `name` or `id` filter that matches nothing returns error code 102
  ("you don't own the document") instead of an empty list. The server reports it as an empty
  result.
- Upload with `?type=web` downloads any URL (a file or a page) on the RAGFlow side, so the MCP
  server never fetches remote content itself.

## Development

```bash
uv sync
uv run pytest
```

## License

MIT
