# ragflow-mcp-server

An MCP server that exposes the [RAGFlow](https://github.com/infiniflow/ragflow) HTTP API
as tools: datasets and documents, parsing, chunks, metadata, retrieval, chat assistants and
their sessions, agents, memories, the file manager with workspace commits, and search apps.

RAGFlow ships its own MCP endpoint, but it only offers `ragflow_list_datasets`,
`ragflow_list_chats` and `ragflow_retrieval`. This server covers every endpoint of the
published HTTP API reference: 92 tools. It was checked tool by tool against a live
RAGFlow 1.0.0-rc2 (Go API) and against that release's source code.

The Python package and the command are called `ragflow-kb-mcp`.

## Tools

| Area | Tools |
|---|---|
| Datasets | `ragflow_list_datasets`, `ragflow_get_dataset`, `ragflow_create_dataset`, `ragflow_update_dataset`, `ragflow_delete_datasets` |
| Documents | `ragflow_list_documents`, `ragflow_upload_document`, `ragflow_update_document`, `ragflow_download_document`, `ragflow_delete_documents` |
| Parsing | `ragflow_parse_documents`, `ragflow_stop_parsing`, `ragflow_ingest_documents` |
| Chunks | `ragflow_list_chunks`, `ragflow_get_chunk`, `ragflow_add_chunk`, `ragflow_update_chunk`, `ragflow_set_chunks_availability`, `ragflow_delete_chunks` |
| Metadata | `ragflow_metadata_summary`, `ragflow_update_metadata` |
| Retrieval | `ragflow_retrieval` |
| Chat assistants | `ragflow_list_chats`, `ragflow_get_chat`, `ragflow_create_chat`, `ragflow_update_chat`, `ragflow_replace_chat_settings`, `ragflow_delete_chat`, `ragflow_delete_chats` |
| Chat sessions | `ragflow_list_chat_sessions`, `ragflow_get_chat_session`, `ragflow_create_chat_session`, `ragflow_rename_chat_session`, `ragflow_delete_chat_sessions`, `ragflow_delete_chat_message`, `ragflow_rate_chat_message` |
| Conversation | `ragflow_chat`, `ragflow_openai_chat_completion`, `ragflow_related_questions`, `ragflow_mind_map`, `ragflow_text_to_speech`, `ragflow_speech_to_text` |
| Agents | `ragflow_list_agents`, `ragflow_get_agent`, `ragflow_list_agent_templates`, `ragflow_create_agent`, `ragflow_update_agent`, `ragflow_delete_agent` |
| Agent runs | `ragflow_create_agent_session`, `ragflow_list_agent_sessions`, `ragflow_delete_agent_sessions`, `ragflow_agent_chat`, `ragflow_upload_agent_file`, `ragflow_download_agent_attachment` |
| Memories | `ragflow_list_memories`, `ragflow_get_memory_config`, `ragflow_create_memory`, `ragflow_update_memory`, `ragflow_delete_memory` |
| Memory messages | `ragflow_list_memory_messages`, `ragflow_add_memory_message`, `ragflow_search_memory_messages`, `ragflow_recent_memory_messages`, `ragflow_get_memory_message`, `ragflow_set_memory_message_status`, `ragflow_forget_memory_message` |
| File manager | `ragflow_list_files`, `ragflow_upload_file`, `ragflow_create_folder`, `ragflow_get_parent_folder`, `ragflow_get_file_ancestors`, `ragflow_download_file`, `ragflow_move_files`, `ragflow_delete_files`, `ragflow_link_files_to_datasets`, `ragflow_upload_attachment` |
| Workspaces | `ragflow_create_commit`, `ragflow_list_commits`, `ragflow_get_commit`, `ragflow_list_commit_files`, `ragflow_diff_commits`, `ragflow_get_uncommitted_changes`, `ragflow_get_commit_tree`, `ragflow_get_commit_file_content`, `ragflow_get_file_versions` |
| Search apps | `ragflow_list_search_apps`, `ragflow_get_search_app`, `ragflow_create_search_app`, `ragflow_update_search_app`, `ragflow_delete_search_app`, `ragflow_search_app_completion` |
| System | `ragflow_health` |

Three tools are not in the published reference but exist in the Go API and make the rest
usable: `ragflow_get_agent`, `ragflow_list_agent_templates` (a template's DSL is the easiest
start for `ragflow_create_agent`) and `ragflow_upload_agent_file`.

Conversation tools never stream: they return the complete answer. Search-app completion
only exists as a stream, so the server reads the stream and returns the final answer.

### Safety

- Every tool has MCP annotations. 39 are read-only. The 16 marked `destructiveHint` cover
  the deletes plus calls that destroy data as a side effect: ingest rerun/clear, linking
  files to datasets, and workspace commits.
- Bulk deletes take explicit IDs. The API's `delete_all` switches are not exposed.
- `ragflow_update_metadata` without `document_ids` or `metadata_condition` would change every
  document, so it asks for `all_documents: true`.
- IDs are validated before they go into a request path, so a crafted ID cannot reach a
  different endpoint.
- `RAGFLOW_DISABLED_TOOLS` removes tools you do not want to offer.

### Uploading

`ragflow_upload_document` takes exactly one source:

- `text`: UTF-8 content such as Markdown notes. Needs a `filename` with an extension.
- `content_base64`: file bytes. The bytes pass through the model's context, so use this
  only for small files.
- `url`: a PDF or a web page that RAGFlow downloads itself (see the note on names below).
- `file_path`: a file on the machine that runs this server. This is off until the operator
  lists allowed directories in `RAGFLOW_UPLOAD_ROOTS`. Paths are resolved, including
  symlinks, before the check.

The file extension decides which RAGFlow parser runs. Parsing starts right after the
upload unless `parse` is false. Set `wait_seconds` (up to 600) to wait until
`ingestion_status` is final. An existing document with the same name makes the upload fail
unless you pass `replace_existing`; then the new file is uploaded first and the old one is
deleted afterwards.

For `url` uploads RAGFlow strips every character except letters, digits, `-` and `_` from
the name, the dot of the extension included, and adds back only `.pdf` or `.html`. So
`example page` is stored as `examplepage.html`, and any other file type from a URL (Markdown,
text, Office files) is rejected. Upload those with `text` or `content_base64` instead. The
duplicate check compares the names RAGFlow would store.

`ragflow_upload_file`, `ragflow_upload_attachment`, `ragflow_upload_agent_file` and
`ragflow_speech_to_text` accept `text`, `content_base64` or `file_path` the same way.

## Configuration

| Variable | Required | Meaning |
|---|---|---|
| `RAGFLOW_BASE_URL` | yes | e.g. `https://ragflow.example.com` (`/api/v1` is added if missing) |
| `RAGFLOW_API_KEY` | yes | RAGFlow API key |
| `RAGFLOW_DEFAULT_EMBEDDING_MODEL` | no | embedding model for new datasets, `model_name@factory` |
| `RAGFLOW_DISABLED_TOOLS` | no | comma-separated tool names to leave out |
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

If RAGFlow's built-in MCP endpoint is also connected, avoid duplicate tool names with
`RAGFLOW_DISABLED_TOOLS=ragflow_list_datasets,ragflow_list_chats,ragflow_retrieval`.

## RAGFlow 1.0 notes

The published HTTP reference and RAGFlow 1.0.0-rc2 differ in many places. Everything below
was checked against a live instance and the release's Go source. The tools already handle
these cases.

**Datasets and documents**
- Dataset creation needs `parse_type: 1` together with `parser_id` (e.g. `general`), and
  rejects `description` and `chunk_method`. The description is set with a follow-up update.
- `include_parsing_status` on the dataset list is accepted and ignored.
- Parsing state is `ingestion_status` (`UNSTART`, `SCHEDULED`, `RUNNING`, `COMPLETED`,
  `FAILED`, `STOPPED`). `progress` reaches 1 while the status is still `RUNNING` and before
  the chunks are written, so only the status says that parsing is finished.
- A document `name` or `id` filter without a match returns error code 102 ("you don't own
  the document") instead of an empty list.
- Upload with `?type=web` downloads the URL on the RAGFlow side, so this server never
  fetches remote content itself. The name is cleaned to `[A-Za-z0-9_-]`, which also removes
  the extension's dot; only `.pdf` and `.html` are added back, so other file types are refused
  with "this type of file has not been supported yet". Local uploads rename a duplicate to
  `name(1).md`.
- Ingest takes `doc_ids` and `run`: `"1"` starts, `"2"` cancels, any other value clears the
  chunks. It works on every dataset, not only on ingestion-pipeline ones.
- Getting a single chunk returns raw storage fields (`available_int`, `docnm_kwd`); the chunk
  list returns `available` as a boolean.

**Chat assistants and sessions**
- A question and its answer share one message ID; deleting it removes both.
- `user_id` on session create and list is ignored; the API key's owner is used.
- `PUT /chats/{id}` replaces `llm_setting` and `prompt_config`, `PATCH` merges them.
  Unknown fields are dropped silently.
- `keywords_similarity_weight` is the stored weight; `vector_similarity_weight` is a legacy
  alias, and the two add up to 1.
- `chat_id` is optional on `/chat/completions`: without it the tenant's default model
  answers with no retrieval and no stored session.
- The session list returns the full message history unless `include_history=false`.
- The OpenAI-compatible endpoint returns a plain OpenAI body without the `{code, data}`
  envelope; references need `extra_body.reference=true`.
- Related questions and mind maps work with an API key, although the reference asks for a
  login token. A mind map runs several LLM calls and can outlast proxy timeouts.

**Agents**
- Agent sessions are deleted with `?ids=a,b` in the query. The documented JSON body is
  ignored, and the API still reports success without deleting anything.
- Every agent update rewrites the `release` flag, so an update without `release=true`
  unpublishes the agent.
- Session creation takes only a name. Values for the Begin component go into the chat
  call's `inputs`. Without `session_id` every exchange is stored as a new session.
- OpenAI-compatible mode answers with a plain OpenAI body and reports errors as HTTP 4xx
  `{"error": ...}`.
- Attachment downloads are `application/octet-stream` unless `ext` is given.

**Memories**
- Creation needs exactly `name`, `memory_type`, `embd_id` and `llm_id`; the description is
  set with a follow-up update. Permissions also accept `all`.
- `memory_type` and `embd_id` cannot change once a memory has messages.
- Adding a message with an API key requires `user_id`.
- The message list's `keywords` parameter is an exact `session_id` filter. `page_size`,
  `top_n` and `limit` are capped at 100.
- Message content carries the vector as `q_4096_vec`, which the tools drop.
- Message routes use `{memory_id}:{message_id}` with an integer message ID. Forgetting only
  sets `forget_at`: the message stays listed and enabled.

**File manager and workspaces**
- `GET /files/{id}/parent` reads the ID from the `file_id` query parameter.
- `POST /files` with JSON and no `type` creates an empty virtual file, not a folder.
- `link-to-datasets` has an undocumented `mode`. Its default, `replace`, removes the files'
  documents from datasets that are not listed, so the tool defaults to `add`. Linking runs
  in the background and does not parse.
- Deleting a folder deletes its contents. A rename cannot change the file extension.
- A workspace is any file-manager folder, and its folder ID is the `workspace_id`.
- Commits rewrite the live files. `add`/`modify` store new text content, `delete`
  soft-deletes, and `rename` renames.
- `POST /documents/upload` stores a runtime attachment for conversations. It creates no
  document and no file-manager entry.
- Agent uploads and attachments cannot be deleted through the API.

**Search apps and health**
- Creation accepts only `name` and `description`. The configuration needs a follow-up `PUT`,
  which requires both `name` and `search_config` and merges the configuration server-side.
- `vector_similarity_weight` is stored as `keywords_similarity_weight = 1 − w`.
- `/system/healthz` needs no authentication, has no envelope, and answers HTTP 500 with
  per-component status when something is down.

## Development

```bash
uv sync
uv run pytest
```

The live tests create objects whose names start with `ragflow-kb-mcp-` and delete them
again. Run them against a RAGFlow you control:

```bash
RAGFLOW_LIVE_TEST=1 RAGFLOW_BASE_URL=... RAGFLOW_API_KEY=... \
RAGFLOW_DEFAULT_EMBEDDING_MODEL=model@factory RAGFLOW_TEST_LLM_ID=model@factory \
RAGFLOW_LIVE_DATASET_ID=<a parsed dataset to read from> uv run pytest tests/test_live*.py
```

## License

MIT
