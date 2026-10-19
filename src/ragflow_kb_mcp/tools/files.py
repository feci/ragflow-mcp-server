"""File manager, runtime attachments, workspace commits and file versions."""

from __future__ import annotations

import mimetypes
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import Field

from ..common import (
    ID_PATTERN,
    DEFAULT_MAX_CHARS,
    DELETE,
    READ,
    UPLOAD,
    WRITE,
    Context,
    FileId,
    MaxChars,
    Page,
    describe_bytes,
    drop_none,
    non_empty,
    pick,
    read_bytes_source,
)

MAX_DIFF_CHARS = 4000
COMMIT_OPERATIONS = {"add", "modify", "delete", "rename"}

FilePageSize = Annotated[int, Field(ge=1, le=100, description="Items per page (RAGFlow caps it at 100)")]
ParentId = Annotated[
    str | None,
    Field(description="Folder ID; omit for the root folder of the file manager"),
]
WorkspaceArg = Annotated[
    str | None,
    Field(
        description="Workspace = a folder in the file manager; pass its folder ID. "
        "Use dataset_id instead to address the folder that holds a dataset's files."
    ),
]
DatasetArg = Annotated[
    str | None,
    Field(description="Dataset ID; resolves to the dataset's folder. Use instead of workspace_id."),
]
CommitId = Annotated[str, Field(pattern=ID_PATTERN, description="Commit ID")]
TextSource = Annotated[str | None, Field(description="UTF-8 text content")]
Base64Source = Annotated[
    str | None, Field(description="File bytes as base64 (small files only)")
]
PathSource = Annotated[
    str | None,
    Field(description="Path on the MCP server host, inside an allowed upload root"),
]


def _file(f: dict[str, Any]) -> dict[str, Any]:
    out = pick(
        f,
        "id",
        "name",
        "type",
        "size",
        "parent_id",
        "source_type",
        "has_child_folder",
        "create_time",
        "update_time",
    )
    if f.get("kbs_info"):
        out["datasets"] = f["kbs_info"]
    return out


def _folder_ref(f: Any) -> Any:
    return pick(f, "id", "name", "parent_id") if isinstance(f, dict) else f


def _commit(c: dict[str, Any]) -> dict[str, Any]:
    return pick(
        c, "id", "folder_id", "parent_id", "message", "author_id", "file_count", "create_time"
    )


def _commit_item(item: dict[str, Any]) -> dict[str, Any]:
    out = pick(
        item, "file_id", "file_name", "operation", "old_name", "new_name", "old_hash", "new_hash"
    )
    diff = item.get("diff")
    if diff:
        out["diff"] = diff[:MAX_DIFF_CHARS]
        if len(diff) > MAX_DIFF_CHARS:
            out["diff_truncated"] = True
    return out


def _workspace_path(workspace_id: str | None, dataset_id: str | None) -> str:
    if bool(workspace_id) == bool(dataset_id):
        raise ValueError("pass exactly one of workspace_id or dataset_id")
    return f"/workspaces/{workspace_id}" if workspace_id else f"/datasets/{dataset_id}"


def register(ctx: Context) -> None:
    client = ctx.client
    tool = ctx.tool

    # File manager

    @tool(READ)
    async def ragflow_list_files(
        parent_id: ParentId = None,
        keywords: Annotated[str | None, Field(description="Match file names")] = None,
        page: Page = 1,
        page_size: FilePageSize = 30,
        orderby: Annotated[str, Field(description="Sort field, e.g. create_time or name")] = "create_time",
        desc: bool = True,
    ) -> dict[str, Any]:
        """List files and folders in a file-manager folder (root when parent_id is omitted), plus the folder above it."""
        data = await client.call(
            "GET",
            "/files",
            params={
                "parent_id": parent_id,
                "keywords": keywords,
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": desc,
            },
        )
        data = data or {}
        files = data.get("files") or []
        return {
            "total": data.get("total", len(files)),
            # RAGFlow's parent_folder is the folder above the listed one.
            "parent_folder": _folder_ref(data.get("parent_folder")),
            "files": [_file(f) for f in files],
        }

    @tool(UPLOAD)
    async def ragflow_upload_file(
        filename: Annotated[
            str | None,
            Field(description="File name; required for text and content_base64"),
        ] = None,
        text: TextSource = None,
        content_base64: Base64Source = None,
        file_path: PathSource = None,
        parent_id: ParentId = None,
    ) -> dict[str, Any]:
        """Upload a file into the file manager (not into a dataset); link it to datasets later if needed."""
        blob, name = read_bytes_source(
            ctx, text=text, content_base64=content_base64, file_path=file_path, filename=filename
        )
        if not name:
            raise ValueError("filename is required for text and content_base64 uploads")
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        data = await client.call(
            "POST",
            "/files",
            data=drop_none({"parent_id": parent_id}),
            files={"file": (name, blob, mime)},
        )
        items = data if isinstance(data, list) else [data]
        return {"files": [_file(f) for f in items if isinstance(f, dict)]}

    @tool(WRITE)
    async def ragflow_create_folder(
        name: Annotated[str, Field(description="Name; must not contain '/'")],
        parent_id: ParentId = None,
        type: Annotated[
            Literal["folder", "virtual"],
            Field(description="folder, or virtual for an empty placeholder file"),
        ] = "folder",
    ) -> dict[str, Any]:
        """Create a folder (or an empty virtual file) in the file manager; a taken name gets a numbered suffix."""
        data = await client.call(
            "POST", "/files", json=drop_none({"name": name, "parent_id": parent_id, "type": type})
        )
        return _file(data or {})

    @tool(READ)
    async def ragflow_get_parent_folder(file_id: FileId) -> dict[str, Any]:
        """Get the folder that directly contains a file or folder."""
        # The handler reads the ID from the query string, not from the path.
        data = await client.call("GET", f"/files/{file_id}/parent", params={"file_id": file_id})
        return {"parent_folder": _folder_ref((data or {}).get("parent_folder"))}

    @tool(READ)
    async def ragflow_get_file_ancestors(file_id: FileId) -> dict[str, Any]:
        """List all folders above a file or folder, up to the root."""
        data = await client.call("GET", f"/files/{file_id}/ancestors")
        folders = (data or {}).get("parent_folders") or []
        return {"parent_folders": [_folder_ref(f) for f in folders]}

    @tool(READ)
    async def ragflow_download_file(
        file_id: FileId,
        max_chars: MaxChars = DEFAULT_MAX_CHARS,
        include_base64: Annotated[
            bool, Field(description="Return binary content as base64 (up to 5 MB)")
        ] = False,
    ) -> dict[str, Any]:
        """Read a file from the file manager. Text comes back as text; binary only on request, as base64."""
        blob, content_type = await client.download(f"/files/{file_id}")
        return describe_bytes(blob, content_type, max_chars, include_base64)

    @tool(WRITE)
    async def ragflow_move_files(
        file_ids: Annotated[list[str], Field(description="Files or folders to move or rename")],
        dest_folder_id: Annotated[
            str | None, Field(description="Destination folder; omit to rename in place")
        ] = None,
        new_name: Annotated[
            str | None,
            Field(description="New name for a single item; the file extension cannot change"),
        ] = None,
    ) -> dict[str, Any]:
        """Move files or folders to another folder, rename one item, or both."""
        ids = non_empty("file_ids", file_ids)
        if not dest_folder_id and not new_name:
            raise ValueError("pass dest_folder_id, new_name or both")
        if new_name and len(ids) > 1:
            raise ValueError("new_name works only with a single file")
        await client.call(
            "POST",
            "/files/move",
            json=drop_none(
                {"src_file_ids": ids, "dest_file_id": dest_folder_id, "new_name": new_name}
            ),
        )
        return {"moved": ids, "dest_folder_id": dest_folder_id, "new_name": new_name}

    @tool(DELETE)
    async def ragflow_delete_files(
        file_ids: Annotated[list[str], Field(description="Files or folders to delete")],
    ) -> dict[str, Any]:
        """Permanently delete files or folders (with their contents) from the file manager."""
        ids = non_empty("file_ids", file_ids)
        await client.call("DELETE", "/files", json={"ids": ids})
        return {"deleted": ids}

    @tool(DELETE)
    async def ragflow_link_files_to_datasets(
        file_ids: Annotated[
            list[str], Field(description="Files to link; a folder links every file inside it")
        ],
        dataset_ids: Annotated[list[str], Field(description="Target datasets")],
        mode: Annotated[
            Literal["add", "replace"],
            Field(
                description="add keeps existing links; replace also removes the documents these "
                "files have in datasets not listed (an empty dataset_ids list removes all)"
            ),
        ] = "add",
    ) -> dict[str, Any]:
        """Turn file-manager files into documents of datasets. Runs in the background and does not parse."""
        ids = non_empty("file_ids", file_ids)
        if not dataset_ids and mode != "replace":
            raise ValueError("dataset_ids must list at least one ID (or use mode=replace to unlink)")
        await client.call(
            "POST",
            "/files/link-to-datasets",
            params={"mode": mode},
            json={"file_ids": ids, "kb_ids": dataset_ids},
        )
        return {
            "linked": ids,
            "dataset_ids": dataset_ids,
            "mode": mode,
            "note": "conversion runs in the background; check ragflow_list_documents, then "
            "start ragflow_parse_documents",
        }

    @tool(UPLOAD)
    async def ragflow_upload_attachment(
        filename: Annotated[
            str | None,
            Field(description="File name; required for text and content_base64"),
        ] = None,
        text: TextSource = None,
        content_base64: Base64Source = None,
        file_path: PathSource = None,
        url: Annotated[
            str | None, Field(description="http(s) URL that RAGFlow downloads instead")
        ] = None,
    ) -> dict[str, Any]:
        """Store a file as a runtime attachment for chat or agent conversations; it joins no dataset."""
        if url is not None:
            if any(v is not None for v in (text, content_base64, file_path)):
                raise ValueError("pass url alone or one of text, content_base64 or file_path")
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("url must be an http or https URL")
            data = await client.call("POST", "/documents/upload", params={"url": url})
        else:
            blob, name = read_bytes_source(
                ctx, text=text, content_base64=content_base64, file_path=file_path, filename=filename
            )
            if not name:
                raise ValueError("filename is required for text and content_base64 uploads")
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
            data = await client.call(
                "POST", "/documents/upload", files={"file": (name, blob, mime)}
            )
        if isinstance(data, list):
            data = data[0] if data else {}
        return pick(data or {}, "id", "name", "size", "extension", "mime_type", "created_at")

    # Workspaces: every file-manager folder can carry a commit history.

    @tool(DELETE)
    async def ragflow_create_commit(
        message: Annotated[str, Field(description="Commit message")],
        files: Annotated[
            list[dict[str, str]],
            Field(
                description='Changes, each {"file_id", "operation", ...}: operation add or modify '
                'needs "file_name" and "content" (text that becomes the file\'s content), delete '
                'needs nothing more, rename needs "old_name" and "new_name". The file_id must be '
                "an existing file in the workspace."
            ),
        ],
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
    ) -> dict[str, Any]:
        """Commit changes to a workspace folder: writes file content, renames, or deletes files, and records history."""
        base = _workspace_path(workspace_id, dataset_id)
        if not files:
            raise ValueError("files must list at least one change")
        for change in files:
            op = change.get("operation")
            if op not in COMMIT_OPERATIONS:
                raise ValueError(f"operation must be one of {sorted(COMMIT_OPERATIONS)}, got {op!r}")
            if not change.get("file_id"):
                raise ValueError("every change needs a file_id")
            if op in ("add", "modify") and "content" not in change:
                raise ValueError(f"{op} needs content")
            if op == "rename" and not change.get("new_name"):
                raise ValueError("rename needs new_name")
        data = await client.call("POST", f"{base}/commits", json={"message": message, "files": files})
        return _commit(data or {})

    @tool(READ)
    async def ragflow_list_commits(
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
        page: Page = 1,
        page_size: Annotated[int, Field(ge=1, le=1000)] = 15,
        desc: bool = True,
    ) -> dict[str, Any]:
        """List the commit history of a workspace folder, newest first by default."""
        base = _workspace_path(workspace_id, dataset_id)
        data = await client.call(
            "GET", f"{base}/commits", params={"page": page, "page_size": page_size, "desc": desc}
        )
        data = data or {}
        commits = data.get("commits") or []
        return {"total": data.get("total", len(commits)), "commits": [_commit(c) for c in commits]}

    @tool(READ)
    async def ragflow_get_commit(
        commit_id: CommitId,
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
    ) -> dict[str, Any]:
        """Get one commit with the list of file changes it made."""
        base = _workspace_path(workspace_id, dataset_id)
        data = await client.call("GET", f"{base}/commits/{commit_id}") or {}
        result = _commit(data)
        if "files" in data:
            result["files"] = [_commit_item(i) for i in data.get("files") or []]
        else:
            result.update({k: v for k, v in data.items() if k not in result})
        return result

    @tool(READ)
    async def ragflow_list_commit_files(
        commit_id: CommitId,
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
    ) -> dict[str, Any]:
        """List the file changes recorded in one commit."""
        base = _workspace_path(workspace_id, dataset_id)
        data = await client.call("GET", f"{base}/commits/{commit_id}/files") or []
        return {"files": [_commit_item(i) for i in data]}

    @tool(READ)
    async def ragflow_diff_commits(
        from_commit_id: CommitId,
        to_commit_id: CommitId,
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
    ) -> dict[str, Any]:
        """Compare two commits of a workspace and list which files were added, modified, deleted or renamed."""
        base = _workspace_path(workspace_id, dataset_id)
        data = await client.call(
            "GET", f"{base}/commits/diff", params={"from": from_commit_id, "to": to_commit_id}
        )
        return {"changes": [_commit_item(i) for i in data or []]}

    @tool(READ)
    async def ragflow_get_uncommitted_changes(
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
    ) -> dict[str, Any]:
        """Show files in a workspace that changed since its last commit (like git status)."""
        base = _workspace_path(workspace_id, dataset_id)
        data = await client.call("GET", f"{base}/changes")
        return {"changes": [_commit_item(i) for i in data or []]}

    @tool(READ)
    async def ragflow_get_commit_tree(
        commit_id: CommitId,
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
    ) -> dict[str, Any]:
        """Get the folder tree of a workspace as it was at one commit."""
        base = _workspace_path(workspace_id, dataset_id)
        return await client.call("GET", f"{base}/commits/{commit_id}/tree") or {}

    @tool(READ)
    async def ragflow_get_commit_file_content(
        commit_id: CommitId,
        file_id: FileId,
        workspace_id: WorkspaceArg = None,
        dataset_id: DatasetArg = None,
        max_chars: MaxChars = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        """Read a file's content as it was at one commit."""
        base = _workspace_path(workspace_id, dataset_id)
        data = await client.call("GET", f"{base}/commits/{commit_id}/files/{file_id}/content")
        content = (data or {}).get("content") or ""
        return {"content": content[:max_chars], "truncated": len(content) > max_chars}

    @tool(READ)
    async def ragflow_get_file_versions(file_id: FileId) -> dict[str, Any]:
        """List the commits that changed one file, with operation, hash and message."""
        data = await client.call("GET", f"/workspace-files/{file_id}/versions")
        return {"versions": data or []}
