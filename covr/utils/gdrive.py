from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
_FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveUploader:  # pylint: disable=no-member
    def __init__(self, folder_id: str, credentials_path: str):
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=_SCOPES
        )
        self._service = build("drive", "v3", credentials=creds)
        self._folder_id = folder_id
        self._folder_cache: dict[tuple[str, str], str] = {}  # (parent_id, name) -> id

    def _get_or_create_folder(self, name: str, parent_id: str) -> str:
        key = (parent_id, name)
        if key in self._folder_cache:
            return self._folder_cache[key]
        results = (
            self._service.files()
            .list(
                q=(
                    f"'{parent_id}' in parents"
                    f" and name='{name}'"
                    f" and mimeType='{_FOLDER_MIME}'"
                    f" and trashed=false"
                ),
                fields="files(id)",
            )
            .execute()
        )
        files = results.get("files", [])
        if files:
            folder_id = files[0]["id"]
        else:
            folder_id = (
                self._service.files()
                .create(
                    body={
                        "name": name,
                        "mimeType": _FOLDER_MIME,
                        "parents": [parent_id],
                    },
                    fields="id",
                )
                .execute()["id"]
            )
        self._folder_cache[key] = folder_id
        return folder_id

    def uploaded_names(self) -> set[str]:
        """Return filenames already present in the Drive root folder."""
        results = (
            self._service.files()
            .list(
                q=f"'{self._folder_id}' in parents and trashed=false",
                fields="files(name)",
                pageSize=1000,
            )
            .execute()
        )
        return {f["name"] for f in results.get("files", [])}

    def upload(self, local_path: Path, relative_path: Path) -> None:
        """Upload local_path into the Drive folder."""
        parent_id = self._folder_id
        for part in relative_path.parts[:-1]:
            parent_id = self._get_or_create_folder(part, parent_id)
        self._service.files().create(
            body={"name": local_path.name, "parents": [parent_id]},
            media_body=MediaFileUpload(str(local_path), resumable=True),
        ).execute()
