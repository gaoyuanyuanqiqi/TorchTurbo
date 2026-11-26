"""Docker registry authentication, manifests, and alias promotion."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

DOCKER_V2 = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_LIST_V2 = "application/vnd.docker.distribution.manifest.list.v2+json"
SUPPORTED_MANIFEST_TYPES = (DOCKER_V2, DOCKER_LIST_V2)
MANIFEST_ACCEPT = ", ".join(SUPPORTED_MANIFEST_TYPES)
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RegistryError(RuntimeError):
    """Raised when registry state cannot be determined or changed safely."""


@dataclass(frozen=True)
class Manifest:
    digest: str
    media_type: str


def verify_manifest(manifest: Manifest) -> None:
    if manifest.media_type not in SUPPORTED_MANIFEST_TYPES:
        raise RegistryError(
            "registry returned unsupported manifest media type "
            f"{manifest.media_type!r}; expected Docker v2 manifest or manifest list"
        )


class Registry:
    def __init__(self, repository: str, *,
                 token_fetcher: Callable[[str, str], str] | None = None,
                 timeout: int = 30) -> None:
        host, separator, repo_path = repository.partition("/")
        if not separator or not host or not repo_path:
            raise RegistryError(f"invalid repository: {repository!r}")
        self.repository = repository
        self.host = host
        self.repo_path = repo_path
        self.timeout = timeout
        self._token_fetcher = token_fetcher or self._swr_token
        self._token: str | None = None

    def _swr_token(self, host: str, repo: str) -> str:
        url = (f"https://{host}/swr/auth/v2/registry/auth/"
               f"?service=dockyard&scope=repository:{repo}:pull")
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return str(json.loads(response.read().decode()).get("token", ""))
        except (urllib.error.URLError, ValueError, OSError):
            return ""

    def _ensure_token(self) -> str:
        if self._token is None:
            self._token = self._token_fetcher(self.host, self.repo_path)
        return self._token

    def find_manifest(self, reference: str) -> Manifest | None:
        token = self._ensure_token()
        if not token:
            raise RegistryError(f"cannot obtain registry token for {self.repository}")
        request = urllib.request.Request(
            f"https://{self.host}/v2/{self.repo_path}/manifests/"
            f"{urllib.parse.quote(reference, safe=':')}",
            headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                raw = json.loads(body.decode() or "{}")
                media_type = (raw.get("mediaType") or
                              response.headers.get("Content-Type", "").split(";", 1)[0])
                digest = response.headers.get("Docker-Content-Digest", "")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RegistryError(
                f"registry manifest read failed for {reference}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, ValueError, OSError) as exc:
            raise RegistryError(
                f"registry manifest read failed for {reference}: {exc}") from exc
        digest = digest or "sha256:" + hashlib.sha256(body).hexdigest()
        return Manifest(digest=digest, media_type=media_type)

    def require_manifest(self, reference: str) -> Manifest:
        manifest = self.find_manifest(reference)
        if manifest is None:
            raise RegistryError(f"registry reference does not exist: {reference}")
        return manifest

    def promote_to_alias(self, alias: str, digest: str) -> None:
        if not _SHA256_DIGEST.fullmatch(digest):
            raise RegistryError(
                f"promote requires sha256:<64 lowercase hex>, got {digest!r}"
            )
        command = ["docker", "buildx", "imagetools", "create", "-t",
                   f"{self.repository}:{alias}", f"{self.repository}@{digest}"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RegistryError(f"promote failed: {detail}")
