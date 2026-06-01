"""Security primitives: PHI-at-rest encryption and session authorization.

Two concerns are handled here, both of which were entirely absent from the
prototype:

1. **Field-level encryption** of cached document content. Retrieved context is
   derived from patient input and is therefore PHI. We encrypt it at the
   application layer (envelope-style, via Fernet/AES-128-CBC+HMAC) so that a
   database compromise alone does not expose clinical text. Embeddings remain
   unencrypted because they must be searchable; this is documented as a residual
   risk (embedding-inversion) to be mitigated with DB-level encryption and access
   control.

2. **Session authorization**. ``session_id`` is client-supplied and is the only
   tenancy boundary, so it must never be trusted on its own. An ``Authorizer``
   callback lets the host application enforce that the calling principal actually
   owns / may access the requested session, closing the IDOR hole.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from .exceptions import AuthorizationError, ConfigurationError
from .logging_utils import get_logger

try:  # cryptography is a hard dep in pyproject, but degrade clearly if missing.
    from cryptography.fernet import Fernet, InvalidToken

    _CRYPTO_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when dep is absent
    _CRYPTO_AVAILABLE = False


_PLAINTEXT_PREFIX = "plain:"
_CIPHER_PREFIX = "enc:"


class ContentCipher:
    """Encrypts/decrypts cached content using a symmetric key.

    The key is read from ``STATEFUL_RAG_ENCRYPTION_KEY`` (a urlsafe-base64 32-byte
    Fernet key). If no key is configured the cipher operates in pass-through mode
    and loudly warns — this keeps local/dev usable while making it impossible to
    *accidentally* run unencrypted in production without a visible warning.
    """

    def __init__(self, key: Optional[str] = None, *, require_key: bool = False):
        self._logger = get_logger()
        key = key or os.environ.get("STATEFUL_RAG_ENCRYPTION_KEY")

        if not key:
            if require_key:
                raise ConfigurationError(
                    "STATEFUL_RAG_ENCRYPTION_KEY is required but not set"
                )
            self._fernet = None
            self._logger.warning(
                "ContentCipher running WITHOUT encryption: cached PHI will be "
                "stored in plaintext. Set STATEFUL_RAG_ENCRYPTION_KEY in production."
            )
            return

        if not _CRYPTO_AVAILABLE:
            raise ConfigurationError(
                "An encryption key was provided but the 'cryptography' package is "
                "not installed."
            )
        try:
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception as exc:  # invalid key format
            raise ConfigurationError(f"Invalid encryption key: {exc}") from exc

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    @staticmethod
    def generate_key() -> str:
        """Generate a fresh Fernet key for operators to store in their secret manager."""
        if not _CRYPTO_AVAILABLE:
            raise ConfigurationError("'cryptography' is not installed")
        return Fernet.generate_key().decode()

    def encrypt(self, plaintext: str) -> str:
        """Return an encrypted, prefix-tagged token for storage."""
        if self._fernet is None:
            return _PLAINTEXT_PREFIX + plaintext
        token = self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        return _CIPHER_PREFIX + token

    def decrypt(self, stored: str) -> str:
        """Decrypt a previously stored token, tolerating legacy/plaintext values."""
        if stored.startswith(_CIPHER_PREFIX):
            if self._fernet is None:
                raise ConfigurationError(
                    "Encountered encrypted content but no encryption key is configured."
                )
            try:
                return self._fernet.decrypt(stored[len(_CIPHER_PREFIX):].encode()).decode("utf-8")
            except InvalidToken as exc:  # wrong key / tampering
                raise ConfigurationError("Failed to decrypt content (key mismatch or tampering)") from exc
        if stored.startswith(_PLAINTEXT_PREFIX):
            return stored[len(_PLAINTEXT_PREFIX):]
        # Legacy rows written before this module existed.
        return stored


# An authorizer receives (session_id, principal) and returns True if access is allowed.
Authorizer = Callable[[str, Optional[str]], bool]

_warned_no_authorizer = False


def _warn_no_authorizer_once() -> None:
    global _warned_no_authorizer
    if not _warned_no_authorizer:
        _warned_no_authorizer = True
        get_logger().warning(
            "No authorizer configured; session access is unauthenticated. "
            "Provide an Authorizer to enforce per-principal session isolation. "
            "(This warning is shown once.)"
        )


def enforce_authorization(
    authorizer: Optional[Authorizer], session_id: str, principal: Optional[str]
) -> None:
    """Run the host-supplied authorizer, raising ``AuthorizationError`` on denial.

    When no authorizer is configured the call is permitted but logged at WARNING,
    because an unauthenticated multi-tenant cache is unsafe for clinical PHI.
    """
    if authorizer is None:
        _warn_no_authorizer_once()
        return
    try:
        allowed = authorizer(session_id, principal)
    except Exception as exc:  # an authorizer that errors must deny, not allow
        raise AuthorizationError(f"Authorization check failed: {exc}") from exc
    if not allowed:
        raise AuthorizationError(
            f"Principal {principal!r} is not authorized for session {session_id!r}"
        )
