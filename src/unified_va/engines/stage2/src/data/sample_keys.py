"""Sample-key helpers with backward-compatible legacy support.

Legacy key format:
  "{id}::{aspect_index}"

New key format (collision-safe across languages):
  "{lang}::{id}::{aspect_index}"
"""

from __future__ import annotations


def legacy_sample_key(sample_id: str, aspect_index: int) -> str:
    """Build the legacy key format."""
    return f"{sample_id}::{aspect_index}"


def make_sample_key(sample_id: str, aspect_index: int, lang: str | None = None) -> str:
    """Build language-aware sample key when language is available.

    Falls back to legacy format for missing/unknown language.
    """
    lang_norm = (lang or "").strip()
    if lang_norm and lang_norm.lower() != "unknown":
        return f"{lang_norm}::{sample_id}::{aspect_index}"
    return legacy_sample_key(sample_id, aspect_index)


def make_sample_key_from_sample(sample: dict) -> str:
    """Build key from an aspect-level sample dict."""
    return make_sample_key(
        sample_id=str(sample["id"]),
        aspect_index=int(sample["aspect_index"]),
        lang=sample.get("lang"),
    )


def to_legacy_sample_key(sample_key: str) -> str:
    """Convert language-aware key to legacy key if possible."""
    parts = sample_key.rsplit("::", 2)
    if len(parts) == 3:
        _, sample_id, aspect_index = parts
        return legacy_sample_key(sample_id, int(aspect_index))
    if len(parts) == 2:
        sample_id, aspect_index = parts
        return legacy_sample_key(sample_id, int(aspect_index))
    return sample_key


def candidate_sample_keys(sample_id: str, aspect_index: int, lang: str | None = None) -> list[str]:
    """Return lookup candidates: [new_key, legacy_key] (deduplicated)."""
    primary = make_sample_key(sample_id, aspect_index, lang=lang)
    legacy = legacy_sample_key(sample_id, aspect_index)
    if primary == legacy:
        return [primary]
    return [primary, legacy]


def candidate_sample_keys_from_sample(sample: dict) -> list[str]:
    """Return key candidates from an aspect-level sample dict."""
    return candidate_sample_keys(
        sample_id=str(sample["id"]),
        aspect_index=int(sample["aspect_index"]),
        lang=sample.get("lang"),
    )
