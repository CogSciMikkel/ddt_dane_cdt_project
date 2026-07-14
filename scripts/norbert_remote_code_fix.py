"""Compatibility patch for loading NorBERT3 through spacy-transformers."""

import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModel


_original_config_from_pretrained = AutoConfig.from_pretrained
_original_model_from_config = AutoModel.from_config


def _qualify_reference(value: Any, repo_id: str) -> Any:
    """Attach the original Hugging Face repository to custom-code references."""
    if isinstance(value, str):
        return value if "--" in value else f"{repo_id}--{value}"

    if isinstance(value, list):
        return [_qualify_reference(item, repo_id) for item in value]

    return value


def _patched_config_from_pretrained(
    cls,
    pretrained_model_name_or_path,
    *args,
    **kwargs,
):
    path = Path(pretrained_model_name_or_path)

    # spacy-transformers saves the Hugging Face config as a temporary JSON file.
    if path.is_file() and path.name == "config.json":
        config_data = json.loads(path.read_text(encoding="utf-8"))

        repo_id = config_data.get("_name_or_path")

        # Fallback for this particular model if the original path was omitted.
        if not repo_id or not isinstance(repo_id, str):
            repo_id = "ltg/norbert3-xs"

        auto_map = config_data.get("auto_map")

        if isinstance(auto_map, dict):
            config_data["auto_map"] = {
                key: _qualify_reference(value, repo_id)
                for key, value in auto_map.items()
            }

            path.write_text(
                json.dumps(config_data),
                encoding="utf-8",
            )

        kwargs.setdefault("trust_remote_code", True)

    return _original_config_from_pretrained(
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    )


def _patched_model_from_config(cls, config, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    return _original_model_from_config(config, **kwargs)


AutoConfig.from_pretrained = classmethod(_patched_config_from_pretrained)
AutoModel.from_config = classmethod(_patched_model_from_config)
