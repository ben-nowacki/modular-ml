import json
import pathlib
import zipfile

from modularml.core.io.artifacts import Artifact
from modularml.core.io.conventions import MML_FILE_EXTENSION


def inspect_packaged_code(path: pathlib.Path) -> dict[str, str]:
    """
    Inspect a ModularML .mml artifact and extract all packaged source code.

    Description:
        Recursively scans the top-level artifact for any nested `.mml` files,
        then inspects each artifact independently to extract packaged Python
        source files without executing them.

        This function is safe to call on untrusted artifacts.

    Args:
        path (Path):
            Path to a `.mml` file.

    Returns:
        Dict[str, str]:
            Mapping from source_ref (e.g. "code/my_scaler.py:MyScaler")
            to the corresponding source code text.

    Raises:
        FileNotFoundError:
            If the artifact file does not exist.
        ValueError:
            If the file is not a valid ModularML artifact.

    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    all_sources: dict[str, str] = {}
    with zipfile.ZipFile(path, "r") as root_zip:
        # 1. Inspect root artifact
        all_sources.update(
            _inspect_mml_zip(
                mml_zip=root_zip,
                root_zip=root_zip,
                label="<root>",
            ),
        )

        # 2. Find and inspect any nested artifacts (.mml) files
        for name in root_zip.namelist():
            if not name.endswith(MML_FILE_EXTENSION):
                continue

            with root_zip.open(name) as nested_bytes, zipfile.ZipFile(nested_bytes) as nested_zip:
                all_sources.update(
                    _inspect_mml_zip(
                        mml_zip=nested_zip,
                        root_zip=root_zip,
                        label=name,
                    ),
                )

    return all_sources


def _inspect_mml_zip(
    *,
    mml_zip: zipfile.ZipFile,
    root_zip: zipfile.ZipFile,
    label: str,
) -> dict[str, str]:
    """Inspect a single opened `.mml` zip archive."""
    sources: dict[str, str] = {}

    # Validate artifact.json
    try:
        with mml_zip.open("artifact.json") as f:
            artifact = Artifact.from_json(json.load(f))
    except KeyError as exc:
        msg = f"Invalid ModularML artifact ({label}): missing artifact.json"
        raise ValueError(msg) from exc

    # Extract config if present
    config_rel_path = artifact.files.get("config")
    if not config_rel_path or not config_rel_path.endswith(".json"):
        return sources

    try:
        with mml_zip.open(config_rel_path) as f:
            config = json.load(f)
    except Exception as exc:
        msg = f"Failed to read config in {label}"
        raise RuntimeError(msg) from exc

    # Recursively scan config for packaged symbols
    def collect(obj: object) -> None:
        if isinstance(obj, dict):
            if obj.get("policy") == "packaged" and "source_ref" in obj:
                source_ref = obj["source_ref"]
                file_path, _ = source_ref.split(":", 1)

                try:
                    # Code paths are relative to root_zip, not artifact zip
                    with root_zip.open(file_path) as src:
                        sources[source_ref] = src.read().decode("utf-8")
                except KeyError as exc:
                    msg = f"Packaged source '{file_path}' not found in {label}"
                    raise FileNotFoundError(msg) from exc

            for v in obj.values():
                collect(v)

        elif isinstance(obj, list):
            for v in obj:
                collect(v)

    collect(config)
    return sources
