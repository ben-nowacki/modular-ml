import json
import pathlib
import zipfile

from modularml.core.io.artifacts import Artifact


def inspect_packaged_code(path: pathlib.Path) -> dict[str, str]:
    """
    Inspect a ModularML .mml artifact and extract all packaged source code.

    Description:
        Opens the ZIP-based ModularML artifact, parses its metadata, and
        returns all bundled Python source files without executing them.
        This is safe to call on untrusted artifacts.

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

    sources: dict[str, str] = {}
    with zipfile.ZipFile(path, "r") as zf:
        # Validate artifact.json
        try:
            with zf.open("artifact.json") as f:
                artifact: Artifact = Artifact.from_json(json.load(f))
        except KeyError as exc:
            raise ValueError("Not a valid ModularML artifact (missing artifact.json).") from exc

        # Extract config from artifact
        config_rel_path = artifact.files.get("config")
        if not config_rel_path:
            return {}

        file_config: pathlib.Path = path / config_rel_path

        # Attempt to load config based on suffix
        config = {}
        if pathlib.Path(file_config).suffix.lower() == ".json":
            try:
                with zf.open(config_rel_path) as f:
                    config = json.load(f)
            except Exception as e:
                msg = f"Failed to read 'config.json'. {e}"
                raise RuntimeError(msg) from e
        else:
            msg = f"Config with suffix '{file_config.suffix}' is not supported yet."
            raise NotImplementedError(msg)

        # Recursively scan for packaged ClassSpecs
        def collect(obj: object) -> None:
            if isinstance(obj, dict):
                # Detect ClassSpec-like dicts
                if obj.get("policy") == "packaged" and "source_ref" in obj:
                    source_ref = obj["source_ref"]
                    file_path, _ = source_ref.split(":", 1)

                    try:
                        with zf.open(file_path) as src:
                            sources[source_ref] = src.read().decode("utf-8")
                    except KeyError as exc:
                        msg = f"Packaged source '{file_path}' not found in artifact."
                        raise FileNotFoundError(msg) from exc

                for v in obj.values():
                    collect(v)

            elif isinstance(obj, list):
                for v in obj:
                    collect(v)

        collect(config)

    return sources
