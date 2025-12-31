import json
import pathlib
import zipfile

from modularml.core.io.artifacts import Artifact


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
        # ================================================
        # Step 1: find *all* .mml files inside the archive (including root)
        # ================================================
        mml_files = {pathlib.Path(name) for name in root_zip.namelist() if name.endswith(".mml")}

        # Root artifact itself may not appear as a nested .mml
        mml_files.add(pathlib.Path(path.name))

        # ================================================
        # Step 2: inspect each .mml independently
        # ================================================
        for mml_rel_path in mml_files:
            all_sources.update(
                _inspect_single_mml(
                    zip_file=root_zip,
                    mml_path=mml_rel_path,
                ),
            )

    return all_sources


def _inspect_single_mml(
    *,
    zip_file: zipfile.ZipFile,
    mml_path: pathlib.Path,
) -> dict[str, str]:
    """Inspect a single `.mml` artifact stored inside a zip file."""
    sources: dict[str, str] = {}
    try:
        with zip_file.open(str(mml_path)) as mml_bytes, zipfile.ZipFile(mml_bytes) as mml_zip:
            # Validate artifact.json
            try:
                with mml_zip.open("artifact.json") as f:
                    artifact = Artifact.from_json(json.load(f))
            except KeyError as e:
                msg = f"Invalid ModularML artifact: {mml_path} (missing artifact.json)"
                raise ValueError(msg) from e

            # Pull config (if any)
            config_rel_path = artifact.files.get("config")
            if not config_rel_path:
                return {}

            if pathlib.Path(config_rel_path).suffix.lower() != ".json":
                return {}

            try:
                with mml_zip.open(config_rel_path) as f:
                    config = json.load(f)
            except Exception as exc:
                msg = f"Failed to read config in {mml_path}"
                raise RuntimeError(msg) from exc

            # Recursively scan config for packaged ClassSpecs
            def collect(obj: object) -> None:
                if isinstance(obj, dict):
                    if obj.get("policy") == "packaged" and "source_ref" in obj:
                        source_ref = obj["source_ref"]
                        file_path, _ = source_ref.split(":", 1)

                        try:
                            with zip_file.open(file_path) as src:
                                sources[source_ref] = src.read().decode("utf-8")
                        except KeyError as exc:
                            msg = f"Packaged source '{file_path}' not found in {mml_path}"
                            raise FileNotFoundError(msg) from exc

                    for v in obj.values():
                        collect(v)

                elif isinstance(obj, list):
                    for v in obj:
                        collect(v)

            collect(config)

    except KeyError:
        # mml_path not actually present as a nested file (e.g. root)
        pass

    return sources

    # with zipfile.ZipFile(path, "r") as zf:
    #     # Validate artifact.json
    #     try:
    #         with zf.open("artifact.json") as f:
    #             artifact: Artifact = Artifact.from_json(json.load(f))
    #     except KeyError as exc:
    #         raise ValueError("Not a valid ModularML artifact (missing artifact.json).") from exc

    #     # Extract config from artifact
    #     config_rel_path = artifact.files.get("config")
    #     if not config_rel_path:
    #         return {}

    #     file_config: pathlib.Path = path / config_rel_path

    #     # Attempt to load config based on suffix
    #     config = {}
    #     if pathlib.Path(file_config).suffix.lower() == ".json":
    #         try:
    #             with zf.open(config_rel_path) as f:
    #                 config = json.load(f)
    #         except Exception as e:
    #             msg = f"Failed to read 'config.json'. {e}"
    #             raise RuntimeError(msg) from e
    #     else:
    #         msg = f"Config with suffix '{file_config.suffix}' is not supported yet."
    #         raise NotImplementedError(msg)

    #     # Recursively scan for packaged ClassSpecs
    #     def collect(obj: object) -> None:
    #         if isinstance(obj, dict):
    #             # Detect ClassSpec-like dicts
    #             if obj.get("policy") == "packaged" and "source_ref" in obj:
    #                 source_ref = obj["source_ref"]
    #                 file_path, _ = source_ref.split(":", 1)

    #                 try:
    #                     with zf.open(file_path) as src:
    #                         sources[source_ref] = src.read().decode("utf-8")
    #                 except KeyError as exc:
    #                     msg = f"Packaged source '{file_path}' not found in artifact."
    #                     raise FileNotFoundError(msg) from exc

    #             for v in obj.values():
    #                 collect(v)

    #         elif isinstance(obj, list):
    #             for v in obj:
    #                 collect(v)

    #     collect(config)

    # return sources
