from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from modularml.core.data.featureset import FeatureSet
from modularml.core.data.schema_constants import DOMAIN_FEATURES, DOMAIN_TAGS, DOMAIN_TARGETS, REP_TRANSFORMED
from modularml.core.io.handlers.handler import TypeHandler
from modularml.core.io.serialization_policy import SerializationPolicy
from modularml.core.references.data_reference import DataReference
from modularml.utils.data.pyarrow_data import hash_pyarrow_table

if TYPE_CHECKING:
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.data.sample_collection import SampleCollection
    from modularml.core.io.serializer import LoadContext, SaveContext
    from modularml.core.splitting.splitter_record import SplitterRecord
    from modularml.core.transforms.scaler_record import ScalerRecord


class FeatureSetHandler(TypeHandler[FeatureSet]):
    """TypeHandler for FeatureSet objects."""

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # FeatureSet Encoding
    # ================================================
    def encode(
        self,
        obj: FeatureSet,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (FeatureSet):
                Object instance to encode.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str | None]: Mapping of "config" and "state" keys to saved files.

        """
        file_mapping = self.encode_config(obj=obj, save_dir=save_dir, ctx=ctx)
        file_mapping.update(self.encode_state(obj=obj, save_dir=save_dir, ctx=ctx))
        return file_mapping

    def encode_config(
        self,
        obj: FeatureSet,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (FeatureSet):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        return super().encode_config(obj=obj, save_dir=save_dir, ctx=ctx, config_rel_path=self.config_rel_path)

    def encode_state(
        self,
        obj: FeatureSet,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (FeatureSet):
                Object to encode state for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

        """
        # FeatureSet.get_state() -> set_state() works within the same runtime environment
        # To preserve underlying Scaler/Splitter classes that may be user-defined, we need
        # to separately save the Scaler/Splitter classes

        # The featureset.fs.mml file is structure as follows:
        # ├── artifact.json             <- FeatureSet identity
        # ├── config.json               <- FeatureSet.get_config()
        # │
        # ├── sample_collection.arrow   <- pyarrow table + schema
        # ├── split_configs.json        <- Splits (FeatureSetView)
        # │
        # ├── scaler_records/
        # │   ├── record_000.json       <- ScalerRecord config
        # │   ├── record_001.json
        # │   └── record_002.json
        # ├── scalers/
        # │   ├── scaler_000.sc.mml     <- Scaler artifact (holds class definition if custom)
        # │   ├── scaler_001.sc.mml
        # │   └── scaler_002.sc.mml
        # │
        # ├── splitter_records/
        # │   ├── record_000.json       <- SplitterRecord config
        # │   ├── record_001.json
        # │   └── record_002.json
        # └── splitters/
        #     ├── splitter_000.sc.mml   <- Splitter artifact (holds class definition if custom)
        #     ├── splitter_001.sc.mml
        #     └── splitter_002.sc.mml

        if not hasattr(obj, "get_state"):
            raise NotImplementedError("FeatureSet must implement a `get_state` method.")

        # Track which files go where:
        file_mapping: dict[str, str] = {}

        # Grab FeatureSet state (holds live instances)
        state = obj.get_state()

        # Save SampleCollection to arrow file (stores data + schema)
        coll: SampleCollection = state["sample_collection"]
        coll_path = coll.save(Path(save_dir) / "sample_collection.arrow")
        file_mapping["sample_collection"] = coll_path.name

        # Save split configs (remove source reference)
        split_cfgs = {}
        splits: dict[str, FeatureSetView] = state["splits"]
        for k, fsv in splits.items():
            split_cfgs[k] = {
                "indices": fsv.indices.tolist() if isinstance(fsv.indices, np.ndarray) else fsv.indices,
                "columns": fsv.columns,
                "label": fsv.label,
            }
        split_cfg_path = self._write_json(
            data=split_cfgs,
            save_path=Path(save_dir) / "split_configs.json",
        )
        file_mapping["split_configs"] = split_cfg_path.name

        # Save Scaler records + artifacts
        # Scalers can hold user-defined code, we need to store serialized artifacts of the Scaler instances in addition to the ScalerRecord attributes
        dir_scaler_rec = Path(save_dir) / "scaler_records"
        dir_scaler_rec.mkdir(exist_ok=True)
        dir_scaler_art = Path(save_dir) / "scalers"
        dir_scaler_art.mkdir(exist_ok=True)

        scaler_cfg_files: list[str] = []
        scaler_recs: list[ScalerRecord] = sorted(state["scaler_records"], key=lambda x: x.order)
        n_digits = (len(scaler_recs) // 10) + 1
        for i, rec in enumerate(scaler_recs):
            # Store modifed scaler record config
            rec_cfg = rec.get_config()
            scaler_name: str = f"scaler_{i:0{n_digits}d}"

            # Save scaler artifact
            scaler_obj = rec.scaler_obj
            if scaler_obj is not None:
                cls_spec = ctx.make_class_spec(
                    cls=scaler_cfg_files.__class__,
                    policy=SerializationPolicy.BUILTIN,
                    builtin_key="Scaler",
                )
                save_path = ctx.emit_mml(
                    obj=scaler_obj,
                    cls_spec=cls_spec,
                    out_path=dir_scaler_art / scaler_name,
                    overwrite=True,
                )
                rec_cfg["scaler_ref"] = str(save_path)[str(save_path).rindex(save_dir.name) + len(save_dir.name) + 1 :]

            # Save config (JSON)
            cfg_path = (dir_scaler_rec / scaler_name).with_suffix(".json")
            save_path = self._write_json(data=rec_cfg, save_path=cfg_path)
            scaler_cfg_files.append(save_path.name)

        file_mapping["scaler_records"] = scaler_cfg_files

        # Save Splitter records + artifacts
        # Splitters can hold user-defined code, we need to store serialized artifacts of the Splitter instances in addition to the SplitterRecord attributes
        dir_splitter_rec = Path(save_dir) / "splitter_records"
        dir_splitter_rec.mkdir(exist_ok=True)
        dir_splitter_art = Path(save_dir) / "splitters"
        dir_splitter_art.mkdir(exist_ok=True)

        splitter_cfg_files: list[str] = []
        splitter_recs: list[SplitterRecord] = state["splitter_records"]
        n_digits = (len(splitter_recs) // 10) + 1
        for i, rec in enumerate(splitter_recs):
            # Store modifed splitter record config
            rec_cfg = rec.get_config()
            splitter_name: str = f"splitter_{i:0{n_digits}d}"

            # Save splitter artifact
            if rec.splitter is not None:
                cls_spec = ctx.make_class_spec(
                    cls=rec.splitter.__class__,
                    policy=SerializationPolicy.BUILTIN,
                    builtin_key="BaseSplitter",
                )
                save_path = ctx.emit_mml(
                    obj=rec.splitter,
                    cls_spec=cls_spec,
                    out_path=dir_splitter_art / splitter_name,
                    overwrite=True,
                )
                rec_cfg["splitter_ref"] = str(save_path)[
                    str(save_path).rindex(save_dir.name) + len(save_dir.name) + 1 :
                ]

            # Save config (JSON)
            cfg_path = (dir_splitter_rec / splitter_name).with_suffix(".json")
            save_path = self._write_json(data=rec_cfg, save_path=cfg_path)
            splitter_cfg_files.append(save_path.name)

        file_mapping["splitter_records"] = splitter_cfg_files

        return file_mapping

    # ================================================
    # Splitter Decoding
    # ================================================
    def decode(
        self,
        cls: type[FeatureSet],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> FeatureSet:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[FeatureSet]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            FeatureSet: The re-instantiated object.

        """
        from modularml.core.data.featureset_view import FeatureSetView

        config: dict[str, Any] = self.decode_config(load_dir=load_dir, ctx=ctx)

        # Instantiate FeatureSet from config
        fs_obj = cls.from_config(config=config)

        # Extract state
        state: dict[str, Any] = self.decode_state(load_dir=load_dir, ctx=ctx)

        # Splits need to instantiated with FeatureSet reference
        split_cfgs: dict[str, Any] = state.pop("split_cfgs")
        splits: dict[str, FeatureSetView] = {
            k: FeatureSetView(
                source=fs_obj,
                indices=cfg["indices"],
                columns=cfg["columns"],
                label=cfg["label"],
            )
            for k, cfg in split_cfgs.items()
        }
        state["splits"] = splits

        # Set state
        fs_obj.set_state(state=state)

        # Scalers don't always have a serializable state
        # To ensure future use of a de-serialized scaler with proper learned state,
        # we need to re-fit scalers in the order they were originally applied.
        # Remove any residual transformed representations (we fully rebuild from raw rep)
        for domain in [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS]:
            for k in fs_obj.collection._get_domain_keys(
                domain=domain,
                include_domain_prefix=False,
                include_rep_suffix=False,
            ):
                if REP_TRANSFORMED in fs_obj.collection._get_rep_keys(domain=domain, key=k):
                    fs_obj.collection.delete_rep(domain=domain, key=k, rep=REP_TRANSFORMED)

        # Reapply scalers (clear those on featureset)
        fs_obj._scaler_recs = []
        for rec in state["scaler_records"]:
            fs_obj.fit_transform(
                scaler=rec.scaler_obj,
                domain=rec.domain,
                keys=rec.keys,
                fit_to_split=rec.fit_split,
                merged_axes=rec.merged_axes,
            )

        return fs_obj

    def decode_config(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> dict[str, Any] | None:
        """
        Decodes config from a json file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            dict[str, Any] | None: The decoded config data.

        """
        return super().decode_config(load_dir=load_dir, ctx=ctx, config_rel_path=self.config_rel_path)

    def decode_state(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> dict[str, Any]:
        """
        Decodes state from a pkl file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            dict[str, Any]: The decoded state data.

        """
        from modularml.core.data.sample_collection import SampleCollection
        from modularml.core.splitting.splitter_record import SplitterRecord
        from modularml.core.transforms.scaler_record import ScalerRecord

        # The featureset.fs.mml file is structure as follows:
        # ├── artifact.json             <- FeatureSet identity
        # ├── config.json               <- FeatureSet.get_config()
        # │
        # ├── sample_collection.arrow   <- pyarrow table + schema
        # ├── split_configs.json        <- Splits (FeatureSetView)
        # │
        # ├── scaler_records/
        # │   ├── record_000.json       <- ScalerRecord config
        # │   ├── record_001.json
        # │   └── record_002.json
        # ├── scalers/
        # │   ├── scaler_000.sc.mml     <- Scaler artifact (holds class definition if custom)
        # │   ├── scaler_001.sc.mml
        # │   └── scaler_002.sc.mml
        # │
        # ├── splitter_records/
        # │   ├── record_000.json       <- SplitterRecord config
        # │   ├── record_001.json
        # │   └── record_002.json
        # └── splitters/
        #     ├── splitter_000.sc.mml   <- Splitter artifact (holds class definition if custom)
        #     ├── splitter_001.sc.mml
        #     └── splitter_002.sc.mml
        state: dict[str, Any] = {}
        load_dir = Path(load_dir)

        # Extract file mapping
        file_mapping: dict[str, Any] = self._read_json(load_dir / "artifact.json")["files"]

        # Decode the SampleCollection
        file_coll: Path = load_dir / file_mapping["sample_collection"]
        coll: SampleCollection = SampleCollection.load(file_coll)
        state["sample_collection"] = coll

        # Add hash for later validation
        state["table_hash"] = hash_pyarrow_table(coll.table)

        # Decode splits (configs only)
        file_split_cfgs = load_dir / file_mapping["split_configs"]
        split_cfgs: dict[str, dict[str, Any]] = self._read_json(file_split_cfgs)
        state["split_cfgs"] = split_cfgs

        # Decode splitter records
        files_split_recs: list[Path] = [(load_dir / "splitter_records" / x) for x in file_mapping["splitter_records"]]
        split_rec_cfgs: list[dict[str, Any]] = [self._read_json(x) for x in files_split_recs]
        split_recs: list[SplitterRecord] = []
        for rec_cfg in split_rec_cfgs:
            # Instantiate splitter via LoadContext
            splitter_obj = ctx.load_from_mml(mml_file=load_dir / rec_cfg["splitter_ref"])

            # Create record
            split_recs.append(
                SplitterRecord(
                    splitter=splitter_obj,
                    applied_to=DataReference.from_config(rec_cfg["applied_to_config"]),
                ),
            )
        state["splitter_records"] = split_recs

        # Decode scaler records
        files_scaler_recs: list[Path] = [(load_dir / "scaler_records" / x) for x in file_mapping["scaler_records"]]
        scaler_rec_cfgs: list[dict[str, Any]] = [self._read_json(x) for x in files_scaler_recs]
        scaler_recs: list[SplitterRecord] = []
        for rec_cfg in scaler_rec_cfgs:
            # Instantiate scaler via LoadContext
            scaler_obj = ctx.load_from_mml(mml_file=load_dir / rec_cfg["scaler_ref"])

            # Need to remove scaler config, as we from_config
            # will fail if scaler is not in SCALER_REGISTRY
            _ = rec_cfg.pop("scaler_config", None)

            # Create record
            s_rec = ScalerRecord.from_config(rec_cfg)
            s_rec = replace(s_rec, scaler_obj=scaler_obj)
            scaler_recs.append(s_rec)

        state["scaler_records"] = scaler_recs

        return state
