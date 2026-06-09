import pickle
import urllib.request
import warnings
from pathlib import Path
from typing import Literal

import numpy as np


def _process_finetuning_data(data: dict) -> dict:
    """
    Process raw finetuning dataset into cleaned dict.

    Steps:
    1. Compute cumulative cycle counts per cell
    2. Align charge and discharge pulses
    3. Create current profiles
    4. Extract and organize all fields

    Args:
        data: Raw data dict from pickle file

    Returns:
        Cleaned data dict ready for FeatureSet

    """

    def _create_cumulative_cycle_counts(
        cell_ids: np.ndarray,
        rpt_nums: np.ndarray,
        cycle_counts: np.ndarray,
    ) -> np.ndarray:
        """
        For each cell_id, compute the cumulative cycle count across its RPT numbers.

        The input arrays are parallel and equal length. Each (cell_id, rpt_num)
        group may repeat multiple times (e.g., 9 duplicates for 9 SOC levels).
        The function collapses duplicates within a group, sorts by RPT number,
        and takes the cumulative sum of cycle_counts for that cell only.

        Args:
            cell_ids: Array of cell identifiers (N,)
            rpt_nums: Array of RPT numbers (N,)
            cycle_counts: Array of cycle counts per RPT (N,)

        Returns:
            Array of cumulative cycle counts (N,)

        """
        cum_cycle_nums = np.empty_like(cycle_counts)

        # Process each cell_id separately
        for cid in np.unique(cell_ids):
            mask = cell_ids == cid

            # Restrict arrays to this cell
            rpt_sub = rpt_nums[mask]
            cycles_sub = cycle_counts[mask]

            # Unique RPT numbers for this cell
            rpt_unique, first_idx = np.unique(rpt_sub, return_index=True)

            # Cycle count per RPT (one value per RPT)
            cycles_per_rpt = cycles_sub[first_idx]

            # Sort by RPT number and take cumulative sum
            order = np.argsort(rpt_unique)
            rpt_sorted = rpt_unique[order]
            cum_cycles = np.cumsum(cycles_per_rpt[order])

            # Map cumulative cycles back to every element of this cell's subset
            cum_map = dict(zip(rpt_sorted, cum_cycles, strict=True))
            cum_cycle_nums[mask] = [cum_map[r] for r in rpt_sub]

        return cum_cycle_nums

    def _align_charge_discharge_pulses(data: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Find matching discharge pulse for each charge pulse.

        Matching criteria:
        - Same cell_id
        - Same group_id
        - Same rpt
        - Same soc
        - Opposite pulse_type (chg vs dchg)

        Args:
            data: Raw data dict with keys: pulse_type, cell_id, group_id, rpt, soc

        Returns:
            Tuple of (charge_indices, discharge_indices) as parallel arrays

        Raises:
            ValueError: If any charge pulse has != 1 matching discharge pulse

        """
        chg_idxs = np.where(data["pulse_type"] == "chg")[0]
        dchg_idxs = []

        for chg_idx in chg_idxs:
            matching_dchg = np.where(
                (data["pulse_type"] == "dchg")
                & (data["cell_id"] == data["cell_id"][chg_idx])
                & (data["group_id"] == data["group_id"][chg_idx])
                & (data["rpt"] == data["rpt"][chg_idx])
                & (data["soc"] == data["soc"][chg_idx]),
            )[0]

            if len(matching_dchg) != 1:
                msg = (
                    f"Expected exactly 1 matching discharge pulse for charge index {chg_idx}. "
                    f"Found {len(matching_dchg)} matches."
                )
                raise ValueError(msg)

            dchg_idxs.append(matching_dchg[0])

        return chg_idxs, np.asarray(dchg_idxs)

    def _create_current_profiles(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Create charge and discharge current profiles.

        Current profile structure:
        - 1s: 0 A (initial rest)
        - 30s: C/5 rate (0.24 A for 1.2 Ah cell)
        - 10s: 1C rate (1.2 A)
        - 60s: 0 A (rest)
        - Total: 101 seconds

        Args:
            n_samples: Number of pulse pairs to create profiles for

        Returns:
            Tuple of (charge_currents, discharge_currents), each shape (n_samples, 101)

        """
        # Define single current profile (101 timesteps)
        c5 = np.full(shape=30, fill_value=1.2 / 5)  # C/5 rate
        c = np.full(shape=10, fill_value=1.2)  # 1C rate
        rest = np.full(shape=60, fill_value=0)  # Rest period

        chg_iprofile_single = np.hstack([0, c5, c, rest])  # Shape: (101,)

        # Replicate for all samples
        chg_currents = np.tile(
            chg_iprofile_single,
            (n_samples, 1),
        )  # Shape: (n_samples, 101)
        dchg_currents = -1 * chg_currents

        return chg_currents, dchg_currents

    # Clean invalid samples
    valid = (
        (data["soc - coulomb"] >= 0)
        & (data["soc - coulomb"] <= 100)
        & (data["soh"] >= 0)
        & (data["soh"] <= 100)
    )
    filtered_data = {key: value[valid] for key, value in data.items()}

    print("Computing cumulative cycle counts...")
    num_cycles_cumulative = _create_cumulative_cycle_counts(
        cell_ids=filtered_data["cell_id"],
        rpt_nums=filtered_data["rpt"],
        cycle_counts=filtered_data["num_cycles"],
    )

    print("Aligning charge and discharge pulses...")
    chg_idxs, dchg_idxs = _align_charge_discharge_pulses(filtered_data)

    print(f"Found {len(chg_idxs)} aligned charge-discharge pulse pairs")

    # Create current profiles
    chg_iprofile, dchg_iprofile = _create_current_profiles(len(chg_idxs))

    # Build cleaned filtered_data dictionary
    cleaned_data = {
        # Features
        "chg_voltage": filtered_data["voltage"][chg_idxs],
        "dchg_voltage": filtered_data["voltage"][dchg_idxs],
        "chg_current": chg_iprofile,
        "dchg_current": dchg_iprofile,
        # Targets
        "soh": filtered_data["soh"][chg_idxs],
        # Tags
        "cell_id": filtered_data["cell_id"][chg_idxs],
        "group_id": filtered_data["group_id"][chg_idxs],
        "rpt": filtered_data["rpt"][chg_idxs],
        "num_cycles": filtered_data["num_cycles"][chg_idxs],
        "num_cycles_cumulative": num_cycles_cumulative[chg_idxs],
        "expected_soc": filtered_data["soc"][chg_idxs],
        "true_soc": filtered_data["soc - coulomb"][chg_idxs],
    }

    print(f"Cleaned data shape: {cleaned_data['chg_voltage'].shape}")
    return cleaned_data


def get_dataset(
    chemistry: Literal["NMC", "LFP"],
    save_dir: Path,
    study: str = "finetuning",
) -> dict:
    """
    Retrieves the specified dataset as a dict ready for FeatureSet.from_dict().

    Downloads raw data if needed, processes it, and caches the cleaned dict
    for fast reload on subsequent calls.

    Args:
        chemistry: Battery chemistry type ("NMC" or "LFP")
        save_dir: Directory for downloads and cache
        study: Dataset identifier (currently only "finetuning" supported)

    Returns:
        Dictionary with keys:
            - chg_voltage: Charge pulse voltage curves (N, 101)
            - dchg_voltage: Discharge pulse voltage curves (N, 101)
            - chg_current: Charge pulse current profiles (N, 101)
            - dchg_current: Discharge pulse current profiles (N, 101)
            - soh: State of health values (N,)
            - cell_id: Cell identifiers (N,)
            - group_id: Cycling group identifiers (N,)
            - rpt: RPT (reference performance test) numbers (N,)
            - num_cycles: Cycle counts per RPT (N,)
            - num_cycles_cumulative: Cumulative cycle counts (N,)
            - expected_soc: Expected SOC levels (N,)
            - true_soc: Coulomb-counted SOC levels (N,)

    Raises:
        ValueError: If study is not supported

    """

    def _download_raw_data() -> Path:
        """Download raw pulse data from GitHub if not already present."""
        raw_data_file = save_dir / f"pulse_data_{chemistry_lower}.pkl"

        if raw_data_file.exists():
            print(f"Raw data already downloaded: {raw_data_file}")
            return raw_data_file

        # Download pulse data
        url_data = (
            f"https://raw.githubusercontent.com/REIL-UConn/fine-tuning-for-rapid-soh-estimation/main/"
            f"processed_data/UConn-ILCC-{chemistry_upper}/data_slowpulse_1.pkl"
        )
        print(f"Downloading {chemistry_upper} pulse data from GitHub...")
        try:
            urllib.request.urlretrieve(url_data, raw_data_file)  # noqa: S310
            print(f"Download complete: {raw_data_file}")
        except Exception as e:
            msg = f"Failed to download {chemistry_upper} data: {e}"
            raise RuntimeError(msg) from e

        return raw_data_file

    if study != "finetuning":
        msg = (
            f"Study '{study}' not supported. Only 'finetuning' is currently available."
        )
        raise ValueError(msg)

    chemistry_lower = chemistry.lower()
    chemistry_upper = chemistry.upper()

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Check for cached cleaned data
    cache_file = save_dir / f"cleaned_data_{chemistry_lower}_{study}.pkl"
    if cache_file.exists():
        # print(f"Loading cached cleaned data for {chemistry_upper}...")
        with cache_file.open("rb") as f:
            return pickle.load(f)
    print(f"Cached data not found. Processing {chemistry_upper} dataset...")

    # Download raw data if needed
    raw_data_file = _download_raw_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        with raw_data_file.open("rb") as f:
            data = pickle.load(f)

    # Process & cache
    cleaned_data = _process_finetuning_data(data)
    print(f"Caching cleaned data to {cache_file}...")
    with cache_file.open("wb") as f:
        pickle.dump(cleaned_data, f)

    return cleaned_data
