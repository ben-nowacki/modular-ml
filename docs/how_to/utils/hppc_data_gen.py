import numpy as np


def get_mock_hppc_data(n_samples: int = 600):
    n_cells = 15
    n_groups = 5

    # HPPC time-segment boundaries (seconds)
    t_ocv_end = 10  # end of OCV observation
    t_chg_end = 20  # end of charge pulse
    t_rest1_end = 60  # end of rest after charge
    t_dchg_end = 70  # end of discharge pulse
    t_rest2_end = 110  # end of rest after discharge

    # Sample metadata
    rng = np.random.default_rng(42)
    cell_ids = rng.integers(1, n_cells + 1, size=n_samples)
    group_ids = rng.integers(1, n_groups + 1, size=n_samples)

    # SOH decreases with group_id (group 1 = fresh, group 5 = aged)
    soh = 100.0 - rng.normal(0, 20.0, size=n_samples)
    soh = np.clip(soh, 50.0, 100.0)

    # OCV reflects SOC — random across a wide range of test conditions
    ocv = rng.uniform(2.0, 3.6, size=n_samples)

    # Electrochemical parameters (degrade linearly with SOH loss)
    fade = 1.0 - soh / 100.0  # 0 (new) -> ~0.5 (aged)
    r0_all = 0.05 + fade * 0.05  # ohmic resistance (Ohm)
    rct_all = 0.08 + fade * 0.1  # charge-transfer resistance (Ohm)

    i_chg = 1.2  # A (charge)
    i_dchg = -1.2  # A (discharge)

    tau_chg = 8.0 * (1 - fade)  # charge-pulse RC time constant (s)
    tau_rest1 = 20.0 * (1 - fade / 2)  # relaxation time constant after charge (s)
    tau_dchg = 6.0 * (1 - fade)  # discharge-pulse RC time constant (s)
    tau_rest2 = 25.0 * (1 - fade / 2)  # relaxation time constant after discharge (s)

    # Synthesize voltage traces
    voltage = np.zeros((n_samples, t_rest2_end))
    noise = 0.001  # V

    for i in range(n_samples):
        v = ocv[i]
        r0 = r0_all[i]
        rc = rct_all[i]
        tau_c = tau_chg[i]
        tau_r1 = tau_rest1[i]
        tau_d = tau_dchg[i]
        tau_r2 = tau_rest2[i]

        # 1. OCV region
        voltage[i, :t_ocv_end] = v + rng.normal(0, noise, t_ocv_end)

        # 2. Charge pulse: ohmic jump + exponential rct_all rise
        t_seg = np.arange(t_chg_end - t_ocv_end)
        v_chg = v + i_chg * r0 + i_chg * rc * (1 - np.exp(-t_seg / tau_c))
        voltage[i, t_ocv_end:t_chg_end] = v_chg + rng.normal(0, noise, len(t_seg))
        v_end_chg = v_chg[-1]

        # 3. Rest after charge: instant ohmic recovery + slow relaxation
        v_relax1 = v + i_chg * rc * 0.15
        v_rest1_0 = v_end_chg - i_chg * r0
        t_seg = np.arange(t_rest1_end - t_chg_end)
        v_rest1 = v_relax1 + (v_rest1_0 - v_relax1) * np.exp(-t_seg / tau_r1)
        voltage[i, t_chg_end:t_rest1_end] = v_rest1 + rng.normal(0, noise, len(t_seg))
        v_before_dchg = v_rest1[-1]

        # 4. Discharge pulse: ohmic drop + exponential rct_all fall
        t_seg = np.arange(t_dchg_end - t_rest1_end)
        v_dchg = (
            v_before_dchg + i_dchg * r0 + i_dchg * rc * (1 - np.exp(-t_seg / tau_d))
        )
        voltage[i, t_rest1_end:t_dchg_end] = v_dchg + rng.normal(0, noise, len(t_seg))
        v_end_dchg = v_dchg[-1]

        # 5. Rest after discharge: instant ohmic recovery + slow relaxation
        v_relax2 = v_before_dchg + i_dchg * rc * 0.15
        v_rest2_0 = v_end_dchg - i_dchg * r0
        t_seg = np.arange(t_rest2_end - t_dchg_end)
        v_rest2 = v_relax2 + (v_rest2_0 - v_relax2) * np.exp(-t_seg / tau_r2)
        voltage[i, t_dchg_end:t_rest2_end] = v_rest2 + rng.normal(0, noise, len(t_seg))

    return voltage, soh, cell_ids, group_ids
