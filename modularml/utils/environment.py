def running_in_notebook() -> bool:
    """Checks is the current enviromment is a Jupyter notebook."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False

        # Jupyter notebooks and qtconsole
        return ip.__class__.__name__ in {  # noqa: TRY300
            "ZMQInteractiveShell",
        }
    except Exception:  # noqa: BLE001
        return False
