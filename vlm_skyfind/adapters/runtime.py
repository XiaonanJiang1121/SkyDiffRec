"""Actionable errors for optional model-family runtimes."""


def missing_runtime(package, setup_target, error):
    raise RuntimeError(
        f"Missing optional package {package!r}. From the VLMSkyFind root run "
        f"`bash scripts/setup_model_runtimes.sh {setup_target}` and retry."
    ) from error
