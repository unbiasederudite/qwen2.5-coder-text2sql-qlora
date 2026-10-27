"""The progress bar of the training."""

from transformers.trainer_callback import ProgressCallback


class MetricsProgressCallback(ProgressCallback):  # type: ignore[misc]
    """Text progress bar that keeps one line and updates its loss on each log."""

    def on_log(
        self,
        args: object,
        state: object,
        control: object,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        """Shows the latest loss after the bar.

        Args:
            args (object): Training arguments, unused.
            state (object): Trainer state, unused.
            control (object): Trainer control, unused.
            logs (dict[str, float] | None): Metrics of the last logging step.
            **kwargs (object): Other callback arguments, unused.
        """
        if self.training_bar is None or not logs or "loss" not in logs:
            return
        self.training_bar.set_postfix(loss=f"{logs['loss']:.4f}")
