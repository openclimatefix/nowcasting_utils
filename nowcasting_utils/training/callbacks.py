"""
Custom callbacks used for training
"""
import os

from pytorch_lightning import Callback, LightningModule, Trainer


class NeptuneModelLogger(Callback):
    """
    Saves out the last and best models after each validation epoch.

    If the files don't exists, does nothing.

    Example::
        from pl_bolts.callbacks import NeptuneModelLogger
        trainer = Trainer(callbacks=[NeptuneModelLogger()])
    """

    def __init__(self) -> None:
        """
        Base initialization, nothing specific needed here
        """
        super().__init__()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Save the best and last model checkpoints to Neptune after each validation

        Args:
            trainer: PyTorchLightning trainer
            pl_module: LightningModule that is being trained

        Returns:
            None
        """
        try:
            trainer.logger.experiment[0]["model_checkpoints/last.ckpt"].upload(
                os.path.join(trainer.default_root_dir, "checkpoints", "last.ckpt")
            )
        except Exception as e:
            print(
                f"No file to upload at "
                f"{os.path.join(trainer.default_root_dir, 'checkpoints', 'last.ckpt')} "
                f"{e}"
            )
            pass

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Save out the best and last model checkpoints at the end of trainer.fit to Neptune

        Args:
            trainer: PyTorchLightning Trainer
            pl_module: LightningModule being used for training

        Returns:
            None
        """
        try:
            trainer.logger.experiment[0]["model_checkpoints/best.ckpt"].upload(
                os.path.join(trainer.default_root_dir, "checkpoints", "best.ckpt"),
            )
        except Exception as e:
            print(
                f"No file to upload at "
                f"{os.path.join(trainer.default_root_dir, 'checkpoints', 'best.ckpt')} "
                f"{e}"
            )
            pass
