"""
Base model class for all ML models.

Useful things like
- Same validation set
- Interface with HuggingFace
"""
from typing import Any, Type

import pytorch_lightning as pl
import torch.nn
import torchvision
from neptune.new.types import File
from nowcasting_dataset.consts import SATELLITE_DATA

from nowcasting_utils.models.hub import (
    NowcastingModelHubMixin,
    load_model_config_from_hf,
    load_pretrained,
)

REGISTERED_MODELS = {}


def register_model(cls: Type[pl.LightningModule]):
    """
    Register model

    Args:
        cls: the model to be registered

    Returns: the registered model

    """
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls


def get_model(name: str) -> Type[pl.LightningModule]:
    """Get model from registered models"""
    global REGISTERED_MODELS
    assert name in REGISTERED_MODELS, f"available class: {REGISTERED_MODELS}"
    return REGISTERED_MODELS[name]


def list_models():
    """List of the registered models"""
    global REGISTERED_MODELS
    return REGISTERED_MODELS.keys()


def split_model_name(model_name):
    """
    Split model name with ':'

    Args:
        model_name: the original model name

    Returns: source name, and the model name

    """
    model_split = model_name.split(":", 1)
    if len(model_split) == 1:
        return "", model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ("satflow", "hf_hub")
        return source_name, model_name


def safe_model_name(model_name, remove_source=True):
    """
    Make a safe model name

    Args:
        model_name: the original model name
        remove_source: flag if to remove the source or not

    Returns: the new model name

    """

    def make_safe(name):
        return "".join(c if c.isalnum() else "_" for c in name).rstrip("_")

    if remove_source:
        model_name = split_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(model_name, pretrained=False, checkpoint_path=None, **kwargs):
    """Create a model

    Almost entirely taken from timm https://github.com/rwightman/pytorch-image-models

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        input_channels (int): number of input channels (default: 12)
        forecast_steps (int): number of steps to forecast (default: 48)
        lr (float): learning rate (default: 0.001)
        **: other kwargs are model specific
    """
    source_name, model_name = split_model_name(model_name)

    # Parameters that aren't supported by all models or are intended to only
    # override model defaults if set should default to None in command line args/cfg.
    # Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if source_name == "hf_hub":
        # For model names specified in the form `hf_hub:path/architecture_name#revision`,
        # load model weights + default_cfg from Hugging Face hub.
        hf_default_cfg, model_name = load_model_config_from_hf(model_name)
        # Want to set the kwargs to correct values
        # Should ignore this after we get it in the model_name
        kwargs.update(hf_default_cfg)
        kwargs.pop("hf_hub")
        kwargs.pop("architecture")

    if model_name in REGISTERED_MODELS:
        model = get_model(model_name)
    else:
        raise RuntimeError("Unknown model (%s)" % model_name)

    if checkpoint_path or pretrained:
        kwargs["checkpoint_path"] = checkpoint_path
        # Readd hf_hub here
        if source_name == "hf_hub":
            kwargs["hf_hub"] = hf_default_cfg.get("hf_hub")
        model = load_pretrained(model, default_cfg=kwargs, in_chans=kwargs["input_channels"])
    else:
        # Initialize model here as LightingModules need a special way of loading checkpoints,
        # this initializes randomly
        model = model(**kwargs)
    return model


class BaseModel(pl.LightningModule, NowcastingModelHubMixin):
    """Base Model for ML models"""

    def __init__(
        self,
        pretrained: bool = False,
        forecast_steps: int = 48,
        input_channels: int = 12,
        output_channels: int = 12,
        lr: float = 0.001,
        visualize: bool = False,
    ):
        """
        Setup the base model class.

        Args:
            pretrained: flag is thie model is pretrained or not
            forecast_steps: the number of forecasts steps
            input_channels: the number of input channels
            output_channels: the number of output channels
            lr: the learning rate
            visualize: if to visualize the resutls or not
        """
        super(BaseModel, self).__init__()
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.lr = lr
        self.pretrained = pretrained
        self.visualize = visualize

    @classmethod
    def from_config(cls, config):
        """
        Get the model from a config file.

        Args:
            config: config file

        Returns: Error, as the model needs to implement this method

        """
        raise NotImplementedError

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        """
        The train or validation step.

        This need to be made for each specific model

        Args:
            batch: the batched data
            batch_idx: the batch index
            is_training: a flag if this is a training step or not

        Returns: The model outputs

        """
        pass

    def training_step(self, batch, batch_idx):
        """
        The training step.

        Args:
            batch: the batch data
            batch_idx: the batch index

        Returns: The model outputs

        """
        return self._train_or_validate_step(batch, batch_idx, is_training=True)

    def validation_step(self, batch, batch_idx):
        """
        Validation step

        Args:
            batch: the batch data
            batch_idx: the batch index

        Returns: The model outputs

        """
        return self._train_or_validate_step(batch, batch_idx, is_training=False)

    def forward(self, x, **kwargs) -> Any:
        """
        Forward method for the model.

        Args:
            x: the input data
            **kwargs: other input needed

        Returns: the model outputs

        """
        return self.model.forward(x, **kwargs)

    def visualize_step(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, batch_idx: int, step: str
    ) -> None:
        """
        Visualization Step

        Args:
            x: input data
            y: the truth
            y_hat: the predictions
            batch_idx: what batch index this is
            step: what step number this is

        """
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment[0]
        # Timesteps per channel
        images = x[0][SATELLITE_DATA].cpu().detach().float()  # T, C, H, W
        future_images = y[0][SATELLITE_DATA].cpu().detach().float()
        generated_images = y_hat[0].cpu().detach().float()
        for i, t in enumerate(images):  # Now would be (C, H, W)
            t = [torch.unsqueeze(img, dim=0) for img in t]
            image_grid = torchvision.utils.make_grid(
                t, nrow=self.input_channels // 2, normalize=True
            )
            # Neptune needs it to be H x W x C, so have to permute
            image_grid = image_grid.permute(1, 2, 0)
            tensorboard[f"{step}/Input_Frame_{i}"].log(File.as_image(image_grid.numpy()))
            t = [torch.unsqueeze(img, dim=0) for img in future_images[i]]
            image_grid = torchvision.utils.make_grid(
                t, nrow=self.output_channels // 2, normalize=True
            )
            image_grid = image_grid.permute(1, 2, 0)
            tensorboard[f"{step}/Target_Frame_{i}"].log(File.as_image(image_grid.numpy()))
            t = [torch.unsqueeze(img, dim=0) for img in generated_images[i]]
            image_grid = torchvision.utils.make_grid(
                t, nrow=self.output_channels // 2, normalize=True
            )
            image_grid = image_grid.permute(1, 2, 0)
            tensorboard[f"{step}/Predicted_Frame_{i}"].log(File.as_image(image_grid.numpy()))
