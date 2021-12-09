""" Test for different losses"""
import pytest
import torch

from nowcasting_utils.models.loss import WeightedLosses, get_loss


def test_weight_losses_weights():
    """Test weighted loss"""
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    assert w.weights.cpu().numpy()[0] == pytest.approx(4 / 3)
    assert w.weights.cpu().numpy()[1] == pytest.approx(2 / 3)


def test_mae_exp():
    """Test MAE exp with weighted loss"""
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    output = torch.Tensor([[1, 3], [1, 3]])
    target = torch.Tensor([[1, 5], [1, 9]])

    loss = w.get_mae_exp(output=output, target=target)

    # 0.5((1-1)*2/3 + (5-3)*1/3) + 0.5((1-1)*2/3 + (9-3)*1/3) = 1/3 + 3/3
    assert loss == pytest.approx(4 / 3)


def test_mse_exp():
    """Test MSE exp with weighted loss"""
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    output = torch.Tensor([[1, 3], [1, 3]])
    target = torch.Tensor([[1, 5], [1, 9]])

    loss = w.get_mse_exp(output=output, target=target)

    # 0.5((1-1)^2*2/3 + (5-3)^2*1/3) + 0.5((1-1)^2*2/3 + (9-3)^2*1/3) = 2/3 + 18/3
    assert loss == pytest.approx(20 / 3)


def test_mae_exp_rand():
    """Test MAE exp with weighted loss  with random tensors"""
    forecast_length = 6
    batch_size = 32

    w = WeightedLosses(forecast_length=6)

    output = torch.randn(batch_size, forecast_length)
    target = torch.randn(batch_size, forecast_length)

    loss = w.get_mae_exp(output=output, target=target)
    assert loss > 0


def test_mse_exp_rand():
    """Test MSE exp with weighted loss  with random tensors"""
    forecast_length = 6
    batch_size = 32

    w = WeightedLosses(forecast_length=6)

    output = torch.randn(batch_size, forecast_length)
    target = torch.randn(batch_size, forecast_length)

    loss = w.get_mse_exp(output=output, target=target)
    assert loss > 0


@pytest.mark.parametrize(
    "loss_name",
    [
        "mse",
        "bce",
        "binary_crossentropy",
        "crossentropy",
        "focal",
        "ssim",
        "ms_ssim",
        "l1",
        "tv",
        "total_variation",
        "ssim_dynamic",
        "gdl",
        "gradient_difference_loss",
        "weighted_mae",
        "weighted_mse",
    ],
)
def test_get_loss(loss_name):
    """Test to get loss name"""
    _ = get_loss(loss_name)


@pytest.mark.parametrize(
    "loss_name",
    [
        "mse",
        "l1",
        "gradient_difference_loss",
    ],
)
def test_video_loss(loss_name):
    """Test video loss"""
    loss = get_loss(loss_name)
    output = torch.randn((2, 24, 12, 512, 512))
    target = torch.randn((2, 24, 12, 512, 512))
    out = loss(output, target)
    assert out > 0


def test_tv_loss():
    """Test TV loss"""
    loss = get_loss("tv")
    output = torch.randn((2, 12, 512, 512))
    out = loss(output)
    assert out > 0


def test_missing_loss():
    """Test to check error is rasied"""
    with pytest.raises(AssertionError):
        _ = get_loss("made_up_metric")


@pytest.mark.parametrize("loss_name", ["ssim", "ms_ssim"])
def test_convert_ssim_loss(loss_name):
    """Test for SSIM"""
    loss = get_loss(loss_name, convert_range=True, channel=12)
    output = torch.randn((2, 12, 512, 512))
    target = torch.randn((2, 12, 512, 512))
    # Convert to -1,1
    output = (output * 2) - 1
    target = (target * 2) - 1
    out = loss(output, target)
    assert out > 0.0


@pytest.mark.parametrize("loss_name", ["ssim", "ms_ssim"])
def test_convert_ssim_loss_no_convert_range(loss_name):
    """Test for SSIM with no converted range"""
    loss = get_loss(loss_name, convert_range=False, channel=12)
    output = torch.randn((2, 12, 512, 512))
    target = torch.randn((2, 12, 512, 512))
    out = loss(output, target)
    assert out > 0.0
