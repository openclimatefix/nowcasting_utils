""" Test for checking metric """
import torch

from nowcasting_utils.models.metrics import mae_each_forecast_horizon, mse_each_forecast_horizon


def test_mse_each_forecast_horizon():
    """ Test MSE for different forecast horizons """
    output = torch.Tensor([[1, 3], [1, 6]])
    target = torch.Tensor([[1, 5], [1, 3]])

    loss = mse_each_forecast_horizon(output=output, target=target)

    assert loss.cpu().numpy()[0] == 0
    assert loss.cpu().numpy()[1] == (2 * 2 + 3 * 3) / 2


def test_mae_each_forecast_horizon():
    """ Test MAE for different forecast horizons """
    output = torch.Tensor([[1, 3], [1, 6]])
    target = torch.Tensor([[1, 5], [1, 3]])

    loss = mae_each_forecast_horizon(output=output, target=target)

    assert loss.cpu().numpy()[0] == 0
    assert loss.cpu().numpy()[1] == (2 + 3) / 2
