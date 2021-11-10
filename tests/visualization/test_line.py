""" Test for making line plots """
import numpy as np

from nowcasting_utils.visualization.line import make_trace, plot_batch_results, plot_one_result


def test_make_trace():
    """Test make line trace"""
    x = np.random.random(7)
    y = np.random.random(7)

    _ = make_trace(x=x, y=y, truth=True)
    _ = make_trace(x=x, y=y, truth=False)
    _ = make_trace(x=x, y=y, truth=True, show_legend=False)


def test_plot_batch_results():
    """Test plot batch results"""
    size = (32, 7)

    x = np.random.random(size)
    y = np.random.random(size)
    y_hat = np.random.random(size)

    _ = plot_batch_results(x=x, y=y, y_hat=y_hat, model_name="test_model")
    # fig.show(renderer='browser')


def test_plot_one_result():
    """Test plot one result"""
    x = np.random.random(7)
    y = np.random.random(7)
    y_hat = np.random.random(7)

    _ = plot_one_result(x=x, y=y, y_hat=y_hat)
