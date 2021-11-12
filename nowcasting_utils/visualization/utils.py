""" Plotting utils functions """
from typing import List


def make_slider(labels: List[str]) -> dict:
    """Make slider for animation"""
    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"frame{k+1}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=600, redraw=True),
                            transition=dict(duration=200),
                        ),
                    ],
                    label=f"{labels[k]}",
                )
                for k in range(0, len(labels))
            ],
            transition=dict(duration=100),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), visible=True, xanchor="center"),
            len=1.0,
        )
    ]
    return sliders


def make_buttons() -> dict:
    """Make buttons Play and Pause"""
    return dict(
        type="buttons",
        buttons=[
            dict(label="Play", method="animate", args=[None]),
            dict(
                args=[
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                label="Pause",
                method="animate",
            ),
        ],
    )
