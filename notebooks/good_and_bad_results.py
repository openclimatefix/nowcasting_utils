""" Plot of good and bad results """
import neptune.new as neptune
import pandas as pd
from plotly.subplots import make_subplots

from nowcasting_utils.visualization.line import make_trace

# oad results from neptune

run = neptune.init(project="OpenClimateFix/predict-pv-yield", run="PRED-658")


epoch = 4
destination_path = "./temp.csv"
results_df = run[f"validation/results/epoch_{epoch}"].download(destination_path)

results = pd.read_csv(destination_path)

# plot results


good_index = [84, 5, 14]
medium_index = [2, 5, 22]
bad_index = [19, 30, 47]

all_indexes = good_index + medium_index + bad_index

traces = []
subplot_titles = []
for i in range(len(all_indexes)):

    print(i)

    index = all_indexes[i] % 32
    batch = all_indexes[i] // 32

    results_one = results[results["batch_index"] == batch]
    results_one = results_one[results_one["example_index"] == index]
    subplot_titles.append(f"GSP id: {int(results_one.iloc[0].gsp_id)}")

    if i == 0:
        show_legend = True
    else:
        show_legend = False

    trace = make_trace(
        y=results_one["forecast_gsp_pv_outturn_mw"],
        x=results_one["target_datetime_utc"],
        truth=False,
        show_legend=show_legend,
    )

    trace_truth = make_trace(
        y=results_one["actual_gsp_pv_outturn_mw"],
        x=results_one["target_datetime_utc"],
        truth=True,
        show_legend=show_legend,
    )

    trace_capacity = make_trace(
        y=results_one["capacity_mwp"],
        x=results_one["target_datetime_utc"],
        name="Installed Capacity",
        show_legend=show_legend,
        color="black",
        truth=False,
        mode="lines",
    )

    traces.append([trace, trace_truth, trace_capacity])


fig = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=subplot_titles,
    # title=f"Example Predictions"
)

for i in range(len(all_indexes)):
    trace, trace_truth, trace_capacity = traces[i]

    col = i % 3 + 1
    row = i // 3 + 1
    fig.add_trace(trace, row, col)
    fig.add_trace(trace_truth, row, col)
    fig.add_trace(trace_capacity, row, col)


fig["layout"]["yaxis"]["title"] = "Solar Generation [MW]"
fig["layout"]["yaxis4"]["title"] = "Solar Generation [MW]"
fig["layout"]["yaxis7"]["title"] = "Solar Generation [MW]"
fig["layout"]["title"] = "Example comparison of Predictions and Targets"
fig.show(renderer="browser")
