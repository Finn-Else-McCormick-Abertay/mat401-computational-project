import matplotlib.pyplot as plt


def plot_xyz(
    data,
    x_key="time",
    y_keys=["x", "y", "z"],
    colors={"x": "red", "y": "green", "z": "blue"},
):
    # y_keys: list[str] = []
    # for potential_y_key in ["x", "y", "z"]:
    #    if potential_y_key in data:
    #        y_keys.append(potential_y_key)

    # fig, axes = plt.subplots(1, len(y_keys), sharey=True)
    fig, axes = plt.subplots(len(y_keys), 1, constrained_layout=True)

    for i in range(len(y_keys)):
        axis = axes[i] if len(y_keys) > 1 else axes
        y_key = y_keys[i]

        color = "black"
        if y_key in colors:
            color = colors[y_key]

        axis.set_xlabel(x_key)
        axis.set_ylabel(y_key)
        axis.plot(data[x_key], data[y_key], color=color)

    return fig
