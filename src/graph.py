import matplotlib.pyplot as plt


def plot_xyz(data, x_key="time", colors={"x": "red", "y": "green", "z": "blue"}):
    y_keys: list[str] = []
    for potential_y_key in ["x", "y", "z"]:
        if potential_y_key in data:
            y_keys.append(potential_y_key)

    fig, axes = plt.subplots(1, len(y_keys), sharey=True)

    for i in range(len(y_keys)):
        axis = axes[i]
        y_key = y_keys[i]

        color = "black"
        if y_key in colors:
            color = colors[y_key]

        axis.set_xlabel(x_key)
        axis.plot(data[x_key], data[y_key], color=color)

    return fig
