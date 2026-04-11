import os
import numpy as np
import matplotlib.pyplot as plt


def plot_darts_predictions(prediction_data, plot_len, n_seqs, save_dir=None):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    keys_to_plot = list(prediction_data.keys())[:n_seqs]

    for key in keys_to_plot:
        true_ts = prediction_data[key]["heldout_y"]
        pred_ts = prediction_data[key]["heldout_predictions"]

        # convert to pandas series (aligned by time index)
        true_series = true_ts[:plot_len]
        forecast_series = pred_ts[:plot_len]

        plt.figure(figsize=(12, 3))
        plt.plot(np.arange(len(true_series)), true_series, label="True values", color="#4d4d4d")
        plt.plot(np.arange(len(forecast_series)), forecast_series, label="Forecasted values", color="xkcd:azure")

        plt.title(f"Predictions: {key}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.xlim(0, len(true_series))
        plt.legend()

        if save_dir is not None:
            fname = f"{key}_predictions_len{plot_len}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_lstm_predictions(prediction_data, plot_len, n_seqs, save_dir=None):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    keys_to_plot = list(prediction_data.keys())[:n_seqs]

    for key in keys_to_plot:
        true_ts = prediction_data[key]["heldout_y"]
        pred_ts = prediction_data[key]["heldout_predictions"].cpu().numpy()

        # convert to pandas series (aligned by time index)
        true_series = true_ts[:plot_len]
        forecast_series = pred_ts[:plot_len]

        plt.figure(figsize=(12, 3))
        plt.plot(np.arange(len(true_series)), true_series, label="True values", color="#4d4d4d")
        plt.plot(np.arange(len(forecast_series)), forecast_series, label="Forecasted values", color="xkcd:azure")

        plt.title(f"Predictions: {key}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.xlim(0, len(true_series))
        plt.legend()

        if save_dir is not None:
            fname = f"{key}_predictions_len{plot_len}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_chronous_predictions(prediction_data, plot_len, n_seqs, save_dir=None):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    keys_to_plot = list(prediction_data.keys())[:n_seqs]

    for key in keys_to_plot:
        true_ts = prediction_data[key]["heldout_y"]
        pred_ts = prediction_data[key]["heldout_predictions"]

        # convert to pandas series (aligned by time index)
        true_series = true_ts[:plot_len]
        forecast_series = pred_ts[:plot_len]

        plt.figure(figsize=(12, 3))
        plt.plot(np.arange(len(true_series)), true_series, label="True values", color="#4d4d4d")
        plt.plot(np.arange(len(forecast_series)), forecast_series, label="Forecasted values", color="xkcd:azure")

        plt.title(f"Predictions: {key}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.xlim(0, len(true_series))
        plt.legend()

        if save_dir is not None:
            fname = f"{key}_predictions_len{plot_len}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_cp_prediction_intervals(log, target_quantiles, plotting_seq_length, save_dir=None):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for key, item in log.items():

        evaluation_results = item["evaluation_results"]

        for quantile_pair in target_quantiles:
            tuple_quantile_pair = tuple(quantile_pair)
            result = evaluation_results[tuple_quantile_pair]

            lower = np.asarray(result["lower_interval"], dtype=float)
            upper = np.asarray(result["upper_interval"], dtype=float)
            y = np.asarray(result["target_y"], dtype=float)
            hat_y = np.asarray(result["target_predictions"], dtype=float)

            seq_len = min(plotting_seq_length, len(y))
            x = np.arange(len(y))

            if "train_residuals_mu" in result and "train_residuals_std" in result:
                # data was normalize, therefore denormalize data before plotting
                lower_value = lower * result["train_residuals_std"] + result["train_residuals_mu"] + hat_y
                upper_value = upper * result["train_residuals_std"] + result["train_residuals_mu"] + hat_y
            else:
                lower_value = lower + hat_y
                upper_value = upper + hat_y

            fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

            front_slice = slice(0, seq_len)
            end_slice = slice(len(y) - seq_len, len(y))

            slice_configs = [
                (axes[0], front_slice, "from beginning"),
                (axes[1], end_slice, "from end"),
            ]

            for ax, seq_slice, title_suffix in slice_configs:
                x_slice = x[seq_slice]
                y_slice = y[seq_slice]
                hat_y_slice = hat_y[seq_slice]
                lower_slice = lower_value[seq_slice]
                upper_slice = upper_value[seq_slice]

                ax.fill_between(
                    x_slice,
                    lower_slice,
                    upper_slice,
                    color="tab:orange",
                    alpha=0.25,
                    label="prediction interval",
                )
                ax.plot(x_slice, y_slice, color="tab:blue", linewidth=2, label="y")
                ax.plot(
                    x_slice,
                    hat_y_slice,
                    color="tab:orange",
                    linestyle="--",
                    linewidth=2,
                    label="$\hat{y}$",
                )
                ax.set_title("{}: {}".format(title_suffix, key))
                ax.set_xlabel("Time")
                ax.set_xlim(x_slice[0], x_slice[-1] if len(x_slice) > 1 else x_slice[0] + 1)

            axes[0].set_ylabel("Value")
            axes[0].legend()
            axes[1].legend()
            fig.suptitle("{} at {}".format(key, tuple_quantile_pair))
            fig.tight_layout()

            if save_dir is not None:
                quantile_tag = "{}-{}".format(
                    int(tuple_quantile_pair[0] * 100),
                    int(tuple_quantile_pair[1] * 100),
                )
                filename = "{}_q{}_len{}.png".format(key, quantile_tag, seq_len)
                fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
    
