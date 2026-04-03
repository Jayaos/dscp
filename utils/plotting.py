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