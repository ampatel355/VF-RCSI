"""Create the Monte Carlo histogram for the mean-reversion agent."""

try:
    from monte_carlo_plot import create_monte_carlo_plot
except ModuleNotFoundError:
    from Code.monte_carlo_plot import create_monte_carlo_plot


def main() -> None:
    # Call the shared plotting function for the mean-reversion agent.
    create_monte_carlo_plot("mean_reversion")


if __name__ == "__main__":
    main()
