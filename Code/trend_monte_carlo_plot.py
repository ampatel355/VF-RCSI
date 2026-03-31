"""Create the Monte Carlo histogram for the trend agent."""

try:
    from monte_carlo_plot import create_monte_carlo_plot
except ModuleNotFoundError:
    from Code.monte_carlo_plot import create_monte_carlo_plot


def main() -> None:
    # Call the shared plotting function for the trend agent.
    create_monte_carlo_plot("trend")


if __name__ == "__main__":
    main()
