from abc import ABC, abstractmethod


class AbstractVisualizer(ABC):
    """Manages the plotting of an input GraphFrame"""

    def __init__(self, *args, **kwargs) -> None:
        """Abstract Visualizer

        Args:
            *args:
            **kwargs:
        """

        pass

    @abstractmethod
    def scatter_plot(self, *args, **kwargs):
        """Plots using the specified arguments. :param * args: :param ** kwargs:

        Args:
            *args:
            **kwargs:
        """
        pass
