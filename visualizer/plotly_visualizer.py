import os
import pathlib
from typing import List, Dict, Tuple
import logging
import networkx as nx
import plotly.graph_objs as go
from plotly.offline import plot

from .visualizer import AbstractVisualizer
from color_log.color_log import ColorLog

logger = ColorLog(logging.getLogger('PlotlyVisualizer'), "yellow")


class PlotlyVisualizer(AbstractVisualizer):
    """Manages the plotting of an input GraphFrame"""

    __slots__ = ('plots_folder', 'plot_name', 'plot_path', 'dimensions', 'save_img')

    plots_folder: str
    plot_name: str
    plot_path: str
    dimensions: int
    save_img: bool
    custom_color_scale: List \
        = [[0.0, 'rgb(50, 245, 155)'], [0.01, 'rgb(49, 136, 169)'], [0.02, 'rgb(235, 120, 83)'],
           [0.03, 'rgb(158, 209, 161)'], [0.04, 'rgb(94, 122, 90)'], [0.06, 'rgb(104, 214, 214)'],
           [0.07, 'rgb(17, 122, 145)'], [0.08, 'rgb(72, 220, 3)'], [0.09, 'rgb(250, 45, 188)'],
           [0.1, 'rgb(225, 165, 176)'], [0.11, 'rgb(2, 141, 134)'], [0.12, 'rgb(58, 123, 65)'],
           [0.13, 'rgb(217, 154, 90)'], [0.15, 'rgb(229, 64, 48)'], [0.16, 'rgb(209, 24, 125)'],
           [0.17, 'rgb(255, 39, 0)'], [0.18, 'rgb(121, 47, 71)'], [0.19, 'rgb(2, 10, 198)'],
           [0.2, 'rgb(91, 185, 190)'], [0.21, 'rgb(169, 12, 23)'], [0.22, 'rgb(200, 235, 15)'],
           [0.24, 'rgb(18, 244, 20)'], [0.25, 'rgb(178, 107, 210)'], [0.26, 'rgb(18, 237, 185)'],
           [0.27, 'rgb(182, 44, 36)'], [0.28, 'rgb(76, 211, 88)'], [0.29, 'rgb(151, 168, 188)'],
           [0.3, 'rgb(198, 226, 43)'], [0.31, 'rgb(83, 227, 25)'], [0.33, 'rgb(221, 0, 147)'],
           [0.34, 'rgb(77, 242, 21)'], [0.35, 'rgb(186, 198, 55)'], [0.36, 'rgb(47, 225, 32)'],
           [0.37, 'rgb(216, 75, 181)'], [0.38, 'rgb(89, 37, 156)'], [0.39, 'rgb(142, 152, 11)'],
           [0.4, 'rgb(220, 105, 41)'], [0.42, 'rgb(20, 159, 108)'], [0.43, 'rgb(3, 68, 224)'],
           [0.44, 'rgb(164, 157, 147)'], [0.45, 'rgb(146, 35, 179)'], [0.46, 'rgb(232, 244, 33)'],
           [0.47, 'rgb(211, 103, 60)'], [0.48, 'rgb(51, 7, 143)'], [0.49, 'rgb(130, 49, 253)'],
           [0.51, 'rgb(89, 7, 203)'], [0.52, 'rgb(170, 58, 230)'], [0.53, 'rgb(188, 146, 77)'],
           [0.54, 'rgb(247, 15, 248)'], [0.55, 'rgb(247, 178, 154)'], [0.56, 'rgb(14, 48, 96)'],
           [0.57, 'rgb(176, 175, 185)'], [0.58, 'rgb(87, 111, 154)'], [0.6, 'rgb(200, 28, 129)'],
           [0.61, 'rgb(87, 128, 161)'], [0.62, 'rgb(186, 255, 166)'], [0.63, 'rgb(246, 106, 82)'],
           [0.64, 'rgb(59, 154, 162)'], [0.65, 'rgb(70, 182, 52)'], [0.66, 'rgb(120, 192, 150)'],
           [0.67, 'rgb(21, 17, 76)'], [0.69, 'rgb(116, 46, 232)'], [0.7, 'rgb(2, 58, 206)'],
           [0.71, 'rgb(193, 72, 119)'], [0.72, 'rgb(207, 9, 250)'], [0.73, 'rgb(1, 241, 52)'],
           [0.74, 'rgb(234, 145, 152)'], [0.75, 'rgb(154, 189, 72)'], [0.76, 'rgb(177, 159, 145)'],
           [0.78, 'rgb(118, 8, 240)'], [0.79, 'rgb(32, 17, 21)'], [0.8, 'rgb(96, 255, 25)'],
           [0.81, 'rgb(138, 51, 136)'], [0.82, 'rgb(179, 148, 50)'], [0.83, 'rgb(106, 151, 41)'],
           [0.84, 'rgb(117, 153, 123)'], [0.85, 'rgb(237, 190, 220)'], [0.87, 'rgb(117, 123, 143)'],
           [0.88, 'rgb(51, 44, 133)'], [0.89, 'rgb(254, 8, 83)'], [0.9, 'rgb(45, 139, 111)'],
           [0.91, 'rgb(117, 210, 59)'], [0.92, 'rgb(251, 6, 105)'], [0.93, 'rgb(90, 216, 101)'],
           [0.94, 'rgb(230, 62, 249)'], [0.96, 'rgb(33, 107, 220)'], [0.97, 'rgb(180, 40, 102)'],
           [0.98, 'rgb(70, 20, 200)'], [0.99, 'rgb(225, 43, 169)'], [1.0, 'rgb(8, 244, 127)']]

    def __init__(self, plots_folder: str, plot_name: str, save_img: bool) -> None:
        """The basic constructor. Creates a new instance of SparkManager using
        the specified settings.

        Args:
            plots_folder (str):
            plot_name (str):
            save_img (bool):
        """

        logger.info("Initializing PlotlyVisualizer..")
        self.plots_folder = plots_folder
        self.plot_name = plot_name
        self.save_img = save_img
        self.plot_path = os.path.join(self.plots_folder, self.plot_name)
        pathlib.Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        super().__init__()

    def scatter_plot(self, g_netx: nx.Graph, loop_counter: int, plot_dimensions: int = 3,
                     custom_node_labels: Dict = None) -> None:
        """Creates a 2d or a 3d plotly scatter plot.

        Args:
            g_netx (nx.Graph):
            loop_counter (int):
            plot_dimensions (int):
            custom_node_labels (Dict):
        """

        if plot_dimensions == 2:
            self._scatter_plot_2d(g_netx=g_netx, loop_counter=loop_counter, custom_node_labels=custom_node_labels)
        elif plot_dimensions == 3:
            self._scatter_plot_3d(g_netx=g_netx, loop_counter=loop_counter, custom_node_labels=custom_node_labels)
        else:
            logger.error("Wrong scatter plot dimensions given: %s. Falling back to 2 dimensions..")
            self._scatter_plot_2d(g_netx=g_netx, loop_counter=loop_counter, custom_node_labels=custom_node_labels)

    def _scatter_plot_3d(self, g_netx: nx.Graph, loop_counter: int, custom_node_labels: Dict = None) -> None:
        """Creates a 3d plotly scatter plot.

        Args:
            g_netx (nx.Graph):
            loop_counter (int):
            custom_node_labels (Dict):
        """

        logger.info("Constructing 3d Scatter plot..")
        # Initialize components and edges
        graph_components, \
        nodes_with_colors, \
        node_labels, \
        graph_edges = self._prepare_data(g_netx=g_netx, custom_node_labels=custom_node_labels)
        # Create Scatter Plot
        scatter_layout = nx.spring_layout(G=g_netx, dim=3)
        Xn = [scatter_layout[k][0] for k in list(scatter_layout.keys())]  # x-coordinates of nodes
        Yn = [scatter_layout[k][1] for k in list(scatter_layout.keys())]  # y-coordinates
        Zn = [scatter_layout[k][2] for k in list(scatter_layout.keys())]  # z-coordinates
        Xe = []
        Ye = []
        Ze = []
        for edge in graph_edges:
            Xe += [scatter_layout[edge[0]][0], scatter_layout[edge[1]][0], None]  # x-coordinates of edge ends
            Ye += [scatter_layout[edge[0]][1], scatter_layout[edge[1]][1], None]  # y-coordinates of edge ends
            Ze += [scatter_layout[edge[0]][2], scatter_layout[edge[1]][2], None]  # z-coordinates of edge ends

        lines = go.Scatter3d(x=Xe, y=Ye, z=Ze,
                             mode='lines', line=dict(color='rgb(90, 90, 90)', width=2.5), hoverinfo='none')
        nodes = go.Scatter3d(x=Xn, y=Yn, z=Zn,
                             mode='markers',
                             marker=dict(symbol='circle', size=6,
                                         color=nodes_with_colors, colorscale=self.custom_color_scale,
                                         line=dict(color='rgb(255,255,255)', width=4)),
                             text=node_labels, hoverinfo='text')

        x_axis = dict(backgroundcolor="rgb(200, 200, 230)", gridcolor="rgb(255, 255, 255)",
                      showbackground=True, zerolinecolor="rgb(255, 255, 255)")
        y_axis = dict(backgroundcolor="rgb(230, 200,230)", gridcolor="rgb(255, 255, 255)",
                      showbackground=True, zerolinecolor="rgb(255, 255, 255)")
        z_axis = dict(
            backgroundcolor="rgb(230, 230,200)", gridcolor="rgb(255, 255, 255)",
            showbackground=True, zerolinecolor="rgb(255, 255, 255)")
        camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.85, y=0.85, z=0.85))
        plot_title = self.plot_name + '_Loop-{}'.format(loop_counter) + "_NumOfCommunities-" + str(
            len(graph_components))
        layout = go.Layout(title=plot_title,
                           scene=dict(xaxis=dict(x_axis), yaxis=dict(y_axis), zaxis=dict(z_axis), camera=camera),
                           margin=dict(t=100), hovermode='closest', showlegend=False, )
        data = [lines, nodes]
        fig = go.Figure(data=data, layout=layout)

        plot_name = os.path.join(self.plot_path, "scatter_3d_loop_{}.html".format(loop_counter))
        logger.info("Plotting...")
        if self.save_img:
            plot(figure_or_data=fig, filename=plot_name, validate=False, auto_open=True,
                 image='png', image_filename=self.plot_name, output_type='file', image_width=1700, image_height=800)
        else:
            plot(figure_or_data=fig, filename=plot_name)

    def _scatter_plot_2d(self, g_netx: nx.Graph, loop_counter: int, custom_node_labels: Dict = None) -> None:
        """Creates a 2d plotly scatter plot. :param g_netx: :type g_netx:
        nx.Graph :param loop_counter: :type loop_counter: int :param
        save_as_image: :type save_as_image: bool :param custom_node_labels:
        :type custom_node_labels: Dict

        Args:
            g_netx (nx.Graph):
            loop_counter (int):
            custom_node_labels (Dict):
        """

        logger.info("Constructing 2d Scatter plot..")
        # Initialize components and edges
        graph_components, \
        nodes_with_colors, \
        node_labels, \
        graph_edges = self._prepare_data(g_netx=g_netx, custom_node_labels=custom_node_labels)
        # Create Scatter Plot
        scatter_layout = nx.spring_layout(G=g_netx, dim=2)
        Xn = [scatter_layout[k][0] for k in list(scatter_layout.keys())]  # x-coordinates of nodes
        Yn = [scatter_layout[k][1] for k in list(scatter_layout.keys())]  # y-coordinates
        Xe = []
        Ye = []
        for edge in graph_edges:
            Xe += [scatter_layout[edge[0]][0], scatter_layout[edge[1]][0], None]  # x-coordinates of edge ends
            Ye += [scatter_layout[edge[0]][1], scatter_layout[edge[1]][1], None]  # y-coordinates of edge ends

        lines = go.Scatter(x=Xe, y=Ye,
                           mode='lines', line=dict(color='rgb(90, 90, 90)', width=2.5), hoverinfo='none')
        nodes = go.Scatter(x=Xn, y=Yn,
                           mode='markers',
                           marker=dict(symbol='circle', size=6,
                                       color=nodes_with_colors, colorscale=self.custom_color_scale,
                                       line=dict(color='rgb(255,255,255)', width=4)),
                           text=node_labels, hoverinfo='text')

        x_axis = dict(backgroundcolor="rgb(200, 200, 230)", gridcolor="rgb(255, 255, 255)",
                      showbackground=True, zerolinecolor="rgb(255, 255, 255)")
        y_axis = dict(backgroundcolor="rgb(230, 200,230)", gridcolor="rgb(255, 255, 255)",
                      showbackground=True, zerolinecolor="rgb(255, 255, 255)")
        plot_title = self.plot_name + '_Loop-{}'.format(loop_counter) + "_NumOfCommunities-" + str(
            len(graph_components))
        layout = go.Layout(title=plot_title, width=1080, height=720,
                           scene=dict(xaxis=dict(x_axis), yaxis=dict(y_axis)),
                           margin=dict(t=100), hovermode='closest', showlegend=False)
        data = [lines, nodes]
        fig = go.Figure(data=data, layout=layout)

        plot_name = os.path.join(self.plot_path, "scatter_2d_loop_{}.html".format(loop_counter))
        logger.info("Plotting...")
        if self.save_img:
            plot(figure_or_data=fig, filename=plot_name, validate=False, auto_open=True,
                 image='png', image_filename=self.plot_name, output_type='file', image_width=1700, image_height=800)
        else:
            plot(figure_or_data=fig, filename=plot_name)

    def _prepare_data(self, g_netx: nx.Graph, custom_node_labels: Dict) -> Tuple[List, List, List, nx.graph.EdgeView]:
        """Create the scatter plot data from a NetworkX Graph.

        Args:
            g_netx (nx.Graph):
            custom_node_labels (Dict):
        """

        graph_components = [comp for comp in nx.connected_components(g_netx)]
        graph_edges = g_netx.edges()
        # Prepare the node groups and colors
        communities_dict = {}
        for community_ind, graph_component in enumerate(graph_components):
            for node in graph_component:
                if custom_node_labels:
                    communities_dict[node] = custom_node_labels[node]
                else:
                    communities_dict[node] = community_ind
        nodes_with_colors = []
        node_labels = []
        for node in g_netx.nodes():
            labels_current = node
            node_labels.append("Node: {}".format(labels_current))
            try:
                nodes_with_colors.append(communities_dict[node])
            except KeyError:
                print("Node %d in small community" % node)

        return graph_components, nodes_with_colors, node_labels, graph_edges
