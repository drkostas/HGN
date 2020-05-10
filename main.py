# Core
import os
import traceback
from typing import Tuple, Dict
import logging
import argparse

# Custom classes
from configuration.configuration import Configuration
from color_log.color_log import ColorLog
from spark_manager import spark_manager
from graph_tools import graph_tools

logger = ColorLog(logging.getLogger('Main'), 'green')


def _setup_log(log_path: str = 'logs/output.log', debug: bool = False) -> None:
    log_path = log_path.split(os.sep)
    if len(log_path) > 1:
        try:
            os.makedirs((os.sep.join(log_path[:-1])))
        except FileExistsError:
            pass
    log_filename = os.sep.join(log_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO if not debug else logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ]
                        )


def _argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='A template for python projects.',
        add_help=False)
    # Required Args
    required_arguments = parser.add_argument_group('Required Arguments')
    config_file_params = {
        'type': argparse.FileType('r'),
        'required': True,
        'help': "The configuration yml file"
    }
    required_arguments.add_argument('-c', '--config-file', **config_file_params)
    # Optional args
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('-d', '--debug', action='store_true', help='Enables the debug log messages')
    optional.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def setup() -> Tuple[Dict, Dict, Dict, str]:
    args = _argparser()
    # Temporary logging
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler()]
                        )
    # Load the configuration
    config = Configuration(config_src=args.config_file)
    input_config = config.get_input_configs()[0]
    run_options_config = config.get_run_options_configs()[0]
    output_config = config.get_output_configs()[0]
    options_id_name = "featMinAvg-{featMinAvg}_rLvl1-{rLvl1}_" \
                      "rLvl2-{rLvl2}_betwThres-{betwThres}_feats-{feats}" \
        .format(featMinAvg=run_options_config['feature_min_avg'],
                rLvl1=run_options_config['r_lvl1_thres'],
                rLvl2=run_options_config['r_lvl2_thres'],
                betwThres=run_options_config['betweenness_thres'],
                feats=''.join([feat[:10] for feat in run_options_config['features_to_check'][1:]]))
    modified_graph_name = os.path.join(input_config['name'], options_id_name)
    _setup_log(os.path.join(output_config['logs_folder'], modified_graph_name + '.log'), debug=args.debug)
    return input_config, run_options_config, output_config, modified_graph_name


def load_graph(spark_manager: spark_manager.SparkManager, config: Dict) -> spark_manager.GraphFrame:
    logger.info("Loading the input graph into a GraphFrame")
    nodes_df = spark_manager.load_nodes_df(path=config['nodes']['path'],
                                           delimiter=config['nodes']['delimiter'],
                                           has_header=config['nodes']['has_header'])
    edges_df = spark_manager.load_edges_df(path=config['edges']['path'],
                                           delimiter=config['edges']['delimiter'],
                                           has_weights=config['edges']['has_weights'],
                                           has_header=config['edges']['has_header'])
    return spark_manager.GraphFrame(nodes_df, edges_df)


# def check_betweenness_and_edgeWeight(self, edge_weights, betweenness):
#     print(colored("[check_betweenness_and_edgeWeight] Deciding which edges to delete", "white"))
#     edges_to_delete1 = edge_weights.join(betweenness, [edge_weights.src == betweenness.edges.src,
#                                                        edge_weights.dst == betweenness.edges.dst], "inner")
#     edges_to_delete2 = edge_weights.join(betweenness, [edge_weights.src == betweenness.edges.dst,
#                                                        edge_weights.dst == betweenness.edges.src], "inner")
#     edges_to_delete = edges_to_delete1.union(edges_to_delete2) \
#         .filter("(edge_weight < {0}) OR (edge_weight >= {0} AND betweenness > {1})".format(gv.EDGE_WEIGHT,
#                                                                                            gv.BETWEENNESS_THRESHOLD)) \
#         .select("src", "dst") \
#         .repartition(4, "src").sortWithinPartitions("src")
#     return edges_to_delete

def main_loop(g: spark_manager.GraphFrame,
              sm: spark_manager.SparkManager,
              gt: graph_tools.GraphTools,
              cosine_similarities: spark_manager.pyspark.sql.DataFrame,
              edge_betweenness: spark_manager.pyspark.sql.DataFrame,
              run_options_config: Dict) -> spark_manager.GraphFrame:
    """The main loop."""

    logger.info("Starting the Main Loop..")
    while True:
        # Increase the loop counter (it used to save to different parquets in each loop)
        sm.loop_counter += 1
        logger.info("Loop %s" % sm.loop_counter)
        # Scan neighborhoods and filter edges based on the r metrics
        sm.unpersist_all_rdds()
        lvl1_neighbors, lvl2_neighbors, \
        edges_r = gt.filter_edges_based_on_r_metrics(g=g,
                                                     r_lvl1_thres=run_options_config['r_lvl1_thres'],
                                                     r_lvl2_thres=run_options_config['r_lvl2_thres'])
        edges_r = sm.reload_df(df=edges_r, name='edges_r')
        # Calculate the edge weights
        sm.unpersist_all_rdds()
        edges_weights = gt.calculate_edge_weights(edges_r=edges_r,
                                                  cosine_similarities=cosine_similarities,
                                                  feature_min_avg=run_options_config['feature_min_avg'])
        edges_weights = sm.reload_df(df=edges_weights, name='edges_weights')
        # Delete Edges based on Edge Weights and Edge Betweenness
        raise NotImplementedError()

    return g


def main() -> None:
    """Run the HGN code.

    :Example:
    python main.py -c confs/template_conf.yml [--debug]
    """

    # Initializing
    input_config, run_options_config, output_config, modified_graph_name = setup()
    sm = spark_manager.SparkManager(graph_name=modified_graph_name,
                                    feature_names=input_config['nodes']['feature_names'],
                                    df_data_folder=output_config['df_data_folder'],
                                    checkpoints_folder=output_config['checkpoints_folder'],
                                    has_edge_weights=input_config['edges']['has_weights'])
    gt = graph_tools.GraphTools(sm=sm, max_sp_length=run_options_config['max_sp_length'])
    logger.debug("Modified Graph Name: %s" % modified_graph_name)
    # Load nodes, edges and create GraphFrame
    g = load_graph(spark_manager=sm, config=input_config)
    # Compute Betweenness and Cosine Similarities
    if output_config['cached_init_step']:
        cosine_similarities = sm.load_from_parquet('cosine_similarities')
        edge_betweenness = sm.load_from_parquet('edge_betweenness')
    else:
        # Generate dummy vectors of the input nodes
        dummy_vectors = sm.create_dummy_vectors(nodes_df=g.vertices,
                                                features_to_check=run_options_config['features_to_check'])
        # Calculate the Cosine Similarities of the input edges
        cosine_similarities = gt.calculate_cosine_similarities(dummy_vectors=dummy_vectors,
                                                               edges_df=g.edges)
        # Calculate Edge Betweenness
        landmarks = g.vertices.select("id").rdd.flatMap(lambda x: x).collect()
        edge_betweenness = gt.calculate_edge_betweenness(g=g, landmarks=landmarks)
        # Save and reload Cosine Similarities and Edge Betweenness
        cosine_similarities = sm.reload_df(df=cosine_similarities, name="cosine_similarities")
        edge_betweenness = sm.reload_df(df=edge_betweenness, name="edge_betweenness")
    # Start the Main Loop of the HGN
    g = main_loop(g=g, sm=sm, gt=gt,
                  cosine_similarities=cosine_similarities, edge_betweenness=edge_betweenness,
                  run_options_config=run_options_config)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
