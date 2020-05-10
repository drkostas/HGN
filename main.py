# Core
import os
import traceback
from typing import Tuple, Dict
import logging
import argparse
import pandas as pd
# Custom classes
from configuration.configuration import Configuration
from color_log.color_log import ColorLog
from spark_manager.spark_manager import SparkManager

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
        .format(featMinAvg=run_options_config['feaure_min_avg'],
                rLvl1=run_options_config['r_lvl1'],
                rLvl2=run_options_config['r_lvl2'],
                betwThres=run_options_config['betweenness_thres'],
                feats=''.join([feat[:10] for feat in run_options_config['features_to_check'][1:]]))
    modified_graph_name = os.path.join(input_config['name'], options_id_name)
    _setup_log(os.path.join(output_config['logs_folder'], modified_graph_name + '.log'), debug=args.debug)
    return input_config, run_options_config, output_config, modified_graph_name


def load_graph(spark_manager: SparkManager, config: Dict) -> SparkManager.GraphFrame:
    logger.info("Loading the input graph into a GraphFrame")
    nodes_df = spark_manager.load_nodes_df(path=config['nodes']['path'],
                                           delimiter=config['nodes']['delimiter'],
                                           has_header=config['nodes']['has_header'])
    edges_df = spark_manager.load_edges_df(path=config['edges']['path'],
                                           delimiter=config['edges']['delimiter'],
                                           has_weights=config['edges']['has_weights'],
                                           has_header=config['edges']['has_header'])
    return spark_manager.GraphFrame(nodes_df, edges_df)


def main() -> None:
    """
    :Example:
    python main.py -c confs/template_conf.yml [--debug]
    """

    # Initializing
    input_config, run_options_config, output_config, modified_graph_name = setup()
    sm = SparkManager(graph_name=modified_graph_name,
                      feature_names=input_config['nodes']['feature_names'],
                      df_data_folder=output_config['df_data_folder'],
                      checkpoints_folder=output_config['checkpoints_folder'])
    logger.debug("Modified Graph Name: %s" % modified_graph_name)
    # Load nodes, edges and create GraphFrame
    g = load_graph(spark_manager=sm, config=input_config)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
