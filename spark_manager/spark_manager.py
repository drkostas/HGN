import os
import shutil
from typing import List, Tuple, Dict
import logging
from functools import reduce
import pyspark
import pyspark.sql
import pyspark.ml.feature
import pyspark.mllib.linalg.distributed
from pyspark.sql.functions import lit, struct
from pyspark.sql.types import StructField, StructType, StringType, LongType, FloatType
from pyspark.storagelevel import StorageLevel
from graphframes import GraphFrame

from color_log.color_log import ColorLog

logger = ColorLog(logging.getLogger('SparkManager'), 'red')
logging.getLogger('py4j').setLevel(logging.INFO)


class SparkManager:
    """Manages the creation of the spark runtime along with any related file
    system operations.
    """
    __slots__ = ('spark_session', 'spark_context', 'sql_context',
                 'graph_name', 'df_data_folder', 'checkpoints_folder', 'feature_names')

    spark_session: pyspark.sql.SparkSession
    spark_context: pyspark.SparkContext
    sql_context: pyspark.sql.SQLContext
    graph_name: str
    df_data_folder: str
    checkpoints_folder: str
    feature_names: List
    loop_counter: int = 0

    def __init__(self, graph_name: str, feature_names: List, df_data_folder: str, checkpoints_folder: str) -> None:
        """The basic constructor. Creates a new instance of SparkManager using
        the specified settings.

        Args:
            graph_name (str):
            feature_names (List):
            df_data_folder (str):
            checkpoints_folder (str):
        """

        logger.info("Initializing SparkManager..")
        # Store object properties
        self.graph_name = graph_name
        self.feature_names = feature_names
        self.df_data_folder = os.path.join(df_data_folder, self.graph_name)
        self.checkpoints_folder = os.path.join(checkpoints_folder, self.graph_name)
        # Delete old files
        self._clean_folder(folder_path=self.df_data_folder)
        self._clean_folder(folder_path=self.checkpoints_folder)
        # Configure spark properties
        conf = pyspark.SparkConf()
        conf.setMaster("local[*]") \
            .setAppName(self.graph_name) \
            .set("spark.submit.deployMode", "client") \
            .set("spark.ui.port", "4040")
        # Instantiate Spark
        self.spark_session = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
        self.spark_context = self.spark_session.sparkContext
        self.sql_context = pyspark.sql.SQLContext(self.spark_context)

    @staticmethod
    def GraphFrame(vertices: pyspark.sql.DataFrame, edges: pyspark.sql.DataFrame) -> GraphFrame:
        """Simply calls the graphframes.GraphFrame :param vertices: :type
        vertices: pyspark.sql.DataFrame :param edges: :type edges:
        pyspark.sql.DataFrame

        Args:
            vertices (pyspark.sql.DataFrame):
            edges (pyspark.sql.DataFrame):
        """

        return GraphFrame(vertices, edges)

    def load_nodes_df(self, path: str, delimiter: str, has_header: bool = False) -> pyspark.sql.DataFrame:
        """Loads the input nodes into a DataFrame.

        Args:
            path (str):
            delimiter (str):
            has_header (bool): If the input file has a header with the column
                names
        """

        logger.debug("Loading nodes_df from path %s.." % path)
        struct_list = [StructField(self.feature_names[0], LongType(), True)]
        for feature in self.feature_names[1:]:
            struct_list.append(StructField(feature, StringType(), True))
        nodes_schema = StructType(struct_list)

        nodes_df = self.sql_context.read.load(path, format="csv", header=has_header, sep=delimiter,
                                              schema=nodes_schema)
        return self.reload_df(df=nodes_df, name="nodes_df")

    def load_edges_df(self, path: str, delimiter: str, has_weights: bool = False,
                      has_header: bool = False) -> pyspark.sql.DataFrame:
        """Loads the input edges into a DataFrame.

        Args:
            path (str):
            delimiter (str):
            has_weights (bool): If the edge of the graph have a weight attribute
            has_header (bool): If the input file has a header with the column
                names
        """

        logger.debug("Loading edges_df from path %s.." % path)
        if has_weights:
            edges_schema = StructType([
                StructField("src", LongType(), True),
                StructField("dst", LongType(), True),
                StructField("weight", FloatType(), True)])
            edges_df = self.sql_context.read.load(path, format="csv", header=has_header, sep=delimiter,
                                                  schema=edges_schema)
        else:
            edges_schema = StructType([
                StructField("src", LongType(), True),
                StructField("dst", LongType(), True)])
            edges_df = self.sql_context.read.load(path, format="csv", header=has_header, sep=delimiter,
                                                  schema=edges_schema)
        # nodesDF = edgesDF.select("src").union(edgesDF.select("dst")).withColumnRenamed('src', 'id').distinct().orderBy("id")
        # elapsed_time = time.time() - start_time
        # print(colored('DataFrames Created: %.3f seconds' % elapsed_time, 'yellow'))
        return self.reload_df(df=edges_df, name="edges_df")

    def create_shortest_paths_df(self, motifs_list: List[pyspark.sql.DataFrame],
                                 has_edge_weights: bool) -> pyspark.sql.DataFrame:
        """Creates the shortest paths DataFrame from a list of motifs.

        Args:
            motifs_list (List[pyspark.sql.DataFrame]):
            has_edge_weights (bool):
        """

        motifs_list_eq = self._add_missing_columns_to_paths_dfs(dfs_list=motifs_list, has_edge_weights=has_edge_weights)
        self.unpersist_all_rdds()
        # motifs = self.union_dfs(motifs_list_eq, 5)
        for motif in motifs_list_eq:
            self.save_to_parquet(df=motif, name="motifs", mode="append", pre_final=True)
        return self.clean_and_reload_df(name="motifs")

    def clean_and_reload_df(self, name: str, df: pyspark.sql.DataFrame = None) -> pyspark.sql.DataFrame:
        """Stores df to temp parquet, drop duplicates and reloads it.

        Args:
            name (str):
            df (pyspark.sql.DataFrame):
        """

        logger.debug("Cleaning and reloading df %s.." % name)
        path = os.path.join(self.df_data_folder, name, str(self.loop_counter))
        if df:
            loaded_df = self.reload_df(df=df, name=name, pre_final=True)
        else:
            loaded_df = self.load_from_parquet(name=name, pre_final=True)
        loaded_df = loaded_df.dropDuplicates()

        self.save_to_parquet(df=loaded_df, name=name, mode="overwrite", pre_final=False)
        if os.path.exists(path + "/" + name + ".pre_final.parquet"):
            shutil.rmtree(path + "/" + name + ".pre_final.parquet", ignore_errors=True)

        return self.load_from_parquet(name=name, pre_final=False)

    def reload_df(self, df: pyspark.sql.DataFrame, name: str, num_partitions: int = None,
                  partition_cols: List[str] = None, pre_final: bool = False) -> pyspark.sql.DataFrame:
        """Saves a DataFrame as parquet and reloads it.

        Args:
            df (pyspark.sql.DataFrame):
            name (str):
            num_partitions (int):
            partition_cols:
            pre_final (bool):
        """

        self.save_to_parquet(df=df, name=name, num_partitions=num_partitions, partition_cols=partition_cols,
                             pre_final=pre_final)
        df = self.load_from_parquet(name=name, pre_final=pre_final)
        df.persist(StorageLevel.MEMORY_AND_DISK)
        return df

    def save_to_parquet(self, df: pyspark.sql.DataFrame, name: str, mode: str = "overwrite",
                        num_partitions: int = None, partition_cols: List[str] = None, pre_final: bool = False):
        """Saves a DataFrame into a parquet file.

        Args:
            df (pyspark.sql.DataFrame):
            name (str):
            mode (str):
            num_partitions (int):
            partition_cols (list):
            pre_final (bool):
        """

        logger.debug("Saving %s to parquet.." % name)
        path = os.path.join(self.df_data_folder, name, str(self.loop_counter))
        if not os.path.exists(path):
            os.makedirs(path)
        if pre_final:
            parquet_name = os.path.join(path, name + ".pre_final.parquet")
        else:
            parquet_name = os.path.join(path, name + ".parquet")

        if partition_cols and num_partitions:
            df.repartition(num_partitions, *partition_cols).write.mode(mode).parquet(parquet_name)
        elif num_partitions and not partition_cols:
            df.repartition(num_partitions).write.mode(mode).parquet(parquet_name)
        elif partition_cols and not num_partitions:
            df.repartition(*partition_cols).write.mode(mode).parquet(parquet_name)
        else:
            df.repartition(1).write.mode(mode).parquet(parquet_name)

    def load_from_parquet(self, name: str, pre_final: bool = False) -> pyspark.sql.DataFrame:
        """Loads a DataFrame from a parquet file.

        Args:
            name (str):
            pre_final (bool):
        """

        logger.debug("Loading from parquet %s.." % name)
        path = os.path.join(self.df_data_folder, name, str(self.loop_counter))
        if pre_final:
            parquet_name = os.path.join(path, name + ".pre_final.parquet")
        else:
            parquet_name = os.path.join(path, name + ".parquet")

        df = self.sql_context.read.format('parquet').load(parquet_name)

        return df

    def unpersist_all_rdds(self) -> None:
        """Unpersists all the rdds using the internal java spark context."""

        logger.debug('Unpersisting all RDDs..')
        [rdd.unpersist() for rdd in list(self.spark_context._jsc.getPersistentRDDs().values())]

    @staticmethod
    def repartition_dfs_list(dfs_list: List[pyspark.sql.DataFrame], num_partitions: int):
        """Repartitions a list of DataFrames into the specified num of
        partitions.

        Args:
            dfs_list (List[pyspark.sql.DataFrame]):
            num_partitions (int):
        """

        return [df.repartition(num_partitions) for df in dfs_list]

    def union_dfs(self, dfs_list: List[pyspark.sql.DataFrame], union_steps: int) -> pyspark.sql.DataFrame:
        """Recursively unifies several DataFrames in the number of specified
        steps.

        Args:
            dfs_list (List[pyspark.sql.DataFrame]):
            union_steps (int): Defines the numbers of union steps - more steps:
                slower but less memory intensive
        """

        return self._reduce_union(*self._recursive_union(dfs_list=dfs_list, union_steps=union_steps))

    @staticmethod
    def _reduce_union(dfs_list: List[pyspark.sql.DataFrame]) -> pyspark.sql.DataFrame:
        """Reduces a list of DataFrames into a single DataFrame using the Union
        function.

        Args:
            dfs_list (List[pyspark.sql.DataFrame]):
        """

        return reduce(pyspark.sql.DataFrame.union, *dfs_list)

    def _recursive_union(self, dfs_list: List[pyspark.sql.DataFrame], union_steps: int) -> List[pyspark.sql.DataFrame]:
        """Implements the recursion function of the DataFrame union.

        Args:
            dfs_list (List[pyspark.sql.DataFrame]):
            union_steps (int): Defines the numbers of union steps - more steps:
                slower but less memory intensive
        """

        unified_dfs_list = []

        for cnt in range(0, len(dfs_list) // union_steps):
            unified_dfs_list.append(self._reduce_union(*dfs_list[cnt * union_steps:(cnt + 1) * union_steps]))

        remaining_unions = len(dfs_list[len(dfs_list) // union_steps * union_steps:])

        if remaining_unions > 0:
            if remaining_unions > union_steps:
                unified_dfs_list = self._recursive_union(dfs_list=unified_dfs_list,
                                                         union_steps=union_steps)
            unified_dfs_list.append(self._reduce_union(dfs_list=dfs_list[len(dfs_list) // union_steps * union_steps:]))

        unified_dfs_list = self.repartition_dfs_list(dfs_list=unified_dfs_list, num_partitions=4)

        if len(unified_dfs_list) > union_steps:
            unified_dfs_list = self._recursive_union(dfs_list=unified_dfs_list,
                                                     union_steps=union_steps)

        return unified_dfs_list

    @staticmethod
    def _add_missing_columns_to_paths_dfs(dfs_list: List[pyspark.sql.DataFrame], has_edge_weights: bool) -> List[
        pyspark.sql.DataFrame]:
        """For a given list of DataFrames containing graph paths, it adds the
        union of all columns to the dfs that are missing them.

        Args:
            dfs_list (List[pyspark.sql.DataFrame]):
            has_edge_weights (bool):
        """

        if has_edge_weights:
            edges_column = struct(*[lit(0).alias('src'), lit(0).alias('dst'), lit(0.0).alias('weight')])
            edges_schema = StructType([
                StructField("src", LongType(), True),
                StructField("dst", LongType(), True),
                StructField("weight", FloatType(), True)
            ])
        else:
            edges_column = struct(*[lit(0).alias('src'), lit(0).alias('dst')])
            edges_schema = StructType([
                StructField("src", LongType(), True),
                StructField("dst", LongType(), True)
            ])

        columns = [df.columns for df in dfs_list]
        column_lengths = [len(column) for column in columns]
        max_length = max(column_lengths)
        max_column = ["e%d" % i for i in range(max_length)]

        for df_count in range(len(dfs_list)):
            if column_lengths[df_count] < max_length:
                missing_columns = list(set(max_column).difference(set(columns[df_count])))
                for missing_column in missing_columns:
                    dfs_list[df_count] = dfs_list[df_count].withColumn(missing_column, edges_column)
                    dfs_list[df_count] = dfs_list[df_count] \
                        .withColumnRenamed(missing_column, '{}_tmp'.format(missing_column))
                    dfs_list[df_count] = dfs_list[df_count].select('*', dfs_list[df_count][
                        '{}_tmp'.format(missing_column)].cast(edges_schema).alias(missing_column)) \
                        .drop('{}_tmp'.format(missing_column))

        return dfs_list

    @staticmethod
    def _clean_folder(folder_path: str) -> None:
        """Removes all files/subfolders from a directory.

        Args:
            folder_path (str):
        """

        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Args:
            exc_type:
            exc_val:
            exc_tb:
        """

        logger.debug("Closing SparkManager..")
        self.spark_context.stop()
