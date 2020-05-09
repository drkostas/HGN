import os
import shutil
from typing import List, Tuple, Dict
import logging
import time
import pyspark
import pyspark.sql
import pyspark.sql.functions
import pyspark.ml.feature
import pyspark.mllib.linalg.distributed
from pyspark.sql.types import StructField, StructType, StringType, LongType, FloatType
from pyspark.storagelevel import StorageLevel
from graphframes import GraphFrame

logger = logging.getLogger('SparkManager')


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
        self.spark_context.setLogLevel("INFO")
        self.sql_context = pyspark.sql.SQLContext(self.spark_context)

    @staticmethod
    def _clean_folder(folder_path: str) -> None:
        """
        Args:
            folder_path (str):
        """

        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path)

    def GraphFrame(self, vertices: pyspark.sql.DataFrame, edges: pyspark.sql.DataFrame) -> GraphFrame:
        """Simply calls the graphframes.GraphFrame :param vertices: :type
        vertices: pyspark.sql.DataFrame :param edges: :type edges:
        pyspark.sql.DataFrame

        Args:
            vertices (pyspark.sql.DataFrame):
            edges (pyspark.sql.DataFrame):
        """

        return GraphFrame(vertices, edges)

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

        path = os.path.join(self.df_data_folder, name, str(self.loop_counter))
        if pre_final:
            parquet_name = os.path.join(path, name + ".pre_final.parquet")
        else:
            parquet_name = os.path.join(path, name + ".parquet")

        df = self.sql_context.read.format('parquet').load(parquet_name)

        return df

    def reload_df(self, df: pyspark.sql.DataFrame, name: str, num_partitions: int = None,
                  partition_cols: List[str] = None, pre_final: bool = False) -> pyspark.sql.DataFrame:
        """
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

    def load_nodes_df(self, path: str, delimiter: str, has_header: bool = False) -> pyspark.sql.DataFrame:
        """Loads the input nodes into a DataFrame.

        Args:
            path (str):
            delimiter (str):
            has_header (bool): If the input file has a header with the column
                names
        """

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

    def unpersist_rdds(self) -> None:
        """Unpersists all the rdds using the internal java spark context."""

        [rdd.unpersist() for rdd in list(self.spark_context._jsc.getPersistentRDDs().values())]

    def clean_and_reload_df(self, df: pyspark.sql.DataFrame, name: str) -> pyspark.sql.DataFrame:
        """Stores df to temp parquet, drop duplicates and reloads it.

        Args:
            df (pyspark.sql.DataFrame):
            name (str):
        """

        path = os.path.join(self.df_data_folder, name, str(self.loop_counter))
        loaded_df = self.reload_df(df=df, name=name, pre_final=True)

        loaded_df = loaded_df.dropDuplicates()

        self.save_to_parquet(df=loaded_df, name=name, mode="overwrite", pre_final=False)
        if os.path.exists(path + "/" + name + ".pre_final.parquet"):
            shutil.rmtree(path + "/" + name + ".pre_final.parquet", ignore_errors=True)

        return self.load_from_parquet(name=name, pre_final=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Args:
            exc_type:
            exc_val:
            exc_tb:
        """
        self.spark_context.stop()
