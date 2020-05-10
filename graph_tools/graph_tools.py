from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as pairwise_cosine_similarities

from spark_manager import spark_manager as sm_lib
from color_log.color_log import ColorLog

logger = ColorLog(logging.getLogger('GraphTools'), 'cyan')


class GraphTools:
    """Manages the creation of the spark runtime along with any related file
    system operations.
    """
    __slots__ = ('sm', 'max_sp_length',)

    sm: sm_lib.SparkManager
    max_sp_length: int

    def __init__(self, sm: sm_lib.SparkManager, max_sp_length: int) -> None:
        """The basic constructor. Creates a new instance of GraphTools using the
        specified settings.

        Args:
            sm (sm_lib.SparkManager):
            max_sp_length (int):
        """

        self.sm = sm
        self.max_sp_length = max_sp_length

    @staticmethod
    def calculate_cosine_similarities(dummy_vectors: sm_lib.pyspark.sql.DataFrame,
                                      edges_df: sm_lib.pyspark.sql.DataFrame) \
            -> sm_lib.pyspark.sql.DataFrame:
        """Calculate cosine similarities for all input edges.

        Args:
            dummy_vectors (sm_lib.pyspark.sql.DataFrame):
            edges_df (sm_lib.pyspark.sql.DataFrame):
        """

        logger.info("Calculating Cosine Similarities..")
        edges_df = edges_df.repartition(4, "src").sortWithinPartitions("src")
        dummy_vectors = dummy_vectors.repartition(4, "id").sortWithinPartitions("id")
        # Join edge with the corresponding vectors
        # noinspection PyTypeChecker
        edge_vectors = edges_df.join(other=dummy_vectors,
                                     on=edges_df.src == dummy_vectors.id,
                                     how="inner") \
            .withColumnRenamed('features', 'features_src') \
            .drop("id") \
            .repartition(4, "dst").sortWithinPartitions("dst") \
            .join(other=dummy_vectors,
                  on=edges_df.dst == dummy_vectors.id,
                  how="inner") \
            .withColumnRenamed('features', 'features_dst') \
            .drop("id")

        # Calculate cosine similarity of the each row's two vectors
        udf_pairwise_cosine_similarities = sm_lib.udf(
            f=lambda col1, col2: float(pairwise_cosine_similarities(np.array((col1, col2)))[0][1]),
            returnType=sm_lib.FloatType()
        )
        cosine_similarities = edge_vectors.withColumn('similarity',
                                                      udf_pairwise_cosine_similarities(edge_vectors.features_src,
                                                                                       edge_vectors.features_dst))

        return cosine_similarities.drop("features_src").drop("features_dst")

    def calculate_edge_betweenness(self, g: sm_lib.GraphFrame, landmarks: List[int]) \
            -> sm_lib.pyspark.sql.DataFrame:
        """Calculate edge betweenness.

        Args:
            g (sm_lib.GraphFrame):
            landmarks (List[int]):
        """

        logger.info("Calculating Edge Betweenness (Max Depth: %s).." % self.max_sp_length)
        # Find all pairs of shortest paths
        shortest_paths_df = self.calculate_all_pairs_shortest_paths(g=g, landmarks=landmarks)
        # Use the shortest paths to find betweenness
        return self.find_betweenness_from_shortest_paths(shortest_paths_df=shortest_paths_df)

    def calculate_all_pairs_shortest_paths(self, g: sm_lib.GraphFrame, landmarks: List[int]) \
            -> sm_lib.pyspark.sql.DataFrame:
        """Calculates all pairs of shortest paths for a GraphFrame.

        Args:
            g (sm_lib.GraphFrame):
            landmarks (List[int]):
        """

        logger.info("Calculating all pairs shortest paths..")
        # Find all the shortest paths lengths
        sp_lengths = self.calculate_sp_lengths(g=g, landmarks=landmarks, batch_size=int(len(landmarks) * 0.5))
        # self.max_sp_length = self.get_max_length(sp_lengths)
        # Calculate all the motifs(paths) with lengths between 1 and self.max_sp_length
        possible_motifs = self.generate_all_possible_motifs(g=g)
        self.sm.unpersist_all_rdds()
        # Calculate all-pairs shortest paths using the distances and the motifs
        shortest_paths_list = self.find_shortest_paths_from_motifs(sp_lengths=sp_lengths,
                                                                   possible_motifs=possible_motifs)
        # Create the shortest paths df from a of shortest paths
        logger.info("Creating the shortest paths df from a of shortest paths..")
        shortest_paths_df = self.sm.get_shortest_paths_df(shortest_paths_list=shortest_paths_list)
        self.sm.unpersist_all_rdds()

        return shortest_paths_df

    def calculate_sp_lengths(self, g: sm_lib.GraphFrame, landmarks: List[int], batch_size: int = 100) \
            -> sm_lib.pyspark.sql.DataFrame:
        """Calculates the distances of all the shortest paths

        Args:
            g (sm_lib.GraphFrame):
            landmarks (List[int]):
            batch_size (int):
        """

        g = self.sm.GraphFrame(g.vertices,
                               g.edges.union(g.edges.select(g.edges.dst.alias("src"), g.edges.src.alias("dst"))))
        if batch_size > len(landmarks): batch_size = len(landmarks) - 1
        sp_lengths_list = []
        logger.info("Calculating shortest paths distances for %d nodes at a time.." % batch_size)

        for ind in tqdm(range(0, len(landmarks), batch_size), desc="Calculating shortest paths distances"):
            current_landmarks = []
            for i in range(batch_size):
                current_landmarks.append(landmarks[ind + i])
                if ind + batch_size + 1 > len(landmarks):
                    break

            paths = g.shortestPaths(landmarks=current_landmarks).persist(sm_lib.StorageLevel.MEMORY_AND_DISK)
            sp_lengths_list.append(paths)

        for sp_length_df in sp_lengths_list:
            sp_length_df = sp_length_df.select("id", sm_lib.explode("distances")) \
                .withColumnRenamed("id", "dst") \
                .withColumnRenamed("key", "src") \
                .withColumnRenamed("value", "distance") \
                .filter("distance<={}".format(self.max_sp_length)) \
                .dropDuplicates(['src', 'dst']) \
                .repartition(4, "src").sortWithinPartitions("src")
            self.sm.save_to_parquet(df=sp_length_df, name="sp_lengths", mode="append", pre_final=True)
            sp_length_df.unpersist()
        # sp_lengths_df = self.sm.union_dfs(sp_lengths_list, 5)
        sp_lengths_df = self.sm.clean_and_reload_df(name="sp_lengths")
        return sp_lengths_df.filter("distance>0")

    # def get_max_length(self, df: sm_lib.pyspark.sql.DataFrame) -> int:
    #     """Calculate the max distance."""
    #
    #     logger.info("Calculating the max sp distance..")
    #     max_length_df = df.agg({"distance": "max"})
    #     return max_length_df.collect()[0][0]

    def generate_all_possible_motifs(self, g: sm_lib.GraphFrame) \
            -> Dict[int, sm_lib.pyspark.sql.DataFrame]:
        """Generates all the possible motifs up to the specified length.

        Args:
            g (sm_lib.GraphFrame):
        """

        logger.info("Generating all the possible paths(motifs) with lengths between 1 and %s: " % self.max_sp_length)
        g = self.sm.GraphFrame(g.vertices,
                               g.edges.union(g.edges.select(g.edges.dst.alias("src"),
                                                            g.edges.src.alias("dst"))))

        possible_motifs = {}
        for length in range(1, self.max_sp_length + 1):
            motifPath = self.create_motif(length)
            current_motif = g.find(motifPath).filter("a != z").dropDuplicates()
            current_motif.persist(sm_lib.StorageLevel.MEMORY_AND_DISK)
            possible_motifs[length] = current_motif
        return possible_motifs

    def find_shortest_paths_from_motifs(self, sp_lengths: sm_lib.pyspark.sql.DataFrame,
                                        possible_motifs: Dict[int, sm_lib.pyspark.sql.DataFrame]) \
            -> List[sm_lib.pyspark.sql.DataFrame]:
        """Find the shortest paths based on the sp distances and the possible
        motifs.

        Args:
            sp_lengths (sm_lib.pyspark.sql.DataFrame):
            possible_motifs (Dict[int, sm_lib.pyspark.sql.DataFrame]):
        """

        logger.info("Calculating all-pairs shortest paths based on the distances and the motifs..")
        shortest_paths_list = []
        for length in range(self.max_sp_length, 0, -1):
            sp_lengths = sp_lengths.filter("distance={}".format(length))
            motifs = possible_motifs[length]
            # splitWeights = [0.5 for _ in list(range(length))*10]
            # inner_sp_lengths_list = sp_lengths_to_split.randomSplit(split_weights)
            # for sp_lengths in tqdm(inner_sp_lengthsList):
            motifs.createOrReplaceTempView("motifs")
            sp_lengths.createOrReplaceTempView("sp_lengths")
            # motifs.persist(StorageLevel.MEMORY_AND_DISK)
            # sp_lengths.persist(StorageLevel.MEMORY_AND_DISK)
            motifs = self.sm.sql_context.sql(
                "SELECT m.* FROM motifs m INNER JOIN sp_lengths l ON (m.a.id = l.dst and m.z.id = l.src)") \
                .dropDuplicates(["a", "z"]).orderBy("a", "z")
            self.sm.spark_session.catalog.dropTempView("motifs")
            self.sm.spark_session.catalog.dropTempView("sp_lengths")
            # motifs.unpersist()
            # sp_lengths.unpersist()
            columns_starting_with_e = [c for c in motifs.columns if (c[0] == 'n' or c == 'a' or c == 'z')]
            shortest_paths_list.append(sm_lib.reduce(sm_lib.pyspark.sql.DataFrame.drop,
                                                     columns_starting_with_e,
                                                     motifs))

        return shortest_paths_list

    @staticmethod
    def create_motif(length: int) -> str:
        """Create a motif string.

        Args:
            length (int):
        """

        motif_path = "(a)-[e0]->"
        for i in range(1, length):
            motif_path += "(n%s);(n%s)-[e%s]->" % (i - 1, i - 1, i)
        motif_path += "(z)"
        return motif_path

    def find_betweenness_from_shortest_paths(self, shortest_paths_df: sm_lib.pyspark.sql.DataFrame) \
            -> sm_lib.pyspark.sql.DataFrame:
        """Find the Girvan-Newman edge betweenness using the shortest paths.

        Args:
            shortest_paths_df (sm_lib.pyspark.sql.DataFrame):
        """

        logger.info("Find the Girvan-Newman edge betweenness using the shortest paths..")
        # Put all edges from the shortest paths df into a single column
        one_column_edges_df = self.put_edges_in_a_column(shortest_paths_df=shortest_paths_df)
        # Calculate the edge occurrences which is basically the betweenness per edge
        edge_occurrences_df = self.count_edge_occurrences(one_column_edges_df=one_column_edges_df)
        edge_occurrences_df.persist(sm_lib.StorageLevel.MEMORY_AND_DISK)

        # max_edges_df = find_max_betweenness_edges(edge_occurrences_df)
        # edges_df = filterout_max_edges(edges_df, max_edges_df)

        return edge_occurrences_df

    @staticmethod
    def put_edges_in_a_column(shortest_paths_df: sm_lib.pyspark.sql.DataFrame) \
            -> sm_lib.pyspark.sql.DataFrame:
        """Take the spread-out edges from all the shortest paths columns and put
        them in a single column.

        Args:
            shortest_paths_df (sm_lib.pyspark.sql.DataFrame):
        """

        logger.info("Putting all the shortest path edges into a single column..")
        columns = [x for x in shortest_paths_df.columns if x[0] == 'e']
        one_column_edges_df = shortest_paths_df
        for col in columns[1:]:
            one_column_edges_df = one_column_edges_df.select(columns[0]).union(shortest_paths_df.select(col))
        return one_column_edges_df

    @staticmethod
    def count_edge_occurrences(one_column_edges_df: sm_lib.pyspark.sql.DataFrame) \
            -> sm_lib.pyspark.sql.DataFrame:
        """Count the occurrences of all the distinct edges.

        Args:
            one_column_edges_df (sm_lib.pyspark.sql.DataFrame):
        """

        logger.info("Counting the edge occurrences..")
        columns = one_column_edges_df.columns
        edge_occurrences_df = one_column_edges_df.withColumnRenamed(columns[0], "edges") \
            .groupBy("edges") \
            .agg({"edges": "count"}) \
            .dropna() \
            .withColumnRenamed("count(edges)", "betweenness")
        return edge_occurrences_df

    # def find_max_betweenness_edges(self, edge_occurrences_df):
    #     max_betweenness = edge_occurrences_df.agg({'betweenness': 'max'}) \
    #         .withColumnRenamed("max(betweenness)", "max_betweenness")
    #     max_edges_df = edge_occurrences_df.join(max_betweenness,
    #                                             edge_occurrences_df.betweenness == max_betweenness.max_betweenness,
    #                                             "inner")
    #     columns = max_edges_df.columns
    #     return max_edges_df.select(columns[0])

    # def filterout_maxedges(self, edges_df, max_edges_df):
    #     cond1 = [edges_df.src == max_edges_df.edges.src, edges_df.dst == max_edges_df.edges.dst]
    #     cond2 = [edges_df.src == max_edges_df.edges.dst, edges_df.dst == max_edges_df.edges.src]
    #     return edges_df.join(max_edges_df, cond1, "left_anti") \
    #         .join(max_edges_df, cond2, "left_anti")

    def filter_edges_based_on_r_metrics(self, g: sm_lib.GraphFrame, r_lvl1_thres: float, r_lvl2_thres: float) \
            -> Tuple[sm_lib.pyspark.sql.DataFrame,
                     sm_lib.pyspark.sql.DataFrame,
                     sm_lib.pyspark.sql.DataFrame]:
        """Scan the neighborhoods, calculate the r metrics and filter edges
        based on the metrics.

        Args:
            g (sm_lib.GraphFrame):
            r_lvl1_thres (float):
            r_lvl2_thres (float):
        """

        logger.info("Scanning Neighborhoods..")
        # Find level 1 and 2 neighbors of each edge
        lvl1_neighbors, lvl2_neighbors = self.find_neighbors(g=g)
        # Calculate r metrics using the neighbors and remove edges based on them
        edges_r = self.remove_edges_using_r_metrics(g=g,
                                                    lvl1_neighbors=lvl1_neighbors,
                                                    lvl2_neighbors=lvl2_neighbors,
                                                    r_lvl1_thres=r_lvl1_thres,
                                                    r_lvl2_thres=r_lvl2_thres)

        return lvl1_neighbors, lvl2_neighbors, edges_r

    def find_neighbors(self, g: sm_lib.GraphFrame) -> List[sm_lib.pyspark.sql.DataFrame]:
        """Find level 1 and level neighbors of each node.

        Args:
            g (sm_lib.GraphFrame):
        """

        logger.info("Searching for level 1 and 2 Neighbors..")
        g = sm_lib.GraphFrame(g.vertices,
                              g.edges.union(g.edges.select(g.edges.dst.alias("src"), g.edges.src.alias("dst"))))

        all_lvl_neighbors = list()
        for path_length in range(1, 3):
            motif_struct = self.create_motif(length=path_length)
            if path_length == 1:
                current_lvl_neighbors = g.find(motif_struct).selectExpr("a.id as id", "z.id as dst") \
                    .repartition(4, "id").sortWithinPartitions("id")
            else:
                # Neighbors in level 2 include also the neighbors from level 1
                current_lvl_neighbors = g.find(motif_struct).selectExpr("a.id as id", "z.id as dst", "n0.id as dst_2") \
                    .repartition(4, "id").sortWithinPartitions("id")
                current_lvl_neighbors = current_lvl_neighbors.select("id", "dst") \
                    .unionByName(current_lvl_neighbors.selectExpr("id", "dst_2 as dst"))

            current_lvl_neighbors = current_lvl_neighbors.filter("id != dst") \
                .dropDuplicates() \
                .groupBy("id") \
                .agg(sm_lib.collect_set("dst"), sm_lib.count("dst")) \
                .withColumnRenamed("collect_set(dst)", "neighbors") \
                .withColumnRenamed("count(dst)", "count") \
                .orderBy(sm_lib.desc("count(dst)")) \
                .repartition(4, "id").sortWithinPartitions("id") \
                .join(g.vertices, "id", "full") \
                .select("id", "count", "neighbors") \
                .fillna({'count': '0'}) \
                .withColumn("neighbors_2", sm_lib.coalesce(sm_lib.col("neighbors"),
                                                           sm_lib.array())) \
                .drop("neighbors") \
                .withColumnRenamed("neighbors_2", "neighbors") \
                .repartition(4, "id").sortWithinPartitions("id")
            all_lvl_neighbors.append(current_lvl_neighbors)

        return all_lvl_neighbors

    def remove_edges_using_r_metrics(self, g: sm_lib.GraphFrame,
                                     lvl1_neighbors: sm_lib.pyspark.sql.DataFrame,
                                     lvl2_neighbors: sm_lib.pyspark.sql.DataFrame,
                                     r_lvl1_thres: float, r_lvl2_thres: float) -> sm_lib.pyspark.sql.DataFrame:
        """Calculate the r metrics and remove edges based on them.

        Args:
            g (sm_lib.GraphFrame):
            lvl1_neighbors (sm_lib.pyspark.sql.DataFrame):
            lvl2_neighbors (sm_lib.pyspark.sql.DataFrame):
            r_lvl1_thres (float):
            r_lvl2_thres (float):
        """

        logger.info("Calculating r metrics and removing edges based on them..")

        # Udf declarations
        @sm_lib.udf(returnType=sm_lib.ArrayType(sm_lib.StringType()))
        def udf_merge_neighbors(neighbors_1, neighbors_2, src, dst):
            neighbors_1 = set(neighbors_1)
            neighbors_1.discard(src)
            neighbors_1.discard(dst)
            neighbors_2 = set(neighbors_2)
            neighbors_2.discard(src)
            neighbors_2.discard(dst)
            return list(set(neighbors_1).intersection(neighbors_2))

        udf_add_counts = sm_lib.udf(f=lambda neighbors: len(neighbors), returnType=sm_lib.IntegerType())
        udf_calculate_r_metrics = sm_lib.udf(f=lambda common_count, count: common_count / count if count > 0 else 0.0,
                                             returnType=sm_lib.FloatType())
        udf_keep_edge_condition = sm_lib.udf(lambda r11, r12, r21, r22: (r11 > r_lvl1_thres or r12 > r_lvl1_thres or
                                                                         r21 > r_lvl2_thres or r22 > r_lvl2_thres),
                                             sm_lib.BooleanType())
        # Remove edges based on r metrics
        edges_r = g.edges.join(lvl1_neighbors, lvl1_neighbors.id == g.edges.src, "inner") \
            .selectExpr("src", "dst", "count as count_src", "neighbors as neighbors_src") \
            .join(lvl1_neighbors, lvl1_neighbors.id == g.edges.dst, "inner") \
            .selectExpr("src", "dst", "count_src", "neighbors_src", "count as count_dst", "neighbors as neighbors_dst") \
            .withColumn("common_neighbors",
                        udf_merge_neighbors(sm_lib.col("neighbors_src"), sm_lib.col("neighbors_dst"),
                                            sm_lib.col("src"), sm_lib.col("dst"))) \
            .withColumn("count_common", udf_add_counts(sm_lib.col("common_neighbors"))) \
            .withColumn("r11", udf_calculate_r_metrics(sm_lib.col("count_common"), sm_lib.col("count_src"))) \
            .withColumn("r12", udf_calculate_r_metrics(sm_lib.col("count_common"), sm_lib.col("count_dst"))) \
            .select("src", "dst", "r11", "r12") \
            .repartition(4, "src").sortWithinPartitions("src") \
            .join(lvl2_neighbors, lvl2_neighbors.id == g.edges.src, "inner") \
            .selectExpr("src", "dst", "r11", "r12", "count as count_src", "neighbors as neighbors_src") \
            .repartition(4, "src").sortWithinPartitions("src") \
            .join(lvl2_neighbors, lvl2_neighbors.id == g.edges.dst, "inner") \
            .selectExpr("src", "dst", "r11", "r12", "count_src", "neighbors_src", "count as count_dst",
                        "neighbors as neighbors_dst") \
            .withColumn("common_neighbors",
                        udf_merge_neighbors(sm_lib.col("neighbors_src"), sm_lib.col("neighbors_dst"),
                                            sm_lib.col("src"), sm_lib.col("dst"))) \
            .withColumn("count_common", udf_add_counts(sm_lib.col("common_neighbors"))) \
            .withColumn("r21", udf_calculate_r_metrics(sm_lib.col("count_common"), sm_lib.col("count_src"))) \
            .withColumn("r22", udf_calculate_r_metrics(sm_lib.col("count_common"), sm_lib.col("count_dst"))) \
            .select("src", "dst", "common_neighbors", "r11", "r12", "r21", "r22") \
            .withColumn("keepit", udf_keep_edge_condition(sm_lib.col("r11"), sm_lib.col("r12"),
                                                          sm_lib.col("r21"), sm_lib.col("r22"))) \
            .repartition(4, "src").sortWithinPartitions("src")

        return edges_r

    def calculate_edge_weights(self, edges_r: sm_lib.pyspark.sql.DataFrame,
                               cosine_similarities: sm_lib.pyspark.sql.DataFrame,
                               feature_min_avg: float) -> sm_lib.pyspark.sql.DataFrame:
        """Calculate edge weights based on the cosing similarities and the r
        metrics.

        Args:
            edges_r (sm_lib.pyspark.sql.DataFrame):
            cosine_similarities (sm_lib.pyspark.sql.DataFrame):
            feature_min_avg (float):
        """

        logger.info("Calculating edge weights..")
        # Create edges weights temp view
        edge_weights = edges_r.filter("keepit == False") \
            .selectExpr("src as nb_src", "dst as nb_dst", "common_neighbors") \
            .select("nb_src", "nb_dst", sm_lib.explode("common_neighbors").alias("common_neighbors_exploded")) \
            .repartition(4, "nb_src").sortWithinPartitions("nb_src")
        edge_weights = self.sm.reload_df(name="edge_weights", df=edge_weights)
        edge_weights = edge_weights.repartition(4, "common_neighbors_exploded") \
            .sortWithinPartitions("common_neighbors_exploded")
        edge_weights.createOrReplaceTempView("edge_weights")
        # Create cosine similarities temp view
        cosine_similarities = self.sm.reload_df(name="cosine_similarities", df=cosine_similarities)
        cosine_similarities = cosine_similarities.repartition(4, "src").sortWithinPartitions("src")
        cosine_similarities.createOrReplaceTempView("cosine_similarities")
        # Calculate edge weights j_1
        logger.debug("Calculating edge_weights j_1..")
        j_1 = self.sm.sql_context.sql("""
                          SELECT nb_src, nb_dst, common_neighbors_exploded, src as j1_src, dst as j1_dst, similarity as j1_similarity
                          FROM cosine_similarities c
                          RIGHT JOIN edge_weights ew
                          ON c.src = ew.common_neighbors_exploded
                          """)
        j_1 = self.sm.reload_df(name="j_1", df=j_1)
        cosine_similarities = cosine_similarities.repartition(4, "dst").sortWithinPartitions("dst")
        cosine_similarities.createOrReplaceTempView("cosine_similarities")
        j_1 = j_1.repartition(4, "common_neighbors_exploded").sortWithinPartitions("common_neighbors_exploded")
        j_1.createOrReplaceTempView("j_1")
        # Calculate edge weights j_2
        logger.debug("Calculating edge_weights j_2..")
        j_2 = self.sm.sql_context.sql("""
                          SELECT j1.*, src as j2_src, dst as j2_dst, similarity as j2_similarity
                          FROM cosine_similarities c
                          RIGHT JOIN j_1 j1
                          ON c.dst = j1.common_neighbors_exploded
                          """)
        j_2 = self.sm.reload_df(name="j_2", df=j_2)
        j_2_left = j_2.select("nb_src", "nb_dst", "j1_src", "j1_dst", "j1_similarity")
        j_2_left = j_2_left.repartition(4, "nb_src", "nb_dst").sortWithinPartitions("nb_src", "nb_dst")
        j_2_left.createOrReplaceTempView("j_2_left")
        j_2_right = j_2.select("nb_src", "nb_dst", "j2_src", "j2_dst", "j2_similarity")
        j_2_right = j_2_right.repartition(4, "nb_src", "nb_dst").sortWithinPartitions("nb_src", "nb_dst")
        j_2_right.createOrReplaceTempView("j_2_right")
        # Calculate edge weights j_3
        logger.debug("Calculating edge_weights j_3..")
        j_3 = self.sm.sql_context.sql("""
                          SELECT jleft.nb_src as e1, jleft.nb_dst as e2, jleft.j1_similarity as similarity
                          FROM (
                                  SELECT * FROM j_2_left
                                  WHERE j1_src IS NOT NULL AND j1_dst IS NOT NULL AND j1_similarity IS NOT NULL
                                  ) jleft                  
                          INNER JOIN (
                                      SELECT * FROM j_2_right
                                      WHERE j2_src IS NOT NULL AND j2_dst IS NOT NULL AND j2_similarity IS NOT NULL
                                      ) jright
                          ON jleft.nb_src = jright.nb_src AND 
                              jleft.nb_dst = jright.nb_dst AND 
                              jleft.j1_src = jright.j2_src AND 
                              jleft.j1_dst = jright.j2_dst AND 
                              jleft.j1_similarity = jright.j2_similarity
                          """).dropDuplicates()
        j_3 = self.sm.reload_df(name="j_3", df=j_3)
        # Calculate final edge weights
        logger.debug("Calculating final edge_weights..")
        edge_weights = j_3.selectExpr("e1 as src", "e2 as dst", "similarity") \
            .groupBy("src", "dst") \
            .agg((sm_lib.count(sm_lib.when(j_3.similarity >= feature_min_avg,
                                           j_3.similarity)) / sm_lib.count(j_3.similarity)).alias("edge_weight")
                 )
        return edge_weights.repartition(4, "src", "dst").sortWithinPartitions("src", "dst")

    def filter_out_small_communities(self, g: sm_lib.GraphFrame, min_node_count: int = 10) \
            -> sm_lib.GraphFrame:
        """Filter out communities smaller than the specified min.

        Args:
            g (sm_lib.GraphFrame):
            min_node_count (int):
        """

        # Find Graph's components
        logger.info('Cleaning communities with less than %s nodes..' % min_node_count)
        components = g.connectedComponents()
        grouped_components = components.groupBy('component').agg({"component": "count"}).filter(
            "count(component)>={}".format(min_node_count))
        filtered_nodes = components.join(grouped_components,
                                         components.component == grouped_components.component,
                                         "leftsemi").drop(
            "component")
        filtered_edges = g.edges.join(filtered_nodes, ((g.edges.src == filtered_nodes.id)), "leftsemi") \
            .join(filtered_nodes, ((g.edges.dst == filtered_nodes.id)), "leftsemi")

        return self.sm.GraphFrame(filtered_nodes, filtered_edges).dropIsolatedVertices()
