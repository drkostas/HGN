# Hybrid Girvan Newman
[![CircleCI](https://circleci.com/gh/drkostas/HGN/tree/master.svg?style=svg)](https://circleci.com/gh/drkostas/HGN/tree/master)
[![GitHub license](https://img.shields.io/badge/license-GNU-blue.svg)](https://raw.githubusercontent.com/drkostas/HGN/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
    + [Environment Variables](#env_variables)
+ [Installing, Testing, Building](#installing)
    + [Available Make Commands](#check_make_commamnds)
    + [Clean Previous Builds](#clean_previous)
    + [Venv and Requirements](#venv_requirements)
    + [Run the tests](#tests)
    + [Build Locally](#build_locally)
+ [Running locally](#run_locally)
	+ [Configuration](#configuration)
	+ [Execution Options](#execution_options)	
+ [Deployment](#deployment)
+ [Continuous Ιntegration](#ci)
+ [Todo](#todo)
+ [Built With](#built_with)
+ [License](#license)
+ [Acknowledgments](#acknowledgments)

## About <a name = "about"></a>

Hybrid Girvan Newman. Code for the paper "[A Distributed Hybrid Community Detection Methodology for Social Networks.](https://www.mdpi.com/1999-4893/12/8/175)"
<br><br>
The proposed methodology is an iterative, divisive community detection process that combines the network topology features 
of loose similarity and local edge betweenness measure, along with the user content information in order to remove the 
inter-connection edges and thus unravel the subjacent community structure. Even if this iterative process might sound 
computationally over-demanding, its application is certainly not prohibitive, since it can be safely concluded 
from the experimentation results that the aforementioned measures are that well-informative and highly representative, 
so merely few iterations are required to converge to the final community hierarchy at any case.
<br><br>
Implementation last tested with [Python 3.6](https://www.python.org/downloads/release/python-36), 
[Apache Spark 2.4.5](https://spark.apache.org/docs/2.4.5/) 
and [GraphFrames 0.8.0](https://github.com/graphframes/graphframes/tree/v0.8.0)

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development 
and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python = 3.6, Apache Spark = 2.4.5, GraphFrames = 0.8.0 
and any Bash based shell (e.g. zsh) installed. For Apache Spark = 2.4.5 you will also need Java 8.

```
$ python3.6 -V
Python 3.6.9

echo $SHELL
/usr/bin/zsh
```

### Set the required environment variables <a name = "env_variables"></a>

In order to run the [main.py](main.py) or the tests you will need to set the following 
environmental variables in your system (or in the [spark.env file](spark.env)):

```bash
$ export SPARK_HOME="<Path to Spark Home>"
$ export PYSPARK_SUBMIT_ARGS="--packages graphframes:graphframes:0.8.0-spark2.4-s_2.11 pyspark-shell"
$ export JAVA_HOME="<Path to Java 8>"

$ cd $SPARK_HOME

/usr/local/spark                                                                                                                                                                                                                                             
$ ./bin/pyspark --version
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.5
      /_/
                        
Using Scala version 2.11.12, OpenJDK 64-Bit Server VM, 1.8.0_252
Branch HEAD
Compiled by user centos on 2020-02-02T19:38:06Z
Revision cee4ecbb16917fa85f02c635925e2687400aa56b
Url https://gitbox.apache.org/repos/asf/spark.git
Type --help for more information.

```

## Installing, Testing, Building <a name = "installing"></a>

All the installation steps are being handled by the [Makefile](Makefile).

<i>If you don't want to go through the setup steps and finish the installation and run the tests,
execute the following command:</i>

```bash
$ make install server=local
```

<i>If you executed the previous command, you can skip through to the [Running locally](#run_locally) section.</i>

### Check the available make commands <a name = "check_make_commamnds"></a>

```bash
$ make help

-----------------------------------------------------------------------------------------------------------
                                              DISPLAYING HELP                                              
-----------------------------------------------------------------------------------------------------------
make delete_venv
       Delete the current venv
make create_venv
       Create a new venv for the specified python version
make requirements
       Upgrade pip and install the requirements
make run_tests
       Run all the tests from the specified folder
make setup
       Call setup.py install
make clean_pyc
       Clean all the pyc files
make clean_build
       Clean all the build folders
make clean
       Call delete_venv clean_pyc clean_build
make install
       Call clean create_venv requirements run_tests setup
make help
       Display this message
-----------------------------------------------------------------------------------------------------------
```

### Clean any previous builds <a name = "clean_previous"></a>

```bash
$ make clean server=local
make delete_venv
make[1]: Entering directory '/home/drkostas/Projects/HGN'
Deleting venv..
rm -rf venv
make[1]: Leaving directory '/home/drkostas/Projects/HGN'
make clean_pyc
make[1]: Entering directory '/home/drkostas/Projects/HGN'
Cleaning pyc files..
find . -name '*.pyc' -delete
find . -name '*.pyo' -delete
find . -name '*~' -delete
make[1]: Leaving directory '/home/drkostas/Projects/HGN'
make clean_build
make[1]: Entering directory '/home/drkostas/Projects/HGN'
Cleaning build directories..
rm --force --recursive build/
rm --force --recursive dist/
rm --force --recursive *.egg-info
make[1]: Leaving directory '/home/drkostas/Projects/HGN'

```

### Create a new venv and install the requirements <a name = "venv_requirements"></a>

```bash
$ make create_venv server=local
Creating venv..
python3.6 -m venv ./venv

$ make requirements server=local
Upgrading pip..
venv/bin/pip install --upgrade pip wheel setuptools
Collecting pip
.................
```

### Run the tests <a name = "tests"></a>

The tests are located in the `tests` folder. To run all of them, execute the following command:

```bash
$ make run_tests server=local
source venv/bin/activate && \
.................
```

### Build the project locally <a name = "build_locally"></a>

To build the project locally using the setup.py command, execute the following command:

```bash
$ make setup server=local
venv/bin/python setup.py install '--local'
running install
.................
```

## Running the code locally <a name = "run_locally"></a>

In order to run the code now, you should place under the [data/input_graphs](data/input_graphs) the graph you 
want the communities to be identified from.<br>
You will also only need to create a yml file for any new graph before executing the [main.py](main.py).

### Modifying the Configuration <a name = "configuration"></a>

There two already configured yml files: [confs/quakers.yml](confs/quakers.yml) 
and [confs/hamsterster.yml](confs/hamsterster.yml) with the following structure:

```yaml
tag: dev  # Required
spark:
  - config:  # The spark settings
      spark.master: local[*]  # Required
      spark.submit.deployMode: client  # Required
      spark_warehouse_folder: data/spark-warehouse  # Required
      spark.ui.port: 4040
      spark.driver.cores: 5
      spark.driver.memory: 8g
      spark.driver.memoryOverhead: 4096
      spark.driver.maxResultSize: 0
      spark.executor.instances: 2
      spark.executor.cores: 3
      spark.executor.memory: 4g
      spark.executor.memoryOverhead: 4096
      spark.sql.broadcastTimeout: 3600
      spark.sql.autoBroadcastJoinThreshold: -1
      spark.sql.shuffle.partitions: 4
      spark.default.parallelism: 4
      spark.network.timeout: 3600s
    dirs:
      df_data_folder: data/dataframes  # Folder to store the DataFrames as parquets
      spark_warehouse_folder: data/spark-warehouse
      checkpoints_folder: data/checkpoints
      communities_csv_folder: data/csv_data  # Folder to save the computed communities as csvs
input:
  - config:  # All properties required
      name: Quakers
      nodes:
        path: data/input_graphs/Quakers/quakers_nodelist.csv2  # Path to the nodes file
        has_header: true  # Whether they have a header with the attribute names
        delimiter: ','
        encoding: ISO-8859-1
        feature_names:  # You can rename the attribute names (the number should be the same as the original)
          - id
          - Historical_Significance
          - Gender
          - Birthdate
          - Deathdate
          - internal_id
      edges:
        path: data/input_graphs/Quakers/quakers_edgelist.csv2  # Path to the edges file
        has_header: true  # Whether they have a header with the source and dest
        has_weights: false  # Whether they have a weight column
        delimiter: ','
    type: local
run_options:  # All properties required
  - config:
      cached_init_step: false  # Whether the cosine similarities and edge_betweenness been already been computed
      # See the paper for info regarding the following attributes
      feature_min_avg: 0.33
      r_lvl1_thres: 0.50
      r_lvl2_thres: 0.85
      max_edge_weight: 0.50
      betweenness_thres: 10
      max_sp_length: 2
      min_comp_size: 2 
      max_steps: 30  # Max steps for the algorithm to run if it doesn't converge
      features_to_check:  # Which attributes to take into consideration for the cosine similarities
        - id
        - Gender
output:  # All properties required
  - config:
      logs_folder: data/logs
      save_communities_to_csvs: false  # Whether to save the computed communities in csvs or not
      visualizer:
        dimensions: 3  # Dimensions of the scatter plot (2 or 3)
        save_img: true
        folder: data/plots
        steps:  # The steps to plot
          - 0   # The step before entering the main loop
          - -1  # The Last step
```

The `!ENV` flag indicates that a environmental value follows. For example you can set: <br>`logs_folder: !ENV ${LOGS_FOLDER}`<br>
You can change the values/environmental var names as you wish.
If a yaml variable name is changed/added/deleted, the corresponding changes should be reflected 
on the [Configuration class](configuration/configuration.py) and the [yml_schema.json](configuration/yml_schema.json) too.

### Execution Options <a name = "execution_options"></a>

First, make sure you are in the created virtual environment:

```bash
$ source venv/bin/activate
(venv) 
OneDrive/Projects/HGN  dev 

$ which python
/home/drkostas/Projects/HGN/venv/bin/python
(venv) 
```

Now, in order to run the code you can either call the `main.py` directly, or the `HGN` console script.

```bash
$ python main.py -h
usage: main.py -c CONFIG_FILE [-d] [-h]

A Distributed Hybrid Community Detection Methodology for Social Networks.

Required Arguments:
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        The configuration yml file

Optional Arguments:
  -d, --debug           Enables the debug log messages
  -h, --help            Show this help message and exit


# Or

$ hgn --help
usage: hgn -c CONFIG_FILE [-d] [-h]

A Distributed Hybrid Community Detection Methodology for Social Networks.

Required Arguments:
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        The configuration yml file

Optional Arguments:
  -d, --debug           Enables the debug log messages
  -h, --help            Show this help message and exit

```

## Deployment <a name = "deployment"></a>

It is recommended that you deploy the application to a Spark Cluster.<br>Please see: 
- [Spark Cluster Overview \[Apache Spark Docs\]](https://spark.apache.org/docs/latest/cluster-overview.html)
- [Apache Spark on Multi Node Cluster \[Medium\]](https://medium.com/ymedialabs-innovation/apache-spark-on-a-multi-node-cluster-b75967c8cb2b)
- [Databricks Cluster](https://docs.databricks.com/clusters/index.html)
- [Flintrock \[Cheap & Easy EC2 Cluster\]](https://github.com/nchammas/flintrock)

## Continuous Integration <a name = "ci"></a>

For the continuous integration, the <b>CircleCI</b> service is being used. 
For more information you can check the [setup guide](https://circleci.com/docs/2.0/language-python/). 

Again, you should set the [above-mentioned environmental variables](#env_variables) ([reference](https://circleci.com/docs/2.0/env-vars/#setting-an-environment-variable-in-a-context))
and for any modifications, edit the [circleci config](/.circleci/config.yml).

## TODO <a name = "todo"></a>

Read the [TODO](TODO.md) to see the current task list.

## Built With <a name = "built_with"></a>

* [Apache Spark 2.4.5](https://spark.apache.org/docs/2.4.5/) - Fast and general-purpose cluster computing system
* [GraphFrames 0.8.0](https://github.com/graphframes/graphframes/tree/v0.8.0) - A package for Apache Spark which provides DataFrame-based Graphs.
* [CircleCI](https://www.circleci.com/) - Continuous Integration service


## License <a name = "license"></a>

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.
