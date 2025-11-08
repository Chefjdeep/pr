#!/bin/bash

# URLs of raw files on GitHub
MAPPER_URL="https://raw.githubusercontent.com/Chefjdeep/pr/main/mapper.py"
REDUCER_URL="https://raw.githubusercontent.com/Chefjdeep/pr/main/reducer.py"
INPUT_URL="https://raw.githubusercontent.com/Chefjdeep/pr/main/logs.txt"

# Local filenames
INPUT_FILE="logs.txt"

# HDFS paths
HDFS_INPUT_DIR="/user/hadoop/input"
HDFS_OUTPUT_DIR="/user/hadoop/output"

# 1. Download files
curl -O $MAPPER_URL
curl -O $REDUCER_URL
curl -O $INPUT_URL

# 2. Remove previous output if exists
hadoop fs -rm -r $HDFS_OUTPUT_DIR

# 3. Create HDFS input directory
hadoop fs -mkdir -p $HDFS_INPUT_DIR

# 4. Upload input file to HDFS
hadoop fs -put -f $INPUT_FILE $HDFS_INPUT_DIR/

# 5. Run Hadoop streaming
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
    -input $HDFS_INPUT_DIR/$INPUT_FILE \
    -output $HDFS_OUTPUT_DIR \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -file mapper.py \
    -file reducer.py

# 6. Show output
echo "====== Hadoop Output ======"
hadoop fs -cat $HDFS_OUTPUT_DIR/part-00000
