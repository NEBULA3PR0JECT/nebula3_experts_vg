from database.arangodb import DatabaseConnector
from config import NEBULA_CONF

import os

"""

--nproc_per_node=1
--master_port=6081
../../evaluate.py
../../dataset/refcoco_data/refcoco_val.tsv
--path=../../checkpoints/refcoco_large_best.pt
--user-dir=../../models/ofa/ofa_module
--task=refcoco
--batch-size=16
--log-format=simple
--log-interval=10
--seed=7
--gen-subset='refcoco_val'
--results-path=../../results/refcoco
--beam=5
--min-len=4
--max-len-a=0
--max-len-b=4
--no-repeat-ngram-size=3
--fp16
--num-workers=0
--inference-pipeline
--model-overrides="{\"data\":\"../../dataset/refcoco_data/refcoco_val.tsv\",\"bpe_dir\":\"../../utils/BPE\",\"selected_cols\":\"0,4,2,3\"}"
"""

class VisualGrounding:
    def __init__(self) -> None:
        self.ARANGO_HOST = os.getenv('nproc_per_node', '1')
        self.ARANGO_PORT = os.getenv('ARANGO_PORT', '8529')
        self.ARANGO_PROXY_PORT = os.getenv('ARANGO_PROXY_PORT', '80')
        self.ARANGO_DB = os.getenv('ARANGO_DB', "nebula_development")
        self.DEFAULT_ARANGO_USER = os.getenv('DEFAULT_ARANGO_USER')
        self.DEFAULT_ARANGO_PASSWORD = os.getenv('DEFAULT_ARANGO_PASSWORD')
        self.ELASIC_HOST = os.getenv('ELASIC_HOST',
                                     'http://tnnb2_master:NeBuLa_2@ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:9200/')
        self.ELASTIC_INDEX = os.getenv('ELASTIC_INDEX', "datadriven")
        self.S3BUCKET = os.getenv('S3BUCKET', "nebula-frames")
        self.MLV_SERVER = os.getenv('MLV_SERVER', 'ec2-3-123-129-35.eu-central-1.compute.amazonaws.com')
        self.MLV_PORT = os.getenv('MLV_PORT', '19530')  # default value
        self.WEB_SERVER = os.getenv('WEB_SERVER', 'http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/')
