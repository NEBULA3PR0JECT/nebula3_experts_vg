from database.arangodb import DatabaseConnector
from config import NEBULA_CONF

import os


class VisualGrounding:
    def __init__(self) -> None:
        self.ARANGO_HOST = os.getenv('ARANGO_HOST', 'ec2-18-158-123-0.eu-central-1.compute.amazonaws.com')
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
