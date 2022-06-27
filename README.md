# nebula3 visual grounding expert

Based on OFA VG:

OFA is a unified multimodal pretrained model that unifies modalities (i.e., cross-modality, vision, language) and tasks (e.g., image generation, visual grounding, image captioning, image classification, text generation, etc.) to a simple sequence-to-sequence learning framework. For more information, please refer to our paper: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework.

https://github.com/OFA-Sys/OFA

## Base module for all experts

## Expert Configuration
- Since experts are started in a microservice container (via gradient or else), it is useless to
  set configuration in the command args, therefore, expert's specific configuration
  (model type, model tune params etc) should come from env.
  Further info can be found in: https://12factor.net/config
- every expert can have something similar to ExpertsConf class like NEBULA_CONF

## CLI
- CLI is now done via the REST api, each expert microservice app has a web server with
  all the apis documented via the /docs (or /redoc) url, which also provide a basic way
  to operate the experts api.
- Postman and other apps can be used to operate the apis
- If you still want to work from terminal, learn curl.

#
- when an expert starts a new taks it has to call self.add_task and when finishes call self.remove_task
- since expert is running in container, all the logs are going to stdout/stderr so that they could be
  seen from outside using docker logs, and that we can aggragate them


# Env params
- EXPERT_RUN_PIPELINE: true/false run pipeline thread (default - false)


# todo:
- add get/set logger level

# Runing example out of MSR-VTT
Pycharm running as sa module "uvicorn"
vg.vg_expert:app --reload --host 0.0.0.0 --port 8889

Env variables 
ARANGO_DB = prodemo
WEB_SERVER =     # empty
ARANGO_HOST 172.83.9.249

 - create an http file api.http
 - From visual studiop copy the following and press the grayed "Send Request

POST http://localhost:8889/predict
content-type: application/json

{
    "movie_id": "Movies/308374",
    "local": false,
    "extra_params": {
        "mdf": 3,
        "caption": "a basketball player jumping"
    }
}

The response will be returned make sure the following msg is typed

INFO:     127.0.0.1:53726 - "POST /predict HTTP/1.1" 200 OK

The groundeing, bounding box, alongwith the log likelihood will be reported back to the u service by the JSON in the TokenRecord format

if variable debug in vg_expert.py is True then also the image with the BB will be plotted

Time measurements:

timing : VG : ~0.5Sec + 3-4 sec download a movie and extract MDF

# For stand alone 
conda install --file environment.yml
Run :
setup.sh
