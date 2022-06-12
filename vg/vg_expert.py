# uvicorn vg.vg_expert:app --reload --host 0.0.0.0 --port 8889
import os
import sys
import json
from fastapi import FastAPI
import urllib
from PIL import Image
from nebula3_experts.experts.service.base_expert import BaseExpert
from nebula3_experts.experts.app import ExpertApp
from nebula3_experts.experts.common.models import ExpertParam, TokenRecord
from nebula3_database.config import NEBULA_CONF
from nebula3_database.movie_db import MOVIE_DB
sys.path.append("/home/hanoch/projects/nebula3_experts_vg/nebula3_experts/nebula3_pipeline")
sys.path.append("/home/hanoch/projects/nebula3_experts_vg/nebula3_experts/nebula3_pipeline/nebula3_database")
sys.path.append("home/hanoch/projects/OFA_fork/OFA/run_scripts/refcoco")
sys.path.append("/home/hanoch/projects/nebula3_experts_vg/vg")
sys.path.append("/home/hanoch/projects/nebula3_experts_vg/vg/run_scripts/refcoco")
from .visual_grounding_inference import OfaMultiModalVisualGrounding
import cv2
# remove for microservice, enable for vscode container
#sys.path.remove("/notebooks")

# sys.path.append("/notebooks/tracker/autotracker")
# sys.path.append("/notebooks/tracker/autotracker/tracking/../../..")

# import tracker.autotracker as at

# ACTION_DETECT = 'detect'
# ACTION_TRACK = 'track'
# ACTION_DEPTH = 'depth'
# ACTION_ALL = 'all' # track+depth

""" Predict params
@param: predict_every: how many frames to track before accepting detection model detections.
@param: merge_iou_threshold: the IOU score threhold for merging items during tracking.
@param: refresh_on_detect: if True, removes all tracked items that were not found by the detection model.
@param: tracker_type - TRACKER_TYPE_KCF / TRACKER_TYPE_CSRT
@param: batch_size
@param: step - array of steps: [detect,track,depth]
"""

class VisualGroundingExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        self.vg_engine = OfaMultiModalVisualGrounding()
        # self.model = self.load_model()
        self.tracker_dispatch_dict = {}
        # after init all
        self.set_active()
        # Database interface for movie download
        config = NEBULA_CONF()
        self.url_prefix = config.get_webserver() #self.url_prefix = self.db_conf.get_webserver()
        self.nre = MOVIE_DB()
        self.db = self.nre.db
        self.temp_file = "/tmp/file.mp4"

    def _download_video_file(self, movie_id):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        # query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc'.format(movie)
        # cursor = self.db.aql.execute(query)
        movie = self.movie_db.get_movie(movie_id)
        url_prefix = self.url_prefix
        if movie:
            try:
                url_link = url_prefix + movie['url_path']
                url_link = url_link.replace(".avi", ".mp4")
                print(url_link)
                urllib.request.urlretrieve(url_link, self.temp_file)
                video = cv2.VideoCapture(self.temp_file)
                fps = video.get(cv2.CAP_PROP_FPS)
            except:
                print(f'An exception occurred while fetching {url_link}')
                result = False
        # return result

        print(fps)
        return (fps, url_link, video)

    def _download_and_get_minfo(self, mid, to_print=False):
        # Download the video locally
        fps, url_link, video = self._download_video_file(mid)
        movie_info = self.nre.get_movie_info(mid)
        fn = self.temp_file
        if to_print:
            print(f"Movie info: {movie_info}")
            print(f"fn path: {fn}")
        return movie_info, fps, fn, video

    def get_name(self):
        return "VisualGroundingExpert"

    def add_expert_apis(self, app: FastAPI):
        pass

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        vg_param, error = self.parse_vg_params(expert_params)
        if error:
            movie_id = expert_params['movie_id']if  expert_params['movie_id']else ''
            return { 'error': f'error {error} for movie: {movie_id}'}

        movie_info, fps, fn, video = self._download_and_get_minfo(expert_params.movie_id, to_print=True)
        bb = list()
        for elem_inx, elem in enumerate(movie_info['scene_elements']):
            mdfs = movie_info['mdfs'][elem_inx]
            for mdf in mdfs:
                if mdf == vg_param['mdf']:
                    feature_mdfs = []
                    video.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = video.read()  # Read the frame
                    bb, scores, lprob = self.vg_engine.find_visual_grounding(Image.fromarray(frame_), vg_param['caption'])

        if bb == []:
            {'error': f"movie frames not found: {vg_param['movie_id']}"}
        return {'result': bb, 'error': error}

    def parse_vg_params(self, expert_params: ExpertParam):
        error = None
        if (expert_params.movie_id is None):
            error = 'no movie id'
            return None, error
        if (expert_params.extra_params is None):
            error = 'no extra_params id'
            return None, error

        mdf = expert_params.extra_params['mdf']
        caption = expert_params.extra_params['caption']
        movie_id = expert_params.movie_id

        vg_param = {'movie_id': movie_id, 'mdf': mdf, 'caption': caption}

        return vg_param, error

    def handle_action_on_movie(self,
                               params,
                               movie_fetched: bool,
                               action_func,
                               transform_func):
        """ on movie
        Args:
            params (StepParam): _description_
            movie_fetched (_type_): indicated if a movie and it's frames already fetched
            since this method can be called for each type: detection/tracking/etc'
        Returns:
            result or error
        """
        error_msg = None
        result = None
        if params['movie_id']is None:
            self.logger.error(f'missing movie_id')
            return { 'error': f"movie frames not found: {params['movie_id']}"}
        try:
            if not movie_fetched:
                movie, mdf = self.get_movie_and_frames(params["movie_id"])
            if movie and mdf:
                # now calling action function
                action_result = action_func(params)
                # now transforming results data
                result = transform_func(action_result, params)
            else:
                error_msg = f"no frames for movie: {params['movie_id']}"
                self.logger.warning(error_msg)
        except Exception as e:
            error_msg = f"exception: {e} on movie: {params['movie_id']}"
            self.logger.error(error_msg)
        finally:
            self.tracker_dispatch_dict.pop(params['movie_id'])
        return result, error_msg

    def get_movie_and_frames(self, movie_id: str):
        """get movie from db and load movie frames from remote if not exists
        Args:
            movie_id (str): the movie id
        Raises:
            ValueError: _description_
        Returns:
            _type_: movie and number of frames
        """
        movie = self.movie_db.get_movie(movie_id)
        mdf = self.movie_db.get_mdfs(movie_id)
        if 0:
            self.tracker_dispatch_dict[movie_id] = {}
            # loading the movie frames
            num_frames = self.movie_s3.downloadDirectoryFroms3(movie_id)
        return movie, mdf

    def get_vg(self):

        return
    # def detect(self, detect_params: StepParam):
    #     """detector step
    #     Args:
    #         detect_params (StepParam): _description_
    #     Returns:
    #         aggs: _description_
    #     """
    #     return self.model.predict_video(detect_params['movie_id'],
    #                              batch_size = detect_params.batch_size,
    #                              pred_every = detect_params.detect_every,
    #                              show_pbar = False)
    #
    # def transform_detection_result(self, detection_result, detect_params: StepParam):
    #     """transform detection result to the token db format
    #     Args:
    #         detection_result (_type_): _description_
    #         output (_type_): json/db - transforming for json output or for db
    #     """
    #     # print(detection_result)
    #     detections = {}
    #     result = list()
    #     for detection in detection_result:
    #         detection_boxes = detection['detection_boxes']
    #         detection_scores = detection['detection_scores']
    #         detection_classes = detection['detection_classes']
    #         for idx in range(len(detection_classes)):
    #             cls = detection_classes[idx]
    #             bbox = detection_boxes[idx]
    #             score = detection_scores[idx]
    #             element = {'bbox': bbox.tolist(), 'score': float(score.flat[0]) }
    #             if cls in detections:
    #                 detections[cls].append(element)
    #             else:
    #                 detections[cls] = [element]
    #             tr = TokenRecord(detect_params['movie_id'],
    #                             0, 0, self.get_name(),
    #                             detections[cls],
    #                             cls,
    #                             {'class': 'Object'})
    #             result.append(tr)
    #     return result

    # def track(self, track_params: StepParam):
    #     track_data = at.tracking_utils.MultiTracker.track_video_objects(
    #             video_path=track_params['movie_id'],
    #             detection_model=self.model,
    #             detect_every=track_params.detect_every,
    #             merge_iou_threshold=track_params.merge_iou_threshold,
    #             tracker_type=track_params.tracker_type,
    #             refresh_on_detect=track_params.refresh_on_detect,
    #             show_pbar=False,
    #             logger=self.logger
    #         )
    #     return track_data

    def transform_tracking_result(self, tracking_result, vg_params):
        # print(tracking_result)
        result = list()
        for oid, data in tracking_result.items():
            label = data['class'] + str(oid)
            print(data['boxes'])
            print(data['scores'])
            print(label)
            bbox = dict()
            for index, box in data['boxes'].items():
                bbox[index] = { 'score': data['scores'][index], 'bbox': box}
            tr = TokenRecord(vg_params['movie_id'],
                             0, 0, self.get_name(),
                             bbox,
                             label,
                             {'class': 'Object'})
            result.append(tr)
        return result

vg_expert = VisualGroundingExpert()
expert_app = ExpertApp(expert=vg_expert)
app = expert_app.get_app()
expert_app.run()

