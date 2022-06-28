# uvicorn vg.vg_expert:app --reload --host 0.0.0.0 --port 8889
import os
import sys
sys.path.append("/notebooks/nebula3_experts_vg/nebula3_experts")
sys.path.append("/notebooks/nebula3_experts_vg/nebula3_experts/nebula3_pipeline")
sys.path.append("/notebooks/nebula3_experts_vg/nebula3_experts/nebula3_pipeline/nebula3_database")
sys.path.append("/notebooks/nebula3_experts_vg/vg/run_scripts/refcoco")
sys.path.append("/notebooks/nebula3_experts_vg/vg")
sys.path.append("/notebooks/nebula3_experts_vg")

import json
from fastapi import FastAPI
import urllib
from PIL import Image

sys.path.append("/notebooks/nebula3_experts")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline/nebula3_database")

from nebula3_experts.experts.service.base_expert import BaseExpert
from nebula3_experts.experts.app import ExpertApp
from nebula3_experts.experts.common.models import ExpertParam, TokenRecord
from nebula3_database.config import NEBULA_CONF
from nebula3_database.movie_db import MOVIE_DB
from nebula3_videoprocessing.videoprocessing.vlm_interface import VlmInterface
from nebula3_experts.nebula3_pipeline.nebula3_database.config import NEBULA_CONF
from .visual_grounding_inference import OfaMultiModalVisualGrounding
import cv2


def plot_vg_over_image(result, frame_, caption, lprob):
    import numpy as np
    print("SoftMax score of the decoder", lprob, lprob.sum())
    print('Caption: {}'.format(caption))
    window_name = 'Image'
    image = np.array(frame_)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    normalizedImg = np.zeros_like(img)
    normalizedImg = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = normalizedImg.astype('uint8')

    image = cv2.rectangle(
        img,
        (int(result[0]["box"][0]), int(result[0]["box"][1])),
        (int(result[0]["box"][2]), int(result[0]["box"][3])),
        (0, 255, 0),
        3
    )
    # print(caption)
    movie_id = '111'
    mdf = '-1'
    path = './'
    file = 'pokemon'
    cv2.imshow(window_name, img)

    cv2.setWindowTitle(window_name, str(movie_id) + '_mdf_' + str(mdf) + '_' + caption)
    cv2.putText(image, caption + '_ prob_' + str(lprob.sum().__format__('.3f')) + str(lprob),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA, org=(10, 40))
    fname = str(file) + '_' + str(caption) + '.png'
    cv2.imwrite(os.path.join(path, fname),
                image)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))


class VisualGroundingVlmImplementation(VlmInterface):
    def __init__(self):
        self.vg_engine = OfaMultiModalVisualGrounding()
    
    def load_image(self):
        pass

    def compute_similarity(self, image : Image, text : list[str]):
        time_measure = False
        if time_measure:
            import time
            since = time.time()

        bb, _, lprob = self.vg_engine.find_visual_grounding(image, text)

        if time_measure:
            time_elapsed = time.time() - since
            print('OFa VG time {:.3f}s'.format(time_elapsed))

        lprob = lprob.sum()
        debug = False
        if debug:
            plot_vg_over_image(bb, image, caption=text, lprob=lprob)

        return bb, lprob.cpu().numpy()


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
        self.nre = self.movie_db
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
                print(fps)
            except:
                print(f'An exception occurred while fetching {url_link}')
                result = False
                fps = -1
                video = {}
        # return result

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
        time_measure = False
        vg_param, error = self.parse_vg_params(expert_params)
        if error:
            movie_id = expert_params['movie_id']if  expert_params['movie_id']else ''
            return { 'error': f'error {error} for movie: {movie_id}'}

        if time_measure:
            import time
            since = time.time()

        movie_info, fps, fn, video = self._download_and_get_minfo(expert_params.movie_id, to_print=True)

        if time_measure:
            time_elapsed = time.time() - since
            print('Loading movie {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        bb = list()
        for elem_inx, elem in enumerate(movie_info['scene_elements']):
            mdfs = movie_info['mdfs'][elem_inx]
            for mdf in mdfs:
                if mdf == vg_param['mdf']:
                    feature_mdfs = []
                    video.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = video.read()  # Read the frame
                    frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)  # HK @@

                    if time_measure:
                        since = time.time()

                    results, scores, lprob = self.vg_engine.find_visual_grounding(Image.fromarray(frame_), vg_param['caption'])

                    if time_measure:
                        time_elapsed = time.time() - since
                        print('OFa VG time {:.3f}s'.format(time_elapsed))

                    lprob = lprob.sum()
                    debug = False
                    if debug:
                        plot_vg_over_image(results, frame_, caption=vg_param['caption'], lprob=lprob)

                    result = self._transform_vg_result(vg_result={'bbox': results[0]['box'], 'lprob': float(lprob)},
                                                       expert_params=vg_param)
                    return [result], error

        if bb == []:
            error = {'error': f"movie frames not found: {vg_param['movie_id']}"}
            return [-1], error




    def _transform_vg_result(self, vg_result, expert_params: ExpertParam):
        # print(vg_result)
        tr = TokenRecord(expert_params['movie_id'],
                         0, 0, self.get_name(),
                         vg_result['bbox'],
                         -1,
                         {'lprob': vg_result['lprob'], 'mdf': expert_params['mdf']})
        return tr

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
        return movie, mdf

    def get_vg(self):

        return

    def transform_vg_result(self, vg_result, vg_params):
        # print(vg_result)
        result = list()
        for oid, data in vg_result.items():
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

