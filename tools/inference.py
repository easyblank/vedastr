import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

data_root = '../data'


from vedastr.runners import InferenceRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('gpus', type=str, help='target gpus')
    args = parser.parse_args()

    return args

class Predictor:
    def __init__(self, runner):
        self.runner = runner
    def load_image(self, image_path):
        img = cv2.imread(str(Path(data_root)/image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def prediction(self, img_path_list):
        preds = list()
        for image_path in tqdm(img_path_list):
            img = self.load_image(image_path)
            text, probs = self.runner(img)
            preds.append(text[0])
            tqdm.write(str((image_path, text[0])))
        print('Done.')
        return preds

def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')
    deploy_cfg['gpu_id'] = args.gpus.replace(" ", "")

    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    
    predictor = Predictor(runner)
    submit = pd.read_csv(str(Path(data_root)/'sample_submission.csv'))
    test_predicts = predictor.prediction(submit['img_path'].values)
    submit['text']  = test_predicts
    print(submit)
    submit.to_csv(str(Path(data_root)/'submit.csv'), index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    main()
