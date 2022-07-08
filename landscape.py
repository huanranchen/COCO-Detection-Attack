import argparse
import os
import torch
from tqdm import tqdm
import random
from tools.parser import ConfigParser
from tools.data_loader import read_img_np_batch
from losses import temp_attack_loss
from evaluate import UniversalPatchEvaluator


class GetLoss():
    def __init__(self, args, total_step=1, use_which_image=None):
        self.batch_size = 16 if use_which_image is None else 1
        # read config file
        self.cfg = ConfigParser(args.config_file)
        self.device = torch.device('cuda')
        self.evaluator = UniversalPatchEvaluator(self.cfg, args, self.device, if_read_patch=False)
        self.img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]
        self.img_names.sort()
        if use_which_image is not None:
            self.img_names = [self.img_names[use_which_image]]
        self.total_step = total_step
        self.use_which_image = use_which_image

    def __call__(self, patch):
        self.evaluator.read_patch_from_memory(patch)
        step = 0
        total_loss = 0
        random.shuffle(self.img_names)
        for index in tqdm(range(0, len(self.img_names), self.batch_size)):
            names = self.img_names[index:index + self.batch_size]
            img_name = names[0].split('/')[-1]
            img_numpy_batch = read_img_np_batch(names, self.cfg.DETECTOR.INPUT_SIZE)

            all_preds = None
            for detector in self.evaluator.detectors:
                # print(detector.name)
                all_preds = None
                img_tensor_batch = detector.init_img_batch(img_numpy_batch)

                # 在干净样本上得到所有的目标检测框，定位patch覆盖的位置
                preds, detections_with_grad = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)
                all_preds = self.evaluator.merge_batch_pred(all_preds, preds)

                # 可以拿到loss的时机1：在干净样本上的loss
                loss1 = temp_attack_loss(detections_with_grad)
                # print(loss1, detections_with_grad.shape)

                has_target = self.evaluator.get_patch_pos_batch(all_preds)
                if not has_target:
                    continue

                # 添加patch，生成对抗样本
                adv_tensor_batch, patch_tmp = self.evaluator.add_universal_patch(img_numpy_batch, detector)
                # 对对抗样本进行目标检测
                preds, detections_with_grad = detector.detect_img_batch_get_bbox_conf(adv_tensor_batch)
                # self.evaluator.get_patch_pos_batch(preds)
                # self.evaluator.imshow_save(img_numpy_batch,save_path = './Draws/out',save_name='WTF.jpg')
                loss2 = temp_attack_loss(detections_with_grad)
                # print(loss2, detections_with_grad.shape)
                total_loss += loss2.item()
                step += 1
                if step > self.total_step:
                    result = total_loss / step
                    if self.use_which_image is not None:
                        return 1 if result > 0.9 else 0
                    return result

        result = total_loss / step
        if self.use_which_image is not None:
            return 1 if result > 0.9 else 0
        return result


if __name__ == '__main__':
    from Draws.Landscape import train_valid_3dlandscape, multi_model_3dlandscape, multi_image_contourf, \
        train_valid_2dlandscape

    train_valid_2dlandscape()
