import sys
sys.path.append('../')
import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from infer import InferenceWrapper2
import cv2
import numpy 
import time
import os

def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))


args_dict = {
    'project_dir': '../',
    'init_experiment_dir': '../runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}


module = InferenceWrapper2(args_dict)

source_data_dict = {
        'source_imgs': np.asarray(Image.open('images/target.jpg')) # H x W x 3
        }

module.initialization(source_data_dict)

src_dict = module.source_data_dict
src_dict, pred_enh_tex_hf_rgbs = module.runner.predict_source_dict(src_dict)

idt_embedding = src_dict['source_idt_embeds']
pred_tex_hf_rgbs = src_dict["pred_tex_hf_rgbs"][:, 0]

vs = cv2.VideoCapture(0)
vs.set(3,1280) #width
vs.set(4,720) #height

while(True):

    (grabbed, camera_frame) = vs.read()

    tgt_image = Image.fromarray(cv2.cvtColor(camera_frame,cv2.COLOR_BGR2RGB))  
    tgt_image = np.asarray(tgt_image)[None]

    start = time.time()
    tgt_pose = module.get_pose(tgt_image, True)
    elapsed_time = time.time() - start
    print ("get_pose_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    tgt_pose_embedding = module.runner.nets["keypoints_embedder"].predict_target_embedding(tgt_pose)
    pred_target_imgs,pred_target_segs = module.runner.nets["inference_generator"].predict_lf_img(idt_embedding, tgt_pose_embedding, pred_tex_hf_rgbs, pred_enh_tex_hf_rgbs)
    elapsed_time = time.time() - start
    print ("detection_time:{0}".format(elapsed_time) + "[sec]")

    pred_img = to_image(pred_target_imgs[0, 0], pred_target_segs[0, 0])
    output_img = cv2.cvtColor(numpy.asarray(pred_img),cv2.COLOR_RGB2BGR) 

    cv2.imshow("cv", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break