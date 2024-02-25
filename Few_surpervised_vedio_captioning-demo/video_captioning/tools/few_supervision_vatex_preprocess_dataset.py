import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
# import h5py
import numpy as np
import torch
import torchvision.models as models
# import skimage.io
# from PIL import Image
import pickle as pkl
from tqdm import tqdm


def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    key_imgs = json.load(open(params['keywords_json'], 'r'))
    j=0
    sentid=0
    images_list=[]
    images_dict={}
    for i in tqdm(range(len(imgs))):
        img_dict = {}
        sent_list = []
        sentid_list = []
        if i < len(key_imgs) and key_imgs[j]['image_id'] == i + 1:
            key_img = key_imgs[j]
            img = imgs[i]
            img_dict['filename'] = img['filename']
            img_dict['imgid'] = img['imgid']
            if params['ground_trues_nums'] > 0:
                for nums in range(params['ground_trues_nums']): #把原来的正确语句加入伪标记中
                    sent_list.append(img['sentences'][nums])
                    sentid_list.append(sentid)
                    sentid = sentid + 1
                keys = list(key_img.keys())[1:params['ground_trues_nums']+1]
            elif params['ground_trues_nums'] == -1:
                for nums in range(len(img['sentences'])):
                    sent_list.append(img['sentences'][nums])
                    sentid_list.append(sentid)
                    sentid = sentid + 1
                keys = list(key_img.keys())[1:]
            else:
                keys = list(key_img.keys())[1:params['sentence_nums']+1]

            for key in keys:
                sentences = key_img[key]
                for sentence in sentences[:params['pseudo_nums']]:
                    sent_dict = {}
                    sent_dict['tokens'] = sentence['sentence'].split()
                    sent_dict['raw'] = sentence['sentence']
                    sent_dict['imgid'] = img['imgid']
                    sent_dict['sentid'] = sentid
                    sentid_list.append(sentid)
                    sentid = sentid + 1
                    sent_list.append(sent_dict)
            img_dict['sentences'] = sent_list
            img_dict['split'] = img['split']
            img_dict['sentids'] = sentid_list
            images_list.append(img_dict)
            j = j + 1
        else:
            img = imgs[i]
            img_dict['filename'] = img['filename']
            img_dict['imgid'] = img['imgid']
            for img_sent in img['sentences']:
                sent_dict = {}
                sent_dict['tokens'] = img_sent['tokens']
                sent_dict['raw'] = img_sent['raw']
                sent_dict['imgid'] = img['imgid']
                sent_dict['sentid'] = sentid
                sentid_list.append(sentid)
                sentid = sentid+1
                sent_list.append(sent_dict)
            img_dict['sentences'] = sent_list
            img_dict['split'] = img['split']
            img_dict['sentids'] = sentid_list
            images_list.append(img_dict)
    images_dict['images'] = images_list
    json.dump(images_dict, open(params['output_dict'], "w") )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/full_supervision/dataset.json')

    parser.add_argument('--keywords_json', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/pseudo_label_generation/outputs/XLNetLMGenerate/VATEX/pseudo_labels.json')

    parser.add_argument('--output_dict', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/few_supervision/dataset.json', help='output dictectory')
    
    parser.add_argument('--ground_trues_nums', default=3, type=int, help='把原来的正确语句加入伪标记中')

    parser.add_argument('--sentence_nums', default=3, type=int, help='选择前几个描述语句')

    parser.add_argument('--pseudo_nums', default=2, type=int, help='伪标记数量')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)