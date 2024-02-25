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
import glob

def main(params):

    data_dict = {}
    img_dict = {}
    sentences_list = []

    imgs = json.load(open(params['input_train_json'], 'r'))
    for img in imgs:
        img_dict['split'] = 'train'
        img_dict['captions'] = img['enCap']
        data_dict[img['videoID']] = img_dict
        img_dict = {}

    imgs = json.load(open(params['input_val_json'], 'r'))
    for img in imgs:
        img_dict['split'] = 'val'
        img_dict['captions'] = img['enCap']
        data_dict[img['videoID']] = img_dict
        img_dict = {}

    imgs = json.load(open(params['input_test_json'], 'r'))
    for img in imgs:
        img_dict['split'] = 'test'
        img_dict['captions'] = img['enCap']
        data_dict[img['videoID']] = img_dict
        img_dict = {}
    
    print(len(data_dict))

    images_list=[]
    images_dict={}
    img_id = 1
    sent_id = 0
    len_sum = 0
    train = 0
    val = 0
    test = 0
    vid_ids = os.listdir("/data16t/wangtao/dataset/xmodaler-VATEX/original/Resnext/")
    for filename in tqdm(vid_ids):
        filename = filename.split('.')[0]
        img_dict = {}
        sent_list = []
        sentid_list = []
        img_dict['filename'] = filename+'.mp4'
        img_dict['imgid'] = img_id
        for sentence in data_dict[filename]['captions']:
            sent_dict = {}
            # sentence = sentence.replace(".", "")
            # sentence = sentence.replace(",", "")
            # sentence = sentence.replace('"', "")
            # sentence = sentence.replace('(', "")
            # sentence = sentence.replace(')', "")
            sent_dict['tokens'] = sentence.split()
            len_sum += len(sentence.split())
            sent_dict['raw'] = sentence

            sent_dict['imgid'] = img_id
            sent_dict['sentid'] = sent_id
            sentid_list.append(sent_id)
            sent_id += 1
            sent_list.append(sent_dict)
        
        img_dict['sentences'] = sent_list
        if (data_dict[filename]['split'] == 'train'):
            train += 1
            img_dict['split'] = 'train'
        elif(data_dict[filename]['split'] == 'val'):
            val += 1
            img_dict['split'] = 'val'
        else:
            test += 1
            img_dict['split'] = 'test'
        img_id += 1
        img_dict['sentids'] = sentid_list
        images_list.append(img_dict)
    mean_sent_len = len_sum / sent_id
    print("平均句子长度:{}".format(mean_sent_len))
    images_dict['images'] = images_list
    json.dump(images_dict, open(params['output_dir'], "w") )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/full_supervision/vatex_training_v1.0.json')
    parser.add_argument('--input_val_json', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/full_supervision/vatex_validation_v1.0.json')
    parser.add_argument('--input_test_json', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/full_supervision/vatex_public_test_english_v1.1.json')
    parser.add_argument('--output_dir', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/full_supervision/dataset.json', help='output dictectory')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)