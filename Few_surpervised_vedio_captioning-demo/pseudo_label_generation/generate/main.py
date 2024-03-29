# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 2:35 PM
# @Author  : He Xingwei

import torch
import numpy as np
from transformers import XLNetTokenizer, XLNetLMHeadModel, XLNetForTokenClassification
import time
import os
import sys
import argparse
sys.path.append('../')
from utils.log import Logger
from lm_generate import LMGenerate
import codecs
from vocabulary.vocab import Vocab
from language_models.lstm_lm import LSTMLanguageModel
import json
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def random_choose_action(prior_probs=[1,1,0]):
    probs = np.array(prior_probs)
    _sum = np.sum(probs)
    probs = probs/_sum
    list_len = len(probs)
    sampled_action = np.random.choice(list_len, 1, p=probs)
    return sampled_action[0]

def random_choose_position(sentence_length):
    return np.random.randint(0, sentence_length)

def check_action_and_postion(action, subword_position, sentence_length, keyword_vector):
    # replace or delete
    if action == 0 or action==2:
        if subword_position == sentence_length or keyword_vector[subword_position] >0:
            return False
        else:
            return True
    # insert
    elif action == 1:
        if subword_position == sentence_length or subword_position == 0 or keyword_vector[subword_position] == 0:
            return True
        # inserting between subwords of a keyword is not allowed.
        if keyword_vector[subword_position]  == keyword_vector[subword_position-1]:
            return False
        else:
            return True
    else:
        raise ValueError(f'''action: {action} is not valid.''')

def classifier_choose_action_position(classifier_model, tokenizer, device, input_text, input_ids, keyword_vector, sentence_length, maxlength, minlength,
                                      mode = 'LSTM', memory=None, learned_prior = False, using_delete = 0):
    """
    copy: 0
    replace: 1
    insert: 2
    delete: 3
    """
    if mode == 0 or mode == 'LSTM': #
        labels = []
        input_ids = []
        words = input_text.split()
        assert len(words) == len(keyword_vector)
        for i, word in enumerate(words):
            ids = tokenizer.encode(word, add_special_tokens=False)
            labels += [i+1]*len(ids)
            input_ids += ids
        labels.append(2000)
    else:
        if input_ids is None:
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id] #[1, 20715, 4099, 16149, 6197, 2]

    input_ids  = torch.tensor([input_ids]) #(1, 6)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = classifier_model(input_ids=input_ids) #(1, 6, 4)
    
    # print(parameter_count_table(classifier_model))
    # flops = FlopCountAnalysis(classifier_model, inputs = (input_ids))
    # print("GFLOPs: ", flops.total()/10**9)

    logits = outputs[0][0] #(6, 4)
    # predict_label = torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits, -1) #(6, 4)
    probabilities = probabilities.cpu().numpy() #(6, 4)

    probabilities[0,:]=0 #<bos>的所有动作置为0 
    probabilities[-1,0]=0 #<eos>的copy动作置为0
    probabilities[-1,1]=0 #<eos>的replace动作置为0
    probabilities[-1,3]=0 #<eos>的delete动作置为0
    # skip the bos token and the copy label
    probabilities = probabilities[1:,1:] #(5, 3)
    if mode == 0 or mode == 'LSTM':
        #extract the output of the first subword as the output the word
        probs = np.zeros([len(keyword_vector)+1, 3])
        pre_label = -1
        index = 0
        for i, _label in enumerate(labels):
            if _label != pre_label:
                probs[index,:] = probabilities[i,:]
                index += 1
                pre_label = _label
        # print(labels, probs, probabilities)
        probabilities = probs
    pre_value = -10000
    for i, value in enumerate(keyword_vector): #[1, 2, 3, 4]
        if value > 0: 
            # set the replace as 0 for keyword
            probabilities[i, 0] = 0
            # set the delete as 0 for keyword 关键词不允许替换和删除
            probabilities[i, 2] = 0

            if value == pre_value: #如果是同一个单词拆分则将add置为0
                probabilities[i,1] = 0
            pre_value = value

    if memory is not None:
        _probabilities = probabilities.copy()
        # if has tried times for (position, action) has over the maximum tried time, set p(position, action) = 0
        for i in range(len(keyword_vector)):
            for j in range(2): # num of actions
                if memory[i,j]<=0:
                    _probabilities[i,j] = 0
        if memory[len(keyword_vector), 1]<=0:
            _probabilities[len(keyword_vector), j] = 0
        if np.sum(_probabilities[:,0])+np.sum(_probabilities[:,1])>0:
            probabilities = _probabilities

    if using_delete==1: #当使用删除时，操作为3个，否则为2个
        num_action = 3
    else:
        num_action = 2

    if learned_prior: #1表示使用分类器给出的动作的先验和0表示使用一致先验
        p1 = np.sum(probabilities[:, 0])
        p2 = np.sum(probabilities[:, 1])
        if using_delete == 1: #是否使用删除
            p3 =  np.sum(probabilities[:, 2])
            prior = [p1,p2,p3] #[0.0, 0.9499064, 0.0]
        else:
            prior = [p1,p2]
        prior = np.array(prior)/np.sum(prior) #[0., 1., 0.]
    else:
        if using_delete == 1:
            prior = [0.3333, 0.3333,0.3334]
        else:
            prior = [0.5, 0.5]
    if sentence_length < maxlength:
        action = np.random.choice(num_action, 1, p = prior)[0] #1， p：与数组a相对应，表示取数组a中每个元素的概率。
    else:
        prior[1] = 0
        prior = np.array(prior)/np.sum(prior)
        action = np.random.choice(num_action, 1, p = prior)[0]

    if action==0 or action==2:
        # replace， insert， delete
        probs = probabilities[:,action]
        if np.sum(probs)==0:
            action=1
            probs = probabilities[:, action]
    else:
        probs = probabilities[:, action] #(5), 选择insert操作

    _sum = np.sum(probs)
    probs = probs / _sum #归一化
    list_len = len(probs) #5
    sampled_id = np.random.choice(list_len, 1, p = probs)[0]
    position = sampled_id 

    return action, position #1, 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--max_keywords', type=int, default=1)
    parser.add_argument('-sn', '--sample_number', type=int, default=100)
    parser.add_argument('--tried_time', type=int, default=0)

    parser.add_argument('--min_length', type=int, default=3, help='the minimum length of the generated sentence.')
    parser.add_argument('--max_length', type=int, default=20, help='the maximum length of the generated sentence.')
    parser.add_argument('--candidate_num', type=int, default=50,
                        help='the number of the candidate sentences for the proposal distribution.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='the number of the candidate sentences for the proposal distribution.')
    parser.add_argument('--repetition_penalty', type=float, default=1.6,
                        help='to alleviate the generation of duplicate tokens.')

    parser.add_argument('--output_file', type=str, default='../outputs/')
    parser.add_argument('--input_file', type=str, default='../inputs/')
    parser.add_argument('--started_sentence_id', type=int, default=1,
                        help='Generate sentences from the specific line of the input file.')
    parser.add_argument('--model_name', type=str,default='XLNetLMGenerate', choices=['LSTMLMGenerate',"XLNetLMGenerate"])
    parser.add_argument('-gm','--generate_candidate_method', type=int,default=2,
                        choices=[0,1,2,3],
                        help='Specify the method to generate candidates: 0: forward lm; 1: backward lm; '
                             '2: both forward and backward lm; 3: masked lm.')
    parser.add_argument('--delete', type=int, default=1)
    parser.add_argument('--random', type=int, default=0,
                        help='randomly choose a action and position if random equals to 1; otherwise choose them with a classifier.')
    parser.add_argument('--learned_prior', type=int, default=1,
                        help='1 denotes using prior given by the classifier for actions '
                             'and 0 denotes using uniform prior.')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--show_log', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='VATEX')
    parser.add_argument('--split', type=int, default=12500, 
                        help='You can divide the dataset into different splits and use multiple GPUs for processing')
    parser.add_argument('--sentence_nums', type=int, default=1)
    # parser.add_argument('--dataset', type=str, default='oxford')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.min_length += args.max_keywords


    log_path = '../logs/{}/{}'.format(args.model_name, args.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    suffix = f'maxlen{args.max_length}_top{args.top_k}_delete{args.delete}_sn{args.sample_number}_gm{args.generate_candidate_method}_rand{args.random}'
    suffix_json = f'maxlen{args.max_length}_top{args.top_k}_delete{args.delete}_sn{args.sample_number}_gm{args.generate_candidate_method}_rand{args.random}'
    if args.random==0:
        suffix +=f'_learned_prior{args.learned_prior}'
    if args.tried_time > 0:
        suffix += f'_tried_time{args.tried_time}'

    suffix += f'_repetition_penalty_{args.repetition_penalty}'
    suffix += f'_keywords_{args.max_keywords}_split_{args.split}_sentences_{args.sentence_nums}.txt'
    suffix_json += f'_keywords_{args.max_keywords}_split_{args.split}_sentences_{args.sentence_nums}.json'

    args.log_file = os.path.join(log_path,suffix)
    if args.started_sentence_id==1 and os.path.exists(args.log_file):
        os.remove(args.log_file)
    print('The log file is ', args.log_file)
    logger = Logger(args.log_file)

    output_path = '../outputs/{}/{}'.format(args.model_name, args.dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    args.output_file = os.path.join(output_path, suffix)
    output_json = os.path.join(output_path, suffix_json)
    if args.started_sentence_id==1 and os.path.exists(args.output_file):
        os.remove(args.output_file)
    print('The output file is ', args.output_file)
    print('The output json is', output_json)

    args.input_file = os.path.join(args.input_file, f'''{args.dataset}/{args.max_keywords}keywords.json''')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.random==0:
        classifier_model_path = '../checkpoints/{}/xlnet_classifier/'.format(args.dataset)
        args.classifier_model_path = classifier_model_path
        classifier_model = XLNetForTokenClassification.from_pretrained(classifier_model_path,num_labels=4)
        classifier_model_tokenizer = XLNetTokenizer.from_pretrained(classifier_model_path)
        logger.logger.info('Initialize backward XLNetForTokenClassification from checkpoint {}.'.format(classifier_model_path))
        classifier_model = classifier_model.to(device)
        classifier_model.eval()
    else:
        classifier_model = None
        classifier_model_tokenizer = None


    if args.model_name == 'LSTMLMGenerate':
        forward_lm_path = '../checkpoints/{}/forward_lstm_lm/best.pt'.format(args.dataset)
        backward_lm_path = '../checkpoints/{}/backward_lstm_lm/best.pt'.format(args.dataset)
        args.forward_lm_path = forward_lm_path
        args.backward_lm_path = backward_lm_path

        forward_lm_tokenizer = Vocab(training_files=[f'../data/{args.dataset}/train.txt'])
        forward_lm = LSTMLanguageModel(vocab_size=forward_lm_tokenizer.vocab_size, dropout=0, hidden_size=256, num_layers=2)
        forward_lm.from_pretrained(forward_lm_path)

        backward_lm = LSTMLanguageModel(vocab_size=forward_lm_tokenizer.vocab_size, dropout=0, hidden_size=256, num_layers=2)
        backward_lm.from_pretrained(backward_lm_path)

        masked_lm = None
        mode = 0
    elif args.model_name =='XLNetLMGenerate':

        forward_lm_path = '../checkpoints/{}/forward_xlnet/'.format(args.dataset)
        args.forward_lm_path = forward_lm_path

        backward_lm_path = '../checkpoints/{}/backward_xlnet/'.format(args.dataset)
        args.backward_lm_path = backward_lm_path

        masked_lm_path = '../checkpoints/{}/xlnet_maskedlm/'.format(args.dataset)
        args.masked_lm_path = masked_lm_path

        forward_lm_tokenizer = XLNetTokenizer.from_pretrained(forward_lm_path)
        forward_lm = XLNetLMHeadModel.from_pretrained(forward_lm_path)
        logger.logger.info('Initialize forward XLNet LM from checkpoint {}.'.format(forward_lm_path))

        # backward_lm_tokenizer = XLNetTokenizer.from_pretrained(backward_lm_path)
        backward_lm = XLNetLMHeadModel.from_pretrained(backward_lm_path)
        logger.logger.info('Initialize backward XLNet LM from checkpoint {}.'.format(backward_lm_path))

        if args.generate_candidate_method==3: #指定候选的生成方式
            masked_lm =XLNetLMHeadModel.from_pretrained(masked_lm_path)
            logger.logger.info('Initialize masked XLNet LM from checkpoint {}.'.format(masked_lm_path))
        else:
            masked_lm = None
        mode = 1
    else:
        raise ValueError('wrong model type.')

    generate_model = LMGenerate(device, forward_lm, forward_lm_tokenizer, backward_lm=backward_lm, masked_lm=masked_lm,  
                            generate_candidate_method=args.generate_candidate_method, candidate_num=args.candidate_num,
                            top_k = args.top_k, repetition_penalty = args.repetition_penalty, classifier_model=classifier_model,
                            classifier_model_tokenizer=classifier_model_tokenizer)


    logger.logger.info(args)

    number_actions = 3
    # acceptance dict
    accept_num = {} # record the number of being accepted for each action.
    act_time = {}   # record the running time for each action.
    total_num = {}  # record the number of being called for each action.
    for i in range(number_actions):
        accept_num[i] = 0
        total_num[i] = 0
        act_time[i] = 0.0
        cls_time = 0

    started_sentence_id = max(1, args.started_sentence_id) #1
    
    imgs =  json.load(open(args.input_file, 'r'))
    img_list = []
    split = args.split
    imgs_id_list = []

    if os.path.exists(output_json):
        imgs_output = json.load(open(output_json, 'r'))
        img_list = imgs_output
        for img in imgs_output:
            imgs_id_list.append(img['image_id'])
    
    # for img in imgs[split - 4000:split]:
    for img in imgs[28000:]:
        img_dir = {}
        if img['image_id'] in imgs_id_list:
            continue
        img_dir['image_id'] = img['image_id']
        if args.sentence_nums > 0:
            sentence_ids = np.arange(0, len(img['sentences']))
            # np.random.shuffle(sentence_ids)
            sentence_ids = sentence_ids[:args.sentence_nums]
            sentences = np.array(img['sentences'], dtype=object)[sentence_ids]
        else:
            sentences = img['sentences']
        for sentence_id, sent in enumerate(sentences):
            # save all generated sentences during the sampling process.
            generated_sentences = []
            input_text = sent # "squirrel eating peanut shell"
            keywords = input_text.split() #['squirrel', 'eating', 'peanut', 'shell']

            # construct keyword_vector
            if args.model_name =='GPT2Generate':
                keyword_vector = []
                input_ids = []
                for i, keyword in enumerate(keywords):
                    subwords = forward_lm_tokenizer.tokenize(' '+keyword)
                    subids = forward_lm_tokenizer.convert_tokens_to_ids(subwords)
                    keyword_vector += [i+1]*len(subwords)
                    input_ids += subids
                sentence_length = len(keyword_vector)

            elif args.model_name == 'XLNetLMGenerate':
                keyword_vector = []
                input_ids = []
                for i, keyword in enumerate(keywords):
                    subwords = forward_lm_tokenizer.tokenize(keyword) #
                    subids = forward_lm_tokenizer.convert_tokens_to_ids(subwords) #
                    keyword_vector += [i+1]*len(subwords) #[1, 2, 3, 4] 一个单词可能会被拆分为多个单词
                    input_ids += subids #[20715, 4099, 16149, 6197]
                sentence_length = len(keyword_vector) #4

            elif args.model_name =='LSTMLMGenerate':
                keyword_vector = []
                sentence_length = len(keywords)
                keyword_vector = list(range(1,sentence_length+1))
                input_ids = forward_lm_tokenizer.encode(input_text, add_special_tokens=False)

            ind = -1
            _is_accept = 1
            memory_dict = {}
            input_text_list =[]
            for sample_index in range(1, args.sample_number+1):
                if args.random:
                    if sentence_length < args.max_length:
                        if args.delete:
                            prior_probs = [1, 1, 1]
                        else:
                            prior_probs = [1, 1, 0]
                    else:
                        if args.delete:
                            prior_probs = [1, 0, 1]
                        else:
                            prior_probs = [1, 0, 0]

                    action = random_choose_action(prior_probs=prior_probs)
                    word_position = random_choose_position(sentence_length+1)
                    while not check_action_and_postion(action, word_position, sentence_length, keyword_vector):
                        action = random_choose_action(prior_probs=prior_probs)
                        word_position = random_choose_position(sentence_length + 1)
                else:
                    if args.tried_time > 0:
                        if _is_accept == 1:
                            # update memory_dict for the new input
                            memory = np.ones([sentence_length + 1, 2]) * args.tried_time
                        else:
                            # old input
                            memory[word_position,action] -= 1
                    else:
                        memory = None

                    action, word_position  = classifier_choose_action_position\
                        (classifier_model, classifier_model_tokenizer, device, input_text, input_ids, keyword_vector, sentence_length,
                        args.max_length, args.min_length, mode, memory = memory, learned_prior = args.learned_prior, using_delete=args.delete)

                if action==0:
                    action_function = generate_model.replace
                elif action==1:
                    action_function = generate_model.insert
                elif action==2:
                    action_function = generate_model.delete
                else:
                    raise ValueError(f'''action: {action} is not valid.''')
                # call action_function
                start = time.time()
                # print(f'''action {action}/{word_position}''')
                # print(input_text, keyword_vector)
                new_input_text, new_input_ids, keyword_vector, _is_accept,  probability, perplexity, sentence_length = \
                    action_function(input_text, input_ids, keyword_vector, keywords, word_position, sentence_length)

                if new_input_text not in input_text_list:
                    input_text_list.append(new_input_text)
                    generated_sentences.append((sample_index, new_input_text, probability, perplexity, len(new_input_text.split())))

                input_text = new_input_text # 'a squirrel eating peanut shell'
                input_ids = new_input_ids #[24, 20715, 4099, 16149, 6197]

                used_time = time.time()-start
                act_time[action] += used_time
                total_num[action] += 1
                accept_num[action] += _is_accept

            log_info = f'''{img['image_id']}->{sentence_id}:\n'''
            for action in range(number_actions):
                if total_num[action]>0:
                    log_info += f'''action {action}: average running time''' + \
                                f'''{act_time[action]:.1f} / {total_num[action]}={act_time[action] / total_num[action]:.3f},''' + \
                                f'''acceptance rate {accept_num[action]} / {total_num[action]}={accept_num[action] / total_num[action]:.3f};\n'''
            logger.logger.info(log_info)
            if args.show_log:
                print("----")
                for sentence in generated_sentences:
                    print(sentence[1])
            # outputs = generated_sentences[100:]
            outputs = generated_sentences
            # choose output from samples
            for num in range(args.min_length, 0, -1):
                outputss = [x for x in outputs if x[-1] >= num]
                if outputss != []:
                    break
            sent_list = []
            with codecs.open(args.output_file, 'a',encoding='utf8') as g:
                # sort generated sentences based on probability values and select sentence with maximum probability
                output = sorted(outputss, key=lambda x: x[2])[-1]
                # sample_index, new_input_text, probability, perplexity, sentence_length
                g.write(f'''img_id: {img['image_id']} sentence_id: {sentence_id} Maximum probability: {output[0]}\t{output[1]}\t{output[2]}\t{output[3]:.3f}\t{output[4]}\n''' )
                # outputs = sorted(outputss, key=lambda x: x[2])
                # for output in outputs[:5]:
                #     sent_dir_maxpro = {}
                #     sent_dir_maxpro['sentence'] = output[1]
                #     sent_dir_maxpro['Maximum_probability'] = output[2]
                #     sent_dir_maxpro['perplexity'] = round(output[3], 3)
                #     sent_dir_maxpro['len'] = output[4]
                #     sent_list.append(sent_dir_maxpro)
                # sort generated sentences based on perplexity values
                output = sorted(outputss, key=lambda x: x[3])[0]
                g.write(f'''img_id: {img['image_id']} sentence_id: {sentence_id} Minimum perplexity: {output[0]}\t{output[1]}\t{output[2]}\t{output[3]:.3f}\t{output[4]}\n''' )
                outputs = sorted(outputss, key=lambda x: x[3])
                for output in outputs[:10]:
                    sent_dir_minper = {}
                    sent_dir_minper['sentence'] = output[1]
                    sent_dir_minper['Minimum_perplexity'] = output[2]
                    sent_dir_minper['perplexity'] = round(output[3], 3)
                    sent_dir_minper['len'] = output[4]
                    sent_list.append(sent_dir_minper)
            str_sentence_id = 'sentence_'+str(sentence_id)
            img_dir[str_sentence_id] = sent_list

        img_list.append(img_dir)
        json.dump(img_list, open(output_json, "w") )
    
