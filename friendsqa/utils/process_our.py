import json
import os
import random
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import torch
from transformers.file_utils import is_torch_available
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor
import collections
import re, string
import copy
from scipy.sparse import coo_matrix
from utils.config import *
# from config import *


# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet", "longformer"}

logger = logging.get_logger(__name__)


map_relations = {'Comment': 1, 'Contrast': 2, 'Correction': 3, 'Question-answer_pair': 4, 'QAP': 4, 
                 'Parallel': 5, 'Acknowledgement': 6, 'Elaboration': 7, 'Clarification_question': 8, 
                 'Conditional': 9, 'Continuation': 10, 'Result': 11, 'Explanation': 12, 'Q-Elab': 13, 
                 'Alternation': 14, 'Narration': 15, 'Background': 16}


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 
               'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
               'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
               'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
               'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
               'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
               'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
               'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
               'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
               'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
               'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 
               'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 
               'wouldn', 'how', 'when', 'what', 'why', 'which', 'where', 'something', 'anything', 'nothing', 'one', 'people',
               'thing', "'s", "'m", 'm', 's', "n't"]



srl_relations = {'B-V': 0, 'ARG0': 1, 'ARG2': 2, 'ARG3': 3, 'ARG4': 3, 'ARG5': 3, 
                 'ARGM-NEG': 4, 'ARGM-TMP': 5, 'ARGM-MOD': 6, 'ARGM-ADV': 7, 
                 'ARGM-LOC': 8, 'ARGM-MNR': 9, 'ARGM-CAU': 10, 'ARGM-DIR': 11}


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def is_number_str(s):
    for si in s:
        if si not in '0123456789':
            return False
    return True

def recover_name(s):
    if '=*=' not in s:
        return s
    sl = s.split('=*=')
    if len(sl) == 2:
        return sl[0]
    j = 0
    for i in range(len(sl)):
        if not is_number_str(sl[i]):
            continue
        else:
            j = i
            break
    if (len(sl) - j) % 2 == 1:
        j += 1
    return '=*='.join(sl[:j])

def edge_has_punct(edge, punct_nodes):
    for (sname, ename) in edge:
        if ename in punct_nodes or sname in punct_nodes:
            return True
    return False


def find_coref_target(target, inner_char_to_word_offset):
    poss = None
    uidx = target.split('-u')[-1]
    yy = ''.join(target.split('-u')[:-1])
    new_name = recover_name(target); char_position = yy.split('=*=')[-2:]
    poss = [inner_char_to_word_offset[int(uidx)][int(char_position[0])], inner_char_to_word_offset[int(uidx)][int(char_position[1])-1]]
    # print('000', target)
    return new_name, poss, uidx


def place_coref_target(name, new_name, new_inner_utterance_spk_relations):
    if name in new_inner_utterance_spk_relations and new_name not in new_inner_utterance_spk_relations[name]:
        new_inner_utterance_spk_relations[name].append(new_name)
    if name not in new_inner_utterance_spk_relations:
        for nme in new_inner_utterance_spk_relations:
            if name in nme or nme in name and new_name not in new_inner_utterance_spk_relations[nme]:
                new_inner_utterance_spk_relations[nme].append(new_name)
                break
        else:
            new_inner_utterance_spk_relations[name] = [new_name]


def remove_stop_words(s):
    sl = s.split()
    stops_words = ['the', "'s", "'m", 'm', 's']
    l = 0; r = len(sl)
    for i, w in enumerate(sl):
        if w in stops_words:
            l = i+1
        else:
            break
    for i in range(len(sl)-1, -1, -1):
        if sl[i] in stops_words:
            r = i
        else:
            break
    return ' '.join(sl[l:r])
    

def to_list(tensor):
    return tensor.detach().cpu().tolist()


class FriendsQAExample(object):
    def __init__(self, qid, content_list, dialogue_graph, ques_to_dialogue_subgraph, doc_tokens, ques_tokens, 
                 start_position, end_position, key_inner_nodes=None, answer_text = None, answers = None, answer_utterance_id=-1):
        self.ques_tokens = ques_tokens
        self.qas_id = qid
        self.content_list = content_list
        self.answer = answer_text
        self.answers = answers
        self.start_position = start_position
        self.end_position = end_position
        self.key_inner_nodes = key_inner_nodes
        self.dialogue_graph = dialogue_graph
        self.ques_to_dialogue_subgraph = ques_to_dialogue_subgraph
        self.doc_tokens = doc_tokens
        self.answer_utterance_id = answer_utterance_id


class FriendsQAFeature(object):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible=None,
        qas_id= None,
        query_end = None,
        query_mapping=None,
        graph_info_dict=None,
        rel_info_dict=None,
        graph_info = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id
        self.query_end = query_end
        self.query_mapping=query_mapping
        self.graph_info_dict = graph_info_dict
        self.rel_info_dict=rel_info_dict
        self.graph_info=graph_info
        


class FriendsQAResult:
    def __init__(self, unique_id, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits    


        
class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        data_info = {}
        data_info['feature_indices'] = torch.tensor(self.features[index].feature_index, dtype=torch.long)
        data_info['input_ids'] = torch.tensor(self.features[index].input_ids, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(self.features[index].token_type_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[index].attention_mask, dtype=torch.long)
        data_info['p_mask'] = torch.tensor(self.features[index].p_mask, dtype=torch.long)
        data_info['start_positions'] = torch.tensor(self.features[index].start_position, dtype=torch.long) if\
                                        self.features[index].start_position is not None else None
        data_info['end_positions'] = torch.tensor(self.features[index].end_position, dtype=torch.long) if\
                                        self.features[index].end_position is not None else None
        data_info['query_mapping'] = torch.tensor(self.features[index].query_mapping, dtype=torch.long) 
        data_info['query_end'] = torch.tensor(self.features[index].query_end-1, dtype=torch.long)
        
        graph_info_dict = self.features[index].graph_info_dict
        
        finegain_nodes_mapping = torch.zeros([args.MAX_UTTERANCE_INNER_NODE_NUM, data_info['input_ids'].size(-1)], dtype=torch.long)                                           
        finegain_nodes_mask = torch.zeros([args.MAX_UTTERANCE_INNER_NODE_NUM], dtype=torch.long) 
        finegain_nodes_type = torch.zeros([args.MAX_UTTERANCE_INNER_NODE_NUM], dtype=torch.long) 
        finegain_nodes_type_mask = torch.zeros([args.MAX_UTTERANCE_INNER_NODE_NUM], dtype=torch.long) 
        finegain_nodes_len = torch.ones([args.MAX_UTTERANCE_INNER_NODE_NUM], dtype=torch.long) 
        finegain_nodes_uid = torch.zeros([args.MAX_UTTERANCE_INNER_NODE_NUM], dtype=torch.long) 
        finegain_nodes_uid_mask = torch.zeros([args.MAX_UTTERANCE_INNER_NODE_NUM], dtype=torch.long) 
        for nodeid, idxs in graph_info_dict['finegain_nodes_to_token_position'].items():
            if nodeid >= args.MAX_UTTERANCE_INNER_NODE_NUM:
                continue
            else:
                nodelen = 0
                for i in range(0, len(idxs), 2):
                    for idx in range(idxs[i], idxs[i+1]+1):
                        finegain_nodes_mapping[nodeid][idx] = 1
                    nodelen += idxs[i+1] - idxs[i] + 1
                finegain_nodes_len[nodeid] = nodelen
                finegain_nodes_mask[nodeid] = 1
            uid = graph_info_dict['finegain_nodes_to_utter_id'][nodeid]
            if uid != -1:
                finegain_nodes_uid[nodeid] = uid
                finegain_nodes_uid_mask[nodeid] = 1
            
            node_type = graph_info_dict['finegain_nodes_to_type'][nodeid]
            if node_type in srl_relations:
                finegain_nodes_type[nodeid] = srl_relations[node_type]
                finegain_nodes_type_mask[nodeid] = 1
        
        spk_nodes_mapping = torch.zeros([args.MAX_SPEAKER_NUM, data_info['input_ids'].size(-1)], dtype=torch.long)                                           
        spk_nodes_mask = torch.zeros([args.MAX_SPEAKER_NUM], dtype=torch.long) 
        spk_nodes_len = torch.ones([args.MAX_SPEAKER_NUM], dtype=torch.long) 
        for nodeid, idxs in graph_info_dict['spk_nodes_to_token_position'].items():
            if nodeid >= args.MAX_SPEAKER_NUM:
                continue
            else:
                nodelen = 0
                for i in range(0, len(idxs), 2):
                    for idx in range(idxs[i], idxs[i+1]+1):
                        spk_nodes_mapping[nodeid][idx] = 1
                    nodelen += idxs[i+1] - idxs[i] + 1
                spk_nodes_len[nodeid] = nodelen
                spk_nodes_mask[nodeid] = 1

        all_utter_first_spk = torch.zeros([args.MAX_UTTERANCE_NUM], dtype=torch.long) * -1    
        all_utter_first_spk_mask = torch.zeros([args.MAX_UTTERANCE_NUM], dtype=torch.long)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        utter_nodes_mask = torch.zeros([args.MAX_UTTERANCE_NUM], dtype=torch.long)
        utter_gather_ids = torch.zeros([args.MAX_UTTERANCE_NUM], dtype=torch.long)
        utter_nodes_mapping = torch.zeros([args.MAX_UTTERANCE_NUM, data_info['input_ids'].size(-1)], dtype=torch.long)    
        for nodeid, idxs in graph_info_dict['utter_nodes_to_token_position'].items():
            if nodeid >= args.MAX_UTTERANCE_NUM:
                continue
            utter_gather_ids[nodeid] = idxs[-1]
            utter_nodes_mask[nodeid] = 1
            for idx in range(idxs[0], idxs[-1]+1):
                utter_nodes_mapping[nodeid][idx] = 1
        for nodeid, idx in graph_info_dict['all_utter_first_spk'].items():
            all_utter_first_spk[nodeid] = idx
            all_utter_first_spk_mask[nodeid] = 1
        
        data_info['graph_nodes_dict'] = {
            'finegain_nodes_mapping': finegain_nodes_mapping, 
            'finegain_nodes_mask':finegain_nodes_mask, 
            'finegain_nodes_len':finegain_nodes_len, 
            'spk_nodes_mapping': spk_nodes_mapping, 
            'spk_nodes_mask':spk_nodes_mask, 
            'spk_nodes_len':spk_nodes_len, 
            'utter_gather_ids':utter_gather_ids, 
            'utter_nodes_mask':utter_nodes_mask, 
            'utter_nodes_mapping':utter_nodes_mapping, 
            'answer_utterance_id':graph_info_dict['answer_utterance_id'], 
            'finegain_nodes_utter_id':finegain_nodes_uid, 
            'finegain_nodes_utter_id_mask':finegain_nodes_uid_mask, 
            'utter_first_spk_ids':all_utter_first_spk, 
            'utter_first_spk_mask':all_utter_first_spk_mask,
            'finegain_nodes_type':finegain_nodes_type, 
            'finegain_nodes_type_mask':finegain_nodes_type_mask,
            'inner_mask':torch.tensor(graph_info_dict['inner_mask'], dtype=torch.long),
            'srl_type':torch.tensor(graph_info_dict['srl_type'], dtype=torch.long),
            'qas_utter_self_mask':torch.tensor(graph_info_dict['qas_utter_self_mask'], dtype=torch.long),
            'key_inner_nodes':torch.tensor(graph_info_dict['key_inner_nodes'], dtype=torch.long),
        }
        
        graph_info = self.features[index].graph_info[0]
        g = {}
        for k, v in graph_info[0].items():
            max_nodes_num1 = graph_info[1][k[0]]
            max_nodes_num2 = graph_info[1][k[-1]]
            if v[0]:
                val = [1]*len(v[0])
                # print(v)
                g[k] = coo_matrix((val, (v[0], v[1])), shape=(max_nodes_num1, max_nodes_num2))
            else:
                g[k] = coo_matrix(([0], ([0], [0])), shape=(max_nodes_num1, max_nodes_num2))
        
        data_info['graph'] = g
        
        return data_info

    def __len__(self):
        return len(self.features)


               
class FriendsQAProcessor(DataProcessor):
    
    def __init__(self, tokenizer, threads):
        self.threads = threads
        self.sep_token = tokenizer.sep_token
    
    def get_train_examples(self, data_dir, filename=None):
        if data_dir is None:
            data_dir = ""

        with open(
            os.path.join(data_dir, filename+'.json'), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        
        fsrl = filename + '-srl.json'
        with open(
            os.path.join(data_dir, fsrl), "r", encoding="utf-8"
        ) as reader:
            srl_data = json.load(reader)["data"]

        fc = filename + 'coref_new2.json'   # have third_person
        with open(
            os.path.join(data_dir, fc), "r", encoding="utf-8"
        ) as reader:
            coref_data = json.load(reader)["data"]
        
        fpos = filename + '_pos.json'
        with open(
            os.path.join(data_dir, fpos), "r", encoding="utf-8"
        ) as reader:
            pos_data = json.load(reader)["data"]
            
        for i, data in enumerate(input_data):
            data['srl'] = srl_data[i]
            data['coref'] = coref_data[i]
            data['pos'] = pos_data[i]
            
        fq = filename + '-qas.json'
        with open(
            os.path.join(data_dir, fq), "r", encoding="utf-8"
        ) as reader:
            q_data = json.load(reader)["data"]

        for data in tqdm(input_data):
            for qa in data["paragraphs"][0]["qas"]:
                qid = qa["id"]
                qa['question_info'] = q_data[qid]
                    
        threads = min(self.threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self._create_examples, is_training=True)
            examples = list(tqdm(
                p.imap(annotate_, input_data),
                total=len(input_data),
                desc="collect friendsqa examples to",
            ))
        examples = [item for sublist in examples for item in sublist]
        
        return examples

    def get_dev_examples(self, data_dir, filename=None, threads=1):
        if data_dir is None:
            data_dir = ""

        with open(
            os.path.join(data_dir, filename+'.json'), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        
        fsrl = filename + '-srl.json'
        with open(
            os.path.join(data_dir, fsrl), "r", encoding="utf-8"
        ) as reader:
            srl_data = json.load(reader)["data"]
        
        fc = filename + 'coref_new2.json'
        with open(
            os.path.join(data_dir, fc), "r", encoding="utf-8"
        ) as reader:
            coref_data = json.load(reader)["data"]
        
        fpos = filename + '-pos.json'
        with open(
            os.path.join(data_dir, fpos), "r", encoding="utf-8"
        ) as reader:
            pos_data = json.load(reader)["data"]
            
        for i, data in enumerate(input_data):
            data['srl'] = srl_data[i]
            data['coref'] = coref_data[i]
            data['pos'] = pos_data[i]
            
        fq = filename + '-qas.json'
        with open(
            os.path.join(data_dir, fq), "r", encoding="utf-8"
        ) as reader:
            q_data = json.load(reader)["data"]
        
        for data in tqdm(input_data):
            for qa in data["paragraphs"][0]["qas"]:
                qid = qa["id"]
                qa['question_info'] = q_data[qid]
                    
        threads = min(self.threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self._create_examples, is_training=False)
            examples = list(tqdm(
                p.imap(annotate_, input_data),
                total=len(input_data),
                desc="collect friendsqa examples to",
            ))

        examples = [item for sublist in examples for item in sublist]
        return examples

    def _create_examples(self, data, is_training):
        examples = []; srl_data = data['srl']; pos_data = data['pos']
        content_list, is_utterance_one_node = [], []
        same_speaker_utter, speaker_nodes_to_char_offests = {}, {}

        for ui, utterance in enumerate(data["paragraphs"][0]["utterances:"]):
            content_list.append(srl_data[str(ui)]['utterance'])
            if not srl_data[str(ui)]["semantic_role_labeling"]:
                is_utterance_one_node.append(True)
            else:
                is_utterance_one_node.append(False)
                
            speaker = ' '.join(utterance["speakers"][0].split(" ")).lower()
            if speaker in same_speaker_utter:
                same_speaker_utter[speaker].append(ui)
            else:
                same_speaker_utter[speaker] = [ui]
            speaker_nodes_to_char_offests[speaker+'=*='+str(ui)] = [0, len(speaker)]
        
        doc_tokens, char_to_word_offset, utterances_index, doc_tokens_posnoun = [], [], [], []; context_text = ''  # record the length of each utterances including [SEP]
        utterances_inner_char_to_word_offset, inner_nodes_to_token_position, inner_edges = {}, {}, {}
        inner_nodes_to_type, utter_nodes, utter2inner_edges = {}, {}, []  # srl nodes and edges
        
        for uidx, utterance_text in enumerate(content_list):
            prev_is_whitespace = True
            single_utterance_char_to_word_offset = []
            for ui, c in enumerate(utterance_text):
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                single_utterance_char_to_word_offset.append(len(doc_tokens) - 1)
                char_to_word_offset.append(len(doc_tokens) - 1)
            utterances_inner_char_to_word_offset[uidx] = single_utterance_char_to_word_offset
            context_text += utterance_text + self.sep_token + ' '
            doc_tokens.append(self.sep_token); 
            char_to_word_offset.extend([len(doc_tokens) - 1 for _ in range(len(self.sep_token + ' '))])
            utterances_index.append(len(doc_tokens)-1)
            
            utter_nodes[uidx] = (single_utterance_char_to_word_offset[0], single_utterance_char_to_word_offset[-1] + 1)

        # similar speaker nodes to together
        all_utter_first_speaker = []
        utter_first_spk_map, first_spk_to_pos = {}, {}
        for u, (sname, position) in enumerate(speaker_nodes_to_char_offests.items()):
            ui = int(sname.split('=*=')[-1]); rename = sname.split('=*=')[0]
            all_utter_first_speaker.append(rename)
            posi = [utterances_inner_char_to_word_offset[ui][position[0]], utterances_inner_char_to_word_offset[ui][position[1]-1]]
            if rename not in first_spk_to_pos:
                first_spk_to_pos[rename] = posi
            else:
                first_spk_to_pos[rename].extend(posi)
            utter_first_spk_map[ui] = rename
            utter2inner_edges.append((ui, rename, 'utter-spk'))
        
        for rename, pos in first_spk_to_pos.items():
            inner_nodes_to_token_position[rename] = pos
            inner_nodes_to_type[rename] = 'spk'

        for uidx, utterance_text in enumerate(content_list):
            single_utterance_char_to_word_offset = utterances_inner_char_to_word_offset[uidx]
            
            utter_pos_tag_info = pos_data[uidx]
            for i, pos_tag_info in enumerate(utter_pos_tag_info['pos']):
                spacy_char_position = utter_pos_tag_info['spacy_token_to_char_position'][i]
                word_position = (single_utterance_char_to_word_offset[spacy_char_position[0]], single_utterance_char_to_word_offset[spacy_char_position[1]-1])
                if pos_tag_info[1] in ("NOUN", "PROPN"):
                    doc_tokens_posnoun.append(word_position)

            words_to_char_pos = srl_data[str(uidx)]["words_to_char_pos"]
            if is_utterance_one_node[uidx]:
                continue
            
            # process srl graph
            single_inner_edges = {}; utter_nodes_have_srl = [0]*len(doc_tokens); verb_nodes = []
            for srl_dict in srl_data[str(uidx)]["semantic_role_labeling"]:
                sub_root_info = srl_dict["verb"]
                if len(sub_root_info) < 3:
                    continue
                if sub_root_info[1] in set(string.punctuation):
                    continue
                if sub_root_info[1]=="'s":
                    continue
                sub_root_char_pos = words_to_char_pos[str(sub_root_info[0])]
                if utterance_text[sub_root_char_pos[0]:sub_root_char_pos[1]] != sub_root_info[1]:
                    continue
                
                pos1 = [
                        single_utterance_char_to_word_offset[sub_root_char_pos[0]],
                        single_utterance_char_to_word_offset[sub_root_char_pos[1]-1]
                        ]
                have_signed, have_signed_len = True, 0
                for o in range(pos1[0], pos1[1]+1):
                    if utter_nodes_have_srl[o] != 1:
                        utter_nodes_have_srl[o] = 1
                        have_signed = False
                    else:
                        have_signed_len += 1
                        
                new_name1 = sub_root_info[1] + '=*=' + str(pos1[0]) + '=*=' + str(pos1[1]) + '-' + str(uidx)
                
                single_srl, have_signed2, have_signed2_len, arg_nodes = False, True, 0, []
                single_single_inner_edges, single_inner_nodes_to_token_position, single_inner_nodes_to_type, single_utter2inner_edges = {}, {}, {}, []
                for node_info in srl_dict["other"]:
                    if len(node_info) < 3:
                        continue
                    if node_info[1] in set(string.punctuation):
                        continue
                    if node_info[1]=="'s":
                        continue
                    if len(node_info) == 3:
                        node_char_pos = words_to_char_pos[str(node_info[0])]
                        assert utterance_text[node_char_pos[0]:node_char_pos[1]] == node_info[1]
                    else:
                        node_char_pos = [words_to_char_pos[str(node_info[0])][0], words_to_char_pos[str(node_info[-3])][1]]
                        
                    pos2 = [
                            single_utterance_char_to_word_offset[node_char_pos[0]],
                            single_utterance_char_to_word_offset[node_char_pos[1]-1]
                            ]
                        
                    new_name2 = ' '.join(doc_tokens[pos2[0]:pos2[1]+1]) + '=*=' + str(pos2[0]) + '=*=' + str(pos2[1]) + '-' + str(uidx)
                    
                    srlable = node_info[2]; relable = ''
                    for arg in ('ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5'):
                        if arg in srlable:
                            single_srl = True; arg_nodes.append(new_name2); relable = arg
                            single_single_inner_edges[(new_name1,new_name2)] = "verb-arg"
                            single_utter2inner_edges.append((uidx, new_name2, 'utter-srl-arg'))
                            break
                    for argm in ('ARGM-NEG', 'ARGM-TMP', 'ARGM-MOD', 'ARGM-ADV', 'ARGM-LOC', 'ARGM-MNR', 'ARGM-CAU', 'ARGM-DIR'):
                        if argm in srlable:
                            single_srl = True; relable = argm
                            single_single_inner_edges[(new_name1,new_name2)] = "verb-argm"
                            single_utter2inner_edges.append((uidx, new_name2, 'utter-srl-argm'))
                            break
                    if not relable:
                        continue
                    
                    for o in range(pos2[0], pos2[1]+1):
                        if utter_nodes_have_srl[o] != 1:
                            utter_nodes_have_srl[o] = 1
                            have_signed2 = False
                        else:
                            have_signed2_len += 1
                            
                    single_inner_nodes_to_token_position[new_name2] = pos2
                    single_inner_nodes_to_type[new_name2] = relable
                
                consider_subgraph = True
                if have_signed and have_signed2 and have_signed_len + have_signed2_len < 8: 
                    consider_subgraph = False
                    # print('not consider_subgraph----------------:', single_inner_nodes_to_token_position)
                    continue
                    
                if single_srl and consider_subgraph:
                    inner_nodes_to_token_position[new_name1] = pos1
                    inner_nodes_to_type[new_name1] = 'B-V'
                    for k, v in single_single_inner_edges.items():
                        single_inner_edges[k] = v
                        inner_edges[k] = v
                    for k, v in single_inner_nodes_to_token_position.items():
                        inner_nodes_to_token_position[k] = v
                    for k, v in single_inner_nodes_to_type.items():
                        inner_nodes_to_type[k] = v
                    utter2inner_edges.extend([e for e in single_utter2inner_edges])
                    utter2inner_edges.append((uidx, new_name1, 'utter-srl-verb'))
                    verb_nodes.append(new_name1)
                    
                    # print(arg_nodes)
                    for i1, argnode1 in enumerate(arg_nodes):   # 核心语义节点的连接：ARG0-ARG1, ARG3-ARG4
                        argt1 = inner_nodes_to_type[argnode1]
                        for i2 in range(i1+1, len(arg_nodes)):
                            argnode2 = arg_nodes[i2]; argt2 = inner_nodes_to_type[argnode2]
                            if argt1 == 'ARG0' and argt2 == 'ARG1':
                                inner_edges[(argnode1,argnode2)] = "arg-arg"
                                single_inner_edges[(argnode1,argnode2)] = "arg-arg"
                            if argt1 == 'ARG1' and argt2 == 'ARG0':
                                inner_edges[(argnode2,argnode1)] = "arg-arg"
                                single_inner_edges[(argnode2,argnode1)] = "arg-arg"
                            if argt1 == 'ARG3' and argt2 == 'ARG4':
                                inner_edges[(argnode1,argnode2)] = "arg-arg"
                                single_inner_edges[(argnode1,argnode2)] = "arg-arg"
                            if argt1 == 'ARG4' and argt2 == 'ARG3':
                                inner_edges[(argnode2,argnode1)] = "arg-arg"
                                single_inner_edges[(argnode2,argnode1)] = "arg-arg"  
                            if len(arg_nodes) == 2 and argt1 == 'ARG2':
                                inner_edges[(argnode2,argnode1)] = "arg-arg"
                                single_inner_edges[(argnode2,argnode1)] = "arg-arg"  
                            if len(arg_nodes) == 2 and argt2 == 'ARG2':
                                inner_edges[(argnode1,argnode2)] = "arg-arg"
                                single_inner_edges[(argnode1,argnode2)] = "arg-arg"  

                if args.ARG_ARGM:
                    if len(arg_nodes) == 1:  # 如果只有1个核心语义节点，则所有附加语义节点与核心语义节点连接
                        for edge, edge_rel in single_single_inner_edges.items():
                            s, e = edge
                            if edge_rel == "verb-argm":
                                inner_edges[(arg_nodes[0],e)] = "arg-argm"
                                single_inner_edges[(arg_nodes[0],e)] = "arg-argm"
                    
                    if len(arg_nodes) > 1:   
                        for edge, edge_rel in single_single_inner_edges.items():   # 附加语义节点与其最近的核心语义节点连接
                            s, e = edge
                            if edge_rel == "verb-argm":
                                e_start, e_end = inner_nodes_to_token_position[e]
                                pathl = 100000; tnode = ''
                                for argn in arg_nodes:
                                    s_start, s_end = inner_nodes_to_token_position[argn]
                                    pl = min([abs(s_start-e_start), abs(s_end-e_end)])
                                    if pl < pathl:
                                        pathl = pl
                                        tnode = argn
                                inner_edges[(tnode,e)] = "arg-argm"
                                single_inner_edges[(tnode,e)] = "arg-argm"

            srl_nodes = list(set([srle[0] for srle in inner_edges] + [srle[1] for srle in inner_edges]))
            for i, node1 in enumerate(srl_nodes):
                pos1 = inner_nodes_to_token_position[node1]
                for j in range(i+1, len(srl_nodes)):
                    node2 = srl_nodes[j]
                    pos2 = inner_nodes_to_token_position[node2]
                    if pos1[0] <= pos2[0] and pos1[1] >= pos2[1]:  # pos1 include pos2
                        if (node1, node2) not in inner_edges:
                            inner_edges[(node1, node2)] = "child"
                            single_inner_edges[(node1, node2)] = "child"
                    elif pos1[0] >= pos2[0] and pos1[1] <= pos2[1]:  # pos2 include pos1
                        if (node2, node1) not in inner_edges:
                            inner_edges[(node2, node1)] = "child"
                            single_inner_edges[(node2, node1)] = "child"
            if not verb_nodes:
                continue
            spke = utter_first_spk_map[uidx] 
            if args.SPK_ARG_AND_ARGM:
                utter_inner_nodes = list(set([n[0] for n in single_inner_edges] + [n[1] for n in single_inner_edges]))
                for node in utter_inner_nodes:
                    inner_edges[(spke, node)] = "spk_srl"
            else:
                verbnode2indegree = {v:0 for v in verb_nodes}
                for verbnode in verb_nodes:
                    for e in inner_edges:
                        if verbnode == e[1]:
                            verbnode2indegree[verbnode] += 1
                min_indegree = min(list(verbnode2indegree.values()))
                for verbnode, indegree in verbnode2indegree.items():
                    if indegree == min_indegree:
                        inner_edges[(spke, verbnode)] = "spk_srl"
        
        # same speaker utterances
        same_speaker_utters = []
        for spk, same_utter_idxs in same_speaker_utter.items():
            same_speaker_utters.append(same_utter_idxs)
            
        all_inner_nodes_name = list(inner_nodes_to_token_position.keys())
        all_inner_nodes_pos = list(inner_nodes_to_token_position.values())
        all_inner_nodes_recover_name = []
        for pos in all_inner_nodes_pos:
            all_inner_nodes_recover_name.append(' '.join(doc_tokens[pos[0]:pos[1]+1]))
        for nid1, node_name1 in enumerate(all_inner_nodes_recover_name):
            ns, ne = inner_nodes_to_token_position[all_inner_nodes_name[nid1]][:2]
            is_noun = False
            for posi in doc_tokens_posnoun:
                if ns<=posi[0] and ne>=posi[1]:
                    is_noun = True
                    # print(node_name1)
                    break
            if node_name1.lower() not in stop_words and len(node_name1) > 2 and is_noun:
                for nid2 in range(nid1+1, len(all_inner_nodes_recover_name)):
                    node_name2 = all_inner_nodes_recover_name[nid2]
                    ns2, ne2 = inner_nodes_to_token_position[all_inner_nodes_name[nid2]][:2]
                    is_noun2 = False
                    for posi in doc_tokens_posnoun:
                        if ns2<=posi[0] and ne2>=posi[1]:
                            is_noun2 = True
                            # print(node_name2)
                            break
                    if is_noun2 and compute_f1(node_name1, node_name2) > 0.5 and node_name2.lower() not in stop_words and len(node_name2) > 2:
                        if (all_inner_nodes_name[nid1], all_inner_nodes_name[nid2]) not in inner_edges and (all_inner_nodes_name[nid2], all_inner_nodes_name[nid1]) not in inner_edges:
                            if compute_f1(node_name1, node_name2) > 0.5 and node_name2.lower() not in stop_words and len(node_name2) > 2:
                                if ((all_inner_nodes_name[nid1], all_inner_nodes_name[nid2]) not in inner_edges) and ((all_inner_nodes_name[nid2], all_inner_nodes_name[nid1]) not in inner_edges):
                                    inner_edges[(all_inner_nodes_name[nid1], all_inner_nodes_name[nid2])] = "similar"
                            elif (node_name1 in node_name2 or node_name2 in node_name1) and node_name2.lower() not in stop_words and len(node_name2) > 2:
                                if ((all_inner_nodes_name[nid1], all_inner_nodes_name[nid2]) not in inner_edges) and ((all_inner_nodes_name[nid2], all_inner_nodes_name[nid1]) not in inner_edges):
                                    inner_edges[(all_inner_nodes_name[nid1], all_inner_nodes_name[nid2])] = "similar"

        new_inner_nodes_to_token_position = copy.deepcopy(inner_nodes_to_token_position)
        coref_clusters = []
        for mode, coref_info in data["coref"].items():
            if mode == 'spk':
                for name, cluster_info in coref_info.items():   
                    if name == '#note#':    continue
                    coref_nodes = []
                    person = [recover_name(cluster_info['begin_speaker'][0]).lower()]
                    if 'first_person' in cluster_info:
                        person += cluster_info['first_person']
                    if 'second_person' in cluster_info:
                        person += cluster_info['second_person']
                    if 'third_person' in cluster_info:
                        person += cluster_info['third_person']
                    if len(person) > 1:
                        for pname in person:
                            pnme = pname.lower()
                            if pnme in inner_nodes_to_token_position:
                                poss = inner_nodes_to_token_position[pnme]
                            else:
                                new_name, poss, ui = find_coref_target(pnme, utterances_inner_char_to_word_offset)
                            find = False
                            for node_name, pos in inner_nodes_to_token_position.items():
                                if pos == poss:
                                    find = True
                                    coref_nodes.append(node_name)
                                elif pos[0] <= poss[0] and pos[1] >= poss[1]:
                                    find = True
                                    coref_nodes.append(pnme)
                                    new_inner_nodes_to_token_position[pnme] = poss
                                    inner_nodes_to_type[pnme] = 'coref'
                                    inner_edges[(node_name,pnme)] = "child"
                            if not find:
                                new_inner_nodes_to_token_position[pnme] = poss
                                inner_nodes_to_type[pnme] = 'coref'
                                utter2inner_edges.append((int(ui), pnme, 'utter-addition'))
                                coref_nodes.append(pnme)
                    if len(coref_nodes) > 1:
                        coref_clusters.append(coref_nodes[:])
            else:
                for cluster_info in coref_info:   
                    coref_nodes = []
                    for pname in cluster_info:
                        find = False
                        pnme = pname.lower()
                        if recover_name(pnme) in stop_words:   continue
                        new_name, poss, ui = find_coref_target(pnme, utterances_inner_char_to_word_offset)
                        for node_name, pos in inner_nodes_to_token_position.items():
                            if pos == poss:
                                find = True
                                coref_nodes.append(node_name)
                            elif pos[0] <= poss[0] and pos[1] >= poss[1]:
                                find = True
                                coref_nodes.append(pnme)
                                new_inner_nodes_to_token_position[pnme] = poss
                                inner_nodes_to_type[pnme] = 'coref'
                                inner_edges[(node_name,pnme)] = "child"
                        if not find:
                            new_inner_nodes_to_token_position[pnme] = poss
                            inner_nodes_to_type[pnme] = 'coref'
                            utter2inner_edges.append((int(ui), pnme, 'utter-addition'))
                            coref_nodes.append(pnme)
                    if len(coref_nodes) > 1:
                        coref_clusters.append(coref_nodes[:])
        
        all_finegaine_nodes = new_inner_nodes_to_token_position
        for cluster in coref_clusters:
            for c in cluster:
                try:
                    assert c in all_finegaine_nodes
                except:
                    print(c)
                    print(all_finegaine_nodes)
                    print('================================')
        
        cross_utterance_relations = []
        for rel in data['relations']:
            y = rel['y']; x = rel['x']; 
            if rel['type'] in map_relations and y < len(content_list) and x < len(content_list):
                cross_utterance_relations.append([(y,x), map_relations[rel['type']]])

        dialogue_graph = {
            'utter_nodes_to_word_position':utter_nodes, 'same_speaker_utters': same_speaker_utters, 'all_utter_first_speaker': all_utter_first_speaker, \
            'cross_utterance_relations': cross_utterance_relations, 'utter_nodes_to_fine-grained_nodes': utter2inner_edges, \
            'inner_edges': inner_edges, 'all_coref_clusters': coref_clusters, 'all_finegain_nodes_to_word_position': all_finegaine_nodes, \
            'all_finegain_nodes_to_type': inner_nodes_to_type, 'first_spk_to_pos':first_spk_to_pos
            }

        for qa in data["paragraphs"][0]["qas"]:
            qid = qa["id"]
            question = qa["question"]

            qas_tokens = []
            q_char_to_word_offset = []
            prev_is_whitespace = True
            # Split on whitespace so that different tokens may be attributed to their original position.
            for c in question:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        qas_tokens.append(c)
                    else:
                        qas_tokens[-1] += c
                    prev_is_whitespace = False
                q_char_to_word_offset.append(len(qas_tokens) - 1)

            # ques nodes
            q_noun_chunks_position = qa["question_info"]["noun_chunks_char_offests"]
            q_noun_chunks = qa["question_info"]["noun_chunks"][:]
            for nid, noun in enumerate(qa["question_info"]["noun_chunks"]):
                for nidd, non in enumerate(qa["question_info"]["noun_chunks"]):
                    if nid != nidd and noun in non and len(noun) < len(non) and \
                    q_noun_chunks_position[nidd][0] <= q_noun_chunks_position[nid][0] and q_noun_chunks_position[nidd][1] >= q_noun_chunks_position[nid][1]:
                        if q_noun_chunks[nid]:
                            q_noun_chunks[nid] = ''
            q_noun_to_dialogue_edges = []; q_noun_nodes = {}
            for nid, noun_chunk in enumerate(q_noun_chunks):
                if len(noun_chunk) < 2 and len(noun_chunk) > len(question) - 5:
                    continue
                noun_chunk_name = noun_chunk + '-' + str(nid)
                poss = [q_char_to_word_offset[q_noun_chunks_position[nid][0]], q_char_to_word_offset[q_noun_chunks_position[nid][1]-1]]
                for srl_node in dialogue_graph['all_finegain_nodes_to_word_position']:
                    srl_node_prename = recover_name(srl_node)
                    if (compute_f1(srl_node_prename, noun_chunk) > 0.5 or list(set(srl_node_prename.split()) & set(noun_chunk.split()))) \
                        and srl_node_prename.lower() not in stop_words and srl_node_prename not in set(string.punctuation):
                        q_noun_to_dialogue_edges.append((noun_chunk_name, srl_node)); q_noun_nodes[noun_chunk_name] = poss

            ques_to_dialogue_subgraph = {
                'q_noun_nodes_to_word_position': q_noun_nodes, 'q_noun_to_dialogue_edges': q_noun_to_dialogue_edges
            }
            answers = qa["answers"]
            if is_training:
                for a_id, answer in enumerate(answers):
                    key_inner_nodes = []
                    utterance_id = answer["utterance_id"]
                    answer_text = answer["answer_text"]
                    is_speaker = answer["is_speaker"]
                    if is_speaker:
                        start_position = utterances_inner_char_to_word_offset[utterance_id][0]
                        end_position = utterances_inner_char_to_word_offset[utterance_id][content_list[utterance_id].find(': ')-1]
                        if answer_text.lower() in all_finegaine_nodes:
                            key_inner_nodes.append(answer_text.lower())
                        else:
                            key_inner_nodes.append(all_utter_first_speaker[utterance_id])
                    else:
                        spk_offest = len(content_list[utterance_id][0:content_list[utterance_id].find(': ')+1].split())
                        if utterance_id > 0:
                            start_position = utterances_index[utterance_id-1] + answer["inner_start"]+1 + spk_offest
                            end_position = utterances_index[utterance_id-1] + answer["inner_end"]+1+ spk_offest
                        else:
                            start_position = answer["inner_start"] + spk_offest
                            end_position = answer["inner_end"] + spk_offest
                        if doc_tokens[start_position:end_position+1] != answer_text.split():
                            inner_start = content_list[utterance_id].find(answer_text)
                            start_position = utterances_inner_char_to_word_offset[utterance_id][inner_start]
                            end_position = utterances_inner_char_to_word_offset[utterance_id][inner_start + len(answer_text)-1]
                        
                        for node, posi in all_finegaine_nodes.items():
                            if posi == [start_position, end_position]:
                                key_inner_nodes.append(node)
                            if posi[0] <= start_position and posi[1] >= end_position:
                                key_inner_nodes.append(node)
                            node_text = ' '.join(doc_tokens[posi[0]: posi[1]+1])
                            f1 = compute_f1(node_text, answer_text)
                            if f1 > 0.6:
                                key_inner_nodes.append(node)

                    exp = FriendsQAExample(qid, content_list, dialogue_graph, ques_to_dialogue_subgraph, doc_tokens, qas_tokens, start_position, end_position, 
                                           key_inner_nodes, answer_text, answer_utterance_id = utterance_id)
                    examples.append(exp)
            else:
                ans = [a["answer_text"] for a in answers]
                examples.append(FriendsQAExample(qid, content_list, dialogue_graph, ques_to_dialogue_subgraph, doc_tokens, qas_tokens, None, None, None, answers = ans))
        return examples


def TokenizeTokens(tokens, tokenizer):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_tokens = []
    for (i, token) in enumerate(tokens):
        orig_to_tok_index.append(len(all_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_tokens.append(sub_token)
    return tok_to_orig_index, orig_to_tok_index, all_tokens


def locate_nodes(nodes_to_word_pos, orig_to_tok_index, old_doc_tokens, new_doc_tokens):
    if not nodes_to_word_pos:
        return {}
    new_nodes_to_word_pos = {}
    for node, pos in nodes_to_word_pos.items():
        new_pos = []
        for i in range(0, len(pos), 2):
            p0 = pos[i]; p1 = pos[i+1]
            s = orig_to_tok_index[p0]
            if p1 < len(old_doc_tokens) - 1:
                e = orig_to_tok_index[p1+1]-1
            else:
                e = len(new_doc_tokens) - 1
            new_pos.extend([s,e])
        new_nodes_to_word_pos[node] = new_pos[:]
    return new_nodes_to_word_pos

def silde_nodes(nodes_to_word_pos, doc_start, doc_end, doc_offset):
    new_nodes_to_word_pos = {}; removes = set()
    for node, pos in nodes_to_word_pos.items():
        new_pos = []
        for i in range(0, len(pos), 2):
            p0 = pos[i]; p1 = pos[i+1]
            if p0 >= doc_start and p1 <= doc_end:
                new_pos.extend([p0-doc_start +doc_offset,p1- doc_start +doc_offset])
        if new_pos:
            new_nodes_to_word_pos[node] = new_pos[:]
        else:
            removes.add(node)
    return new_nodes_to_word_pos, removes

def silde_utter_nodes(nodes_to_word_pos, doc_start, doc_end, doc_offset, query_end):
    new_nodes_to_word_pos = {}; removes = set()
    for node, pos in nodes_to_word_pos.items():
        new_pos = []
        p0 = pos[0]; p1 = pos[1]
        if p0 > doc_start and p1 <= doc_end+1:
            new_pos = [p0-doc_start + doc_offset,p1-doc_start + doc_offset]
        elif p0 > doc_start and p0 < doc_end and p1 > doc_end + 1:
            new_pos = [p0-doc_start + doc_offset,-1]
        elif p0 <= doc_start and p1 < doc_end+1 and p1 > doc_start:
            new_pos = [query_end,p1-doc_start + doc_offset]
        if new_pos:
            new_nodes_to_word_pos[node] = new_pos
        else:
            removes.add(node)
    return new_nodes_to_word_pos, removes

def silde_utter_involved_edges(utter_nodes_to_finegrained_nodes, cross_utterance_relations, remove_utter_nodes, all_removes):
    new_cross_utterance_relations = []
    for cross_info in cross_utterance_relations:
        if cross_info[0][0] not in remove_utter_nodes and cross_info[0][1] not in remove_utter_nodes:
            new_cross_utterance_relations.append(cross_info[0])
    utter2spk = []
    utter2srl = []
    for (u, f, rel) in utter_nodes_to_finegrained_nodes:
        if u not in remove_utter_nodes and f not in all_removes:
            if rel == 'utter-spk':
                utter2spk.append((u,f))
            else:
                utter2srl.append((u,f))
    return new_cross_utterance_relations, utter2spk, utter2srl

def silde_inner_edges(edges, removes):
    srl_edges = []; child_edges = []; similar_edges = []; spksrl_edges = []
    for edge, rell in edges.items():
        if edge[0] not in removes and edge[1] not in removes:
            # print(edge)
            if rell in ('arg-arg', 'arg-argm', 'verb-arg','verb-argm'):
                srl_edges.append(edge)
            if rell == 'child':
                child_edges.append(edge)
            elif rell == 'similar':
                similar_edges.append(edge)
            elif rell == 'spk_srl':
                spksrl_edges.append(edge)
    return srl_edges, child_edges, similar_edges, spksrl_edges

def silde_cluster_edges(clusters, removes):
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for c in cluster:
            if c not in removes:
                new_cluster.append(c)
        new_clusters.append(new_cluster[:])
    return new_clusters

def silde_cluster(clusters, doc_start, doc_end, doc_offset):
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for pos in cluster:
            new_pos = []
            for i in range(0, len(pos), 2):
                p0 = pos[i]; p1 = pos[i+1]
                if p0 > doc_start and p1 <= doc_end:
                    new_pos.extend([p0-doc_start + doc_offset,p1-doc_start + doc_offset])
            if new_pos:
                new_cluster.append(new_pos)
        if len(new_cluster) > 1:
            new_clusters.append(new_cluster[:])
    return new_clusters

def text_position(new_nodes_to_word_pos, old_nodes_to_word_pos, new_tokens, old_tokens):
    for node, pos1 in new_nodes_to_word_pos.items():
        pos2 = old_nodes_to_word_pos[node]
        if len(pos1) == len(pos2) and len(pos1) == 2:
            assert old_tokens[pos2[0]:pos2[1]+1] == new_tokens[pos1[0]:pos1[1]+1]

def node_name_to_id(nodes_to_token_position, ques_nodes_to_ids=None, ques_nodes_to_token_position=None, sort = False):
    nodes_to_ids, new_nodes_to_token_position = {}, {}
    if ques_nodes_to_ids and ques_nodes_to_token_position:
        nodes_to_ids, new_nodes_to_token_position = copy.deepcopy(ques_nodes_to_ids), copy.deepcopy(ques_nodes_to_token_position)
        for i, node in enumerate(nodes_to_token_position):
            ii = i + len(ques_nodes_to_ids)
            nodes_to_ids[node] = ii
            new_nodes_to_token_position[ii] = nodes_to_token_position[node]
        # print('====', list(nodes_to_ids.values()))
    else:
        # for i, node in enumerate(nodes_to_token_position):
        ll = list(nodes_to_token_position.keys())
        if sort:  ll = sorted(ll)
        for i, k in enumerate(ll):
            nodes_to_ids[k] = i
            new_nodes_to_token_position[i] = nodes_to_token_position[k]
        # print('=================', list(nodes_to_ids.values()))
    return nodes_to_ids, new_nodes_to_token_position


def friendsqa_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training, 
    MAX_UTTERANCE_INNER_NUM = args.MAX_UTTERANCE_INNER_NODE_NUM, 
    MAX_UTTERANCE_NUM = args.MAX_UTTERANCE_NUM,
    MAX_SPEAKER_NUM = args.MAX_SPEAKER_NUM, 
    MAX_QUESTION_INNER_NODE_NUM = args.MAX_QUESTION_INNER_NODE_NUM
):
    # print('----------------------------------------')
    features = []

    tok_to_orig_index, orig_to_tok_index, all_doc_tokens = TokenizeTokens(example.doc_tokens, tokenizer)

    utter_nodes_to_token_position = locate_nodes(example.dialogue_graph['utter_nodes_to_word_position'], orig_to_tok_index, example.doc_tokens, all_doc_tokens)
    for utter_node, p in utter_nodes_to_token_position.items():
        assert all_doc_tokens[p[-1]] == tokenizer.sep_token
    first_spk_to_token_position = locate_nodes(example.dialogue_graph['first_spk_to_pos'], orig_to_tok_index, example.doc_tokens, all_doc_tokens)
    finegain_nodes_to_token_position = locate_nodes(example.dialogue_graph['all_finegain_nodes_to_word_position'], orig_to_tok_index, example.doc_tokens, all_doc_tokens)

    if is_training:
        answer_text = example.answer
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, answer_text
        )

    q_tok_to_orig_index, q_orig_to_tok_index, all_ques_tokens = TokenizeTokens(example.ques_tokens, tokenizer)
    q_nodes_to_token_position = locate_nodes(
        example.ques_to_dialogue_subgraph['q_noun_nodes_to_word_position'], q_orig_to_tok_index, example.ques_tokens, all_ques_tokens
        )

    q_noun_to_dialogue_edges = example.ques_to_dialogue_subgraph['q_noun_to_dialogue_edges']
    remove_ques_nodes = set()
    if len(all_ques_tokens) > max_query_length:
        truncated_query = all_ques_tokens[-max_query_length:]
        if q_nodes_to_token_position:
            for token, pos in q_nodes_to_token_position.items():
                if pos[1] >= max_query_length:
                    remove_ques_nodes.add(token)
            if remove_ques_nodes:
                q_noun_to_dialogue_edges_new = []
                for node_pair in q_noun_to_dialogue_edges:
                    if node_pair[0] not in remove_ques_nodes:
                        q_noun_to_dialogue_edges_new.append(node_pair)
                q_noun_to_dialogue_edges = q_noun_to_dialogue_edges_new
    else:
        truncated_query = all_ques_tokens
    truncated_query = tokenizer.convert_tokens_to_ids(truncated_query)
    
    # tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    if tokenizer_type != "xlnet" and q_nodes_to_token_position:
        new_q_nodes_to_token_position = {}
        for n, p in q_nodes_to_token_position.items():
            new_q_nodes_to_token_position[n] = [p[0]+1, p[1]+1]
        q_nodes_to_token_position = new_q_nodes_to_token_position
        
    spans = []
    all_doc_tokens = all_doc_tokens[:-1]
    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )
        
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]
            
        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])

        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1
        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        doc_start = span["start"]
        doc_end = span["start"] + span["length"] - 1

        if tokenizer.padding_side == "left":
            doc_offset = 0
        else:
            doc_offset = len(truncated_query) + sequence_added_tokens

        query_end = span["truncated_query_with_special_tokens_length"]
        cur_utter_nodes_to_token_position, remove_utter_nodes = silde_utter_nodes(utter_nodes_to_token_position, doc_start, doc_end, doc_offset, query_end)

        for utter_node, p in cur_utter_nodes_to_token_position.items():
            if p[-1] == -1:
                cur_utter_nodes_to_token_position[utter_node] = [p[0], len(span['tokens'])-1]
            try:
                assert span['tokens'][p[-1]] == tokenizer.sep_token
            except:
                print(span['tokens'][p[0]:p[-1]+1])

        cur_finegain_nodes_to_token_position, remove_finegain_nodes = silde_nodes(finegain_nodes_to_token_position, doc_start, doc_end, doc_offset)
        cur_finegain_nodes_to_type = {}
        for n, t in example.dialogue_graph['all_finegain_nodes_to_type'].items():
            if n not in remove_finegain_nodes:
                cur_finegain_nodes_to_type[n] = t
    
        first_spk_nodes_to_token_position, remove_spk_nodes = silde_nodes(first_spk_to_token_position, doc_start, doc_end, doc_offset)

        srl_final_edges, child_edges, similar_edges, spksrl_edges = silde_inner_edges(example.dialogue_graph['inner_edges'], remove_finegain_nodes)
        finegain_coref_clusters = silde_cluster_edges(example.dialogue_graph['all_coref_clusters'], remove_finegain_nodes)
        cross_utterance_relations, utter2spk, utter2srl = silde_utter_involved_edges(
            example.dialogue_graph['utter_nodes_to_fine-grained_nodes'], example.dialogue_graph['cross_utterance_relations'], remove_utter_nodes, remove_finegain_nodes
            )
        q_noun_to_final_dialogue_edges = []
        for edge in q_noun_to_dialogue_edges:
            if edge[1] not in remove_finegain_nodes:
                q_noun_to_final_dialogue_edges.append(edge)
       
        start_position = 0; end_position = 0
        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
            else:
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
    
        finegain_nodes_to_ids, new_finegain_nodes_to_token_position = node_name_to_id(cur_finegain_nodes_to_token_position)
        new_finegain_nodes_to_type = {}
        for n, t in cur_finegain_nodes_to_type.items():
            new_finegain_nodes_to_type[finegain_nodes_to_ids[n]] = t
        utter_nodes_to_ids, new_utter_nodes_to_token_position = node_name_to_id(cur_utter_nodes_to_token_position, sort = True)
        first_spk_nodes_to_ids, new_first_spk_nodes_to_token_position = node_name_to_id(first_spk_nodes_to_token_position)

        all_utter_first_spk = {}
        for i, spk in enumerate(example.dialogue_graph['all_utter_first_speaker']):
            if i not in remove_utter_nodes and spk not in remove_spk_nodes:
                uid = utter_nodes_to_ids[i]; sid = first_spk_nodes_to_ids[spk]
                if uid < MAX_UTTERANCE_NUM and sid < MAX_SPEAKER_NUM:
                    all_utter_first_spk[uid] = sid
        
        MAX_UTTERANCE_INNER_NODE_NUM = MAX_UTTERANCE_INNER_NUM
        
        for utter_node, p in new_utter_nodes_to_token_position.items():
            assert span['tokens'][p[-1]] == tokenizer.sep_token
        
        answer_utterance_id = MAX_UTTERANCE_NUM
        if example.answer_utterance_id > -1 and is_training and start_position and end_position:
            answer_utterance_id = utter_nodes_to_ids[example.answer_utterance_id] if example.answer_utterance_id in utter_nodes_to_ids else MAX_UTTERANCE_NUM
            if answer_utterance_id < MAX_UTTERANCE_NUM:
                assert new_utter_nodes_to_token_position[answer_utterance_id][0] <= start_position and new_utter_nodes_to_token_position[answer_utterance_id][1] >= end_position

        key_inner_nodes = np.zeros((MAX_UTTERANCE_INNER_NODE_NUM))
        if example.key_inner_nodes:
            for raw_node_name in example.key_inner_nodes:
                if raw_node_name in remove_finegain_nodes:
                    continue
                nid = finegain_nodes_to_ids[raw_node_name]
                if nid < MAX_UTTERANCE_INNER_NODE_NUM:
                    key_inner_nodes[nid] = 1

        # srl-srl
        inner_mask = np.zeros((MAX_UTTERANCE_INNER_NODE_NUM, MAX_UTTERANCE_INNER_NODE_NUM, 4))
        inner_type = np.zeros((MAX_UTTERANCE_INNER_NODE_NUM))
        for edge in srl_final_edges:  
            s, e = finegain_nodes_to_ids[edge[0]], finegain_nodes_to_ids[edge[1]]
            if s < MAX_UTTERANCE_INNER_NODE_NUM and e < MAX_UTTERANCE_INNER_NODE_NUM:
                inner_mask[s,e,0] = 1
                inner_mask[e,s,0] = 1
                inner_mask[s,s,0] = 1
                inner_mask[e,e,0] = 1
            
        for n, node_type in enumerate(new_finegain_nodes_to_type):
            if node_type in srl_relations:
                inner_type[n] = srl_relations[node_type]       
        
        # utter-to-self
        qas_utter_self_mask = np.zeros((len(span["input_ids"]), len(span["input_ids"])))
        for idxs in new_utter_nodes_to_token_position.values():
            qas_utter_self_mask[idxs[0]:idxs[1], idxs[0]:idxs[1]] = 1
        qas_utter_self_mask[0:query_end, :] = 1
        qas_utter_self_mask[:, 0:query_end] = 1
        for i in range(len(span["input_ids"])):
            if span["input_ids"][i] == tokenizer.pad_token_id:
                qas_utter_self_mask[i : i + 1, :] = 0
                qas_utter_self_mask[:, i : i + 1] = 0
        
        # coref
        for cluster in finegain_coref_clusters:
            for i1, c1 in enumerate(cluster):
                s = finegain_nodes_to_ids[c1]
                if s < MAX_UTTERANCE_INNER_NODE_NUM:
                    for i2 in range(i1+1, len(cluster)):
                        e = finegain_nodes_to_ids[cluster[i2]]
                        if e < MAX_UTTERANCE_INNER_NODE_NUM:
                            inner_mask[s,e,1] = 1
                            inner_mask[e,s,1] = 1
                            inner_mask[s,s,1] = 1
                            inner_mask[e,e,1] = 1
        for edge in similar_edges:
            s, e = finegain_nodes_to_ids[edge[0]], finegain_nodes_to_ids[edge[1]]
            if s < MAX_UTTERANCE_INNER_NODE_NUM and e < MAX_UTTERANCE_INNER_NODE_NUM:
                inner_mask[s,e,2] = 1
                inner_mask[e,s,2] = 1
                inner_mask[s,s,2] = 1
                inner_mask[e,e,2] = 1
        for edge in child_edges:
            s, e = finegain_nodes_to_ids[edge[0]], finegain_nodes_to_ids[edge[1]]
            if s < MAX_UTTERANCE_INNER_NODE_NUM and e < MAX_UTTERANCE_INNER_NODE_NUM:
                inner_mask[s,e,2] = 1
                inner_mask[e,s,2] = 1
                inner_mask[s,s,2] = 1
                inner_mask[e,e,2] = 1
        # spk-srl_root
        for edge in spksrl_edges:
            s, e = finegain_nodes_to_ids[edge[0]], finegain_nodes_to_ids[edge[1]]
            if s < MAX_UTTERANCE_INNER_NODE_NUM and e < MAX_UTTERANCE_INNER_NODE_NUM:
                inner_mask[s,e,3] = 1
                inner_mask[e,s,3] = 1
                
        # utter-srl
        utterance_to_srl_s = []; utterance_to_srl_e = []
        for edge in utter2srl:
            s, e = utter_nodes_to_ids[int(edge[0])], finegain_nodes_to_ids[edge[1]]
            utterance_to_srl_s.append(s); utterance_to_srl_e.append(e)

        # utter-spk
        utterance_to_speaker_s = []; utterance_to_speaker_e = []
        for edge in utter2spk:
            s, e = utter_nodes_to_ids[edge[0]], first_spk_nodes_to_ids[edge[1]]
            if s < MAX_UTTERANCE_NUM and e < MAX_SPEAKER_NUM:
                utterance_to_speaker_s.append(s); utterance_to_speaker_e.append(e)
        
        utterance_from_utterance_s, utterance_from_utterance_e = [], []
        for edge in cross_utterance_relations:
            s, e = utter_nodes_to_ids[edge[0]], utter_nodes_to_ids[edge[1]]
            if s < MAX_UTTERANCE_NUM and e < MAX_UTTERANCE_NUM:
                utterance_from_utterance_s.append(s)
                utterance_from_utterance_e.append(e)
        
        # ques-inner
        question_inner_edges_s, question_inner_edges_e = [], []
        for edge in q_noun_to_final_dialogue_edges:
            e = finegain_nodes_to_ids[edge[1]]
            if e < MAX_UTTERANCE_INNER_NODE_NUM:
                question_inner_edges_s.append(0); question_inner_edges_e.append(e)
        
        new_finegain_nodes_to_utter_id = {}

        for node, pos in new_finegain_nodes_to_token_position.items():
            ppos = sorted(pos); mi, ma = min(ppos), max(ppos)
            find = False
            for u, pos2 in new_utter_nodes_to_token_position.items():
                if mi >= pos2[0] and ma <= pos2[1]:
                    new_finegain_nodes_to_utter_id[node] = u
                    find = True
                    break
            if not find:
                new_finegain_nodes_to_utter_id[node] = -1
                
        # ques-utter
        question_utterance_edges_s = []; question_utterance_edges_e = []
        for nid in utter_nodes_to_ids.values():
            if nid < MAX_UTTERANCE_NUM:
                question_utterance_edges_s.append(0)
                question_utterance_edges_e.append(nid)
                    
        graph_dict1 = {
            ('utterance', 'from', 'utterance'): (utterance_from_utterance_s, utterance_from_utterance_e),
            ('utterance', 'to', 'utterance'): (utterance_from_utterance_e, utterance_from_utterance_s),
            ('utterance', 'speaker', 'utter_inner_node'): (utterance_to_speaker_s, utterance_to_speaker_e),
            ('utter_inner_node', 'speak', 'utterance'): (utterance_to_speaker_e, utterance_to_speaker_s),
            ('question', 'link1', 'utterance'): (question_utterance_edges_s, question_utterance_edges_e),
            ('utterance', 'link2', 'question'): (question_utterance_edges_e, question_utterance_edges_s),
        }
        num_nodes_dict1 = {
            'utterance': MAX_UTTERANCE_NUM,
            'utter_inner_node': MAX_SPEAKER_NUM,
            'question': 1,
        }
        graph1 = [graph_dict1, num_nodes_dict1]
            
        utter_p_mask = [0] * MAX_UTTERANCE_NUM
        for i in utter_nodes_to_ids.values():
            utter_p_mask[i] = 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        graph_info_dict = {
            'finegain_nodes_to_token_position':new_finegain_nodes_to_token_position, 
            'spk_nodes_to_token_position':new_first_spk_nodes_to_token_position, 
            'finegain_nodes_to_utter_id':new_finegain_nodes_to_utter_id, 
            'finegain_nodes_to_type':new_finegain_nodes_to_type,
            'utter_nodes_to_token_position':new_utter_nodes_to_token_position,
            'all_utter_first_spk':all_utter_first_spk, 'inner_mask':inner_mask, 'srl_type':inner_type, 'qas_utter_self_mask':qas_utter_self_mask,
            'answer_utterance_id':answer_utterance_id, 'utter_p_mask':utter_p_mask,
            'key_inner_nodes':key_inner_nodes
        }
        query_mapping = np.zeros_like(span["attention_mask"])
        query_mapping[:query_end] = 1

        Feature = FriendsQAFeature(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                qas_id=example.qas_id,
                query_end=query_end,
                query_mapping=query_mapping,
                graph_info_dict = graph_info_dict,
                graph_info = [graph1],
            )
        features.append(Feature)
    return features


def friendsqa_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def friendsqa_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    # Defining helper methods
    random.seed(42)
    features = []

    threads = min(threads, cpu_count()) 
    with Pool(threads, initializer=friendsqa_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            friendsqa_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert friendsqa examples to features",
                disable=not tqdm_enabled,
            )
        )
    # for i in tqdm(range(len(examples))):
    #     # try:
    #     features.append(friendsqa_convert_example_to_features(examples[i], 
    #                                                 max_seq_length=max_seq_length,
    #                                                 doc_stride=doc_stride,
    #                                                 max_query_length=max_query_length,
    #                                                 padding_strategy=padding_strategy,
    #                                                 is_training=is_training, 
    #                                                 tokenizer = tokenizer))
    
    new_features = []
    unique_id = 1000000000
    example_index = 0; feature_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            example_feature.feature_index = feature_index
            new_features.append(example_feature)
            unique_id += 1; feature_index += 1
        example_index += 1
    features = new_features
    del new_features
    
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")
     
        dataset = Dataset(features)
        return features, dataset
    else:
        return features



