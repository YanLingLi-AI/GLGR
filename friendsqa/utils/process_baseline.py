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
from utils.config import *


# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet", "longformer"}

logger = logging.get_logger(__name__)


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
    # if len(doc_spans) == 1:
    # return True
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


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class FriendsQAExample(object):
    def __init__(self, qid, content_list, doc_tokens,  ques_tokens, start_position, end_position, answer_text = None, answers = None):
        self.ques_tokens = ques_tokens
        self.qas_id = qid
        self.answer = answer_text
        self.answers = answers
        self.start_position = start_position
        self.end_position = end_position
        self.content_list = content_list
        self.doc_tokens = doc_tokens


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


class FriendsQAResult:
    def __init__(self, unique_id, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits    


class FriendsQAResult2:
    def __init__(self, unique_id, start_top_log_probs, start_top_index, end_top_log_probs, end_top_index):
        self.unique_id = unique_id
        self.start_logits = start_top_log_probs
        self.start_top_index  = start_top_index 
        self.end_logits = end_top_log_probs
        self.end_top_index = end_top_index


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
        
        for i, data in enumerate(input_data):
            data['srl'] = srl_data[i]
                   
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
        
        for i, data in enumerate(input_data):
            data['srl'] = srl_data[i]
            
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
        examples = []; srl_data = data['srl']
        content_list = []
        for ui, utterance in enumerate(data["paragraphs"][0]["utterances:"]):
            content_list.append(srl_data[str(ui)]['utterance'])
        
        doc_tokens, char_to_word_offset = [], []; context_text = ''; utterances_index = []  # record the length of each utterances including [SEP]
        utterances_inner_char_to_word_offset = {}
        
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
            # utterances_index = utterances_index[:-1]

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

            answers = qa["answers"]
            if is_training:
                for a_id, answer in enumerate(answers):
                    utterance_id = answer["utterance_id"]
                    answer_text = answer["answer_text"]

                    is_speaker = answer["is_speaker"]
                    if is_speaker:
                        start_position = utterances_inner_char_to_word_offset[utterance_id][0]
                        end_position = utterances_inner_char_to_word_offset[utterance_id][content_list[utterance_id].find(': ')-1]
                    else:
                        spk_offest = len(content_list[utterance_id][0:content_list[utterance_id].find(': ')+1].split())
                        if utterance_id > 0:
                            start_position = utterances_index[utterance_id-1] + answer["inner_start"]+1 + spk_offest
                            end_position = utterances_index[utterance_id-1] + answer["inner_end"]+1+ spk_offest
                        else:
                            start_position = answer["inner_start"] + spk_offest
                            end_position = answer["inner_end"] + spk_offest

                        if doc_tokens[start_position:end_position+1] != answer_text.split():  # 标注错误！！
                            inner_start = content_list[utterance_id].find(answer_text)
                            start_position = utterances_inner_char_to_word_offset[utterance_id][inner_start]
                            end_position = utterances_inner_char_to_word_offset[utterance_id][inner_start + len(answer_text)-1]
                            answer_text = ' '.join(doc_tokens[start_position:end_position+1])

                    exp = FriendsQAExample(qid, content_list, doc_tokens, qas_tokens, start_position, end_position, answer_text)
                    examples.append(exp)
            else:
                ans = [a["answer_text"] for a in answers]
                examples.append(FriendsQAExample(qid, content_list, doc_tokens, qas_tokens, None, None, answers = ans))
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

         
def friendsqa_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training, 
):
    features = []

    tok_to_orig_index, orig_to_tok_index, all_doc_tokens = TokenizeTokens(example.doc_tokens[:-1], tokenizer)
    
    if is_training:
        answer_text = example.answer
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens[:-1]) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, answer_text
        )
    
    q_tok_to_orig_index, q_orig_to_tok_index, all_ques_tokens = TokenizeTokens(example.ques_tokens, tokenizer)

    if len(all_ques_tokens) > max_query_length:
        truncated_query = all_ques_tokens[-max_query_length:]
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
        
    spans = []
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
                desc="convert molweni examples to features",
                disable=not tqdm_enabled,
            )
        )
    
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



if __name__ == "__main__":
    input_file = "/SISDC_GPFS/Home_SE/hy-suda/lyl/molweni/data/train.json"
    # speaker_mask_path = "data/speaker_mask_dev.json"

    from transformers import XLNetTokenizerFast, ElectraTokenizer, BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')
    tokenizer = ElectraTokenizer.from_pretrained('/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/electra-base/')
    processor = FriendsQAProcessor(tokenizer=tokenizer, threads = 15)
    examples = processor.get_train_examples(data_dir = "/SISDC_GPFS/Home_SE/hy-suda/lyl/MDRC-Graph/friendsqa/data", filename="train")
    # for exp in tqdm(examples):
    #     friendsqa_convert_example_to_features(exp, tokenizer, 512, 128, 32, 'max_length', True)
    # print('===========================================')
    # friendsqa_convert_examples_to_features(examples, tokenizer, 512, 128, 32, True, threads=15)

        
    