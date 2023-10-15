import os
import argparse
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


parser = argparse.ArgumentParser(description='Parameters for FriendsQA dataset')

# Required parameters
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    # required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    # required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    # required=True,
    help="The output directory where the model checkpoints and predictions will be written.",
)

# Other parameters
parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    help="The input data dir. Should contain the .json files for the task."
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--train_file",
    default=None,
    type=str,
    help="The input training file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--dev_file",
    default=None,
    type=str,
    help="The input evaluation file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--test_file",
    default=None,
    type=str,
    help="The input test file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--null_score_diff_threshold",
    type=float,
    default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.",
)

parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
    "longer than this will be truncated, and sequences shorter than this will be padded.",
)
parser.add_argument(
    "--doc_stride",
    default=128,
    type=int,
    help="When splitting up a long document into chunks, how much stride to take between chunks.",
)
parser.add_argument(
    "--max_query_length",
    default=32,
    type=int,
    help="The maximum number of tokens for the question. Questions longer than this will "
    "be truncated to this length.",
)
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
)
parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
)

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument("--learning_rate", default=1.2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--learning_rate2", default=1e-3, type=float, help="The initial learning rate for Adam.")


parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
)
parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help="The maximum length of an answer that can be generated. This is needed because the start "
    "and end predictions are not conditioned on one another.",
)
parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal QuAC evaluation.",
)
parser.add_argument(
    "--lang_id",
    default=0,
    type=int,
    help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
)

parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
)
parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
parser.add_argument(
    "--cache_prefix",
    default=None,
    type=str,
    help="prefix for cached file of datasets, features, and examples",
)

parser.add_argument('--warmup_proportion', type=float, default=0.01)
parser.add_argument('--SPK_ARG_AND_ARGM', type=bool, default=True)
parser.add_argument('--question_mean_pool', action="store_true")
parser.add_argument('--no_srl_role', action="store_true")
parser.add_argument('--no_srl', action="store_true")
parser.add_argument('--no_child', action="store_true")
parser.add_argument('--srl_fullconnected', action="store_true")
parser.add_argument('--no_coref', action="store_true")
parser.add_argument('--same_spk_utr', action="store_true")
parser.add_argument('--ARG_ARGM', action="store_true")
parser.add_argument('--differ_learning_rates', action="store_true")

parser.add_argument('--MAX_UTTERANCE_INNER_NODE_NUM', type=int, default=172)
parser.add_argument('--MAX_UTTERANCE_NUM', type=int, default=38)
parser.add_argument('--MAX_QUESTION_INNER_NODE_NUM', type=int, default=6)
parser.add_argument('--MAX_SPEAKER_NUM', type=int, default=6)
parser.add_argument('--b', type=float, default=0.1)
parser.add_argument('--flood', action="store_true")

parser.add_argument('--num_gat_layers', type=int, default=2)
parser.add_argument('--num_gat_heads', type=int, default=8)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--inner_graph_layers', type=int, default=2)
parser.add_argument('--gnn', type=str, default="GAT")
parser.add_argument('--mode', type=str, default="simple")
parser.add_argument('--start_n_top', type=int, default=5)
parser.add_argument('--end_n_top', type=int, default=5)
parser.add_argument('--utter_num_mode', type=int, default=1)
args = parser.parse_args()

