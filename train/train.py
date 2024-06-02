# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import random
import argparse

from src.data.loader import check_data_params, load_data, load_data_video
from src.model import check_model_params, build_model, build_model_video
import torch
import numpy as np
import os
import time

# distributed torch debug
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")


    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")

    # memory parameters
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")
    parser.add_argument("--word_mass", type=float, default=0,
                        help="Randomly mask input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")
    parser.add_argument("--mono_data_ratio", type=float, default=1,
                        help="Ratio of data used")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--min_len", type=int, default=0,
                        help="Minimum length of sentences (after BPE)")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="Log printing interval")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")
    parser.add_argument("--lambda_mass", type=str, default="1",
                        help="MASS coefficient")
    parser.add_argument("--lambda_span", type=str, default="10000",
                        help="Span coefficient")

    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--mass_steps", type=str, default="",
                        help="MASS prediction steps")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")
    parser.add_argument("--text_model_path", type=str, default="")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")


    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # online self-training
    parser.add_argument('--stbt', action='store_true', default=False,
                        help='Use self-training back-translation')
    parser.add_argument("--stbt_start_epoch", type=int, default=0,
                        help="self-training start epoch")
    parser.add_argument("--lambda_st", type=str, default="0.01",
                        help="ST coefficient")
    parser.add_argument("--stbt_add_noise", action='store_true', default=False,
                        help="add noise in src sentence in self-training")

    # seed
    parser.add_argument("--seed", type=int, default=-1, help="If >= 0, set the seed")

    # video model
    parser.add_argument("--video_batch_size", type=int, default=32,
                        help="Number of videolf.output_frames per batch")
    parser.add_argument('--video_data_path', type=str, default='bair.hdf5')
    parser.add_argument('--sequence_length', type=int, default=8)  #
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--video_model_path', type=str, default='bair.hdf5')
    parser.add_argument('--vqvae_path', type=str,
                        default='/vqvae/epoch=1-step=179999.ckpt')
    parser.add_argument('--videoAttentionStack_hidden_dim', type=int, default=576)
    parser.add_argument('--n_cond_frames', type=int, default=2)

    return parser


def main(params):
    # load data
    start = time.time()
    data = load_data(params)
    data_video = load_data_video(params)
    end = time.time()

    # build model
    if params.encoder_only:
        model = build_model(params, data['dico'])
    else:
        encoder, decoder = build_model(params, data['dico'])
        VideoGPT = build_model_video(params)

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # freeze
    if params.freezeEnc:
        for param in VideoGPT.vqvae.encoder.parameters():
            param.requires_grad = False

    if params.freezeDec:
        for param in decoder.parameters():
            param.requires_grad = False
        for param in VideoGPT.vqvae.decoder.parameters():
            param.requires_grad = False

    if params.freezeTextModel:
        for param in decoder.parameters():
            param.requires_grad = False
        for param in encoder.parameters():
            param.requires_grad = False

    if params.freezeVideoModel:
        for param in VideoGPT.parameters():
            param.requires_grad = False

    flag = -1
    # language model training
    for _ in range(0, params.max_epoch):
        flag = flag + 1
        logger.info("============ Starting epoch %i ... ============" % epoch)
        if params.freezeInturn:
            if flag % 2 == 0:
                for param in VideoGPT.parameters():
                    param.requires_grad = False
            elif flag % 2 == 1:
                for param in VideoGPT.parameters():
                    param.requires_grad = False

        n_sentences = 0


        while n_sentences < params.max_epoch:
            # denoising auto-encoder steps
            for lang in shuf_order(params.ae_steps):
                if lang == 'video':
                    visual_mt_step(lang, lang, params.lambda_ae)

                elif lang == 'en':
                    mt_step(lang, lang, params.lambda_ae)
            # back-translation steps
            for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                # bt_step(lang1, lang2, lang3, params.lambda_bt)
                if lang1 == 'en' and lang2 == 'video':
                    bt_step_t2v2t(lang1, lang2, lang1, params.lambda_bt)
                elif lang1 == 'video' and lang2 == 'en':
                    bt_step_v2t2v(lang1, lang2, lang3, params.lambda_bt)

            iter()

        logger.info("============ End of epoch %i ============" % epoch)


        # end of epoch
        # save_best_model(scores) # after write evaluator
        save_periodic()
        epoch += 1


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    if params.seed >= 0:
        print('| Set seed {}'.format(params.seed))
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        random.seed(params.seed)
        np.random.seed(params.seed)

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        # params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.exp_id = 'debug_%08i' % 1
        params.debug_slurm = True
        params.debug_train = True

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
