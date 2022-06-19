import argparse
import time
import torch
from loguru import logger
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
import transformers
import pickle
import sys
from utils import set_logger, set_random_seed
from sklearn.model_selection import train_test_split
from data_parallel import BalancedDataParallel
from transformers import GPT2LMHeadModel, GPT2Config
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from dataset import CPMDataset
from torch.utils.tensorboard import SummaryWriter


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_ids', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--vocab_path', default='vocab/chinese_vocab.model', type=str, required=False,
                        help='sp模型路径')
    parser.add_argument('--model_config', default='config/cpm-small.json', type=str, required=False,
                        help='需要从头训练一个模型时，模型参数的配置文件')
    parser.add_argument('--train_path', default='data/train.pkl', type=str, required=False, help='经过预处理之后的数据存放路径')
    parser.add_argument('--max_len', default=200, type=int, required=False, help='训练时，输入数据的最大长度')

    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--epochs', default=40, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练的batch size')
    parser.add_argument('--gpu0_bsz', default=6, type=int, required=False, help='0号卡的batch size')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument("--save_step", type=int, default=1, help="every eval_step to save model")
    parser.add_argument('--gradient_accumulation_steps', default=6, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_path', default='output/train', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--seed', type=int, default=1234, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')
    args = parser.parse_args()
    return args


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=5)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_dataset(logger, args):
    """
    加载训练集
    """
    logger.info("loading training dataset")
    train_path = args.train_path

    with open(train_path, "rb") as f:
        train_list = pickle.load(f)

    # test
    # train_list = train_list[:24]
    logger.info('len of train data:{}'.format(len(train_list)))
    train_dataset = CPMDataset(train_list, args.max_len)

    return train_dataset


def train(model, logger, train_dataset, writer, args):
    model.train()
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info('start training')
    device = args.device
    ignore_index = args.ignore_index
    step = 0
    train_loss = 0
    train_acc = 0
    log_step = args.log_step
    save_step = args.save_step

    # ========== start training ========== #
    for epoch in range(args.epochs):
        logger.info('start {}th epoch training'.format(epoch + 1))
        for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
            step += 1
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()  # 多卡损失的均值

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            batch_acc = batch_correct_num / batch_total_num
            train_loss += loss
            train_acc += batch_acc

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if step % args.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if step % log_step == 0:
                train_loss = train_loss / log_step
                train_acc = train_acc / log_step
                # 训练集指标
                logger.info('Epoch {} step {} train Loss {:.4f}, train ACC {:.4f}'.format(epoch + 1, step, train_loss,
                                                                                          train_acc))
                writer.add_scalar('train loss', train_loss, step)
                writer.add_scalar('train acc', train_acc, step)
                train_loss = 0
                train_acc = 0

            if step % save_step == 0:
                logger.info('Saving model at Epoch {} step {}'.format(epoch + 1, step))
                model_path = join(args.output_path, 'epoch_{}-step_{}'.format(epoch + 1, step))
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(model_path)

    logger.info('training finished')


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def main():
    # 初始化参数
    args = set_args()

    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    args.cuda = not args.no_cuda

    # if args.batch_size < 2048 and args.warmup_steps <= 4000:
    #     print('[Warning] The warmup steps may be not enough.\n' \
    #           '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
    #           'Using smaller batch w/o longer warmup may cause ' \
    #           'the warmup stage ends with only little data trained.')

    # 创建日志对象
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    # 初始化tensorboard
    writer = SummaryWriter(args.output_path)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    # 设置随机种子
    set_random_seed(args.seed, args.cuda)

    # 初始化tokenizer
    # tokenizer = CpmTokenizer(vocab_file=args.vocab_path)
    # args.eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    # args.pad_id = tokenizer.pad_token_id

    # 创建模型的输出目录
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # 创建模型
    if args.pretrained_model:  # 加载预训练模型
        logger.info('')
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    # assert model.config.vocab_size == tokenizer.vocab_size

    # 多卡并行训练模型
    if args.cuda and torch.cuda.device_count() > 1:
        # model = DataParallel(model).cuda()
        model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info("Number of teacher parameter: %.2fM" % (num_parameters / 1e6))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset = load_dataset(logger, args)

    train(model, logger, train_dataset, writer, args)


if __name__ == '__main__':
    main()
