import argparse
from loguru import logger

import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import GPT2LMHeadModel, GPT2Config
import os
from os.path import join
import random
import pickle
import time
import torch.nn.utils.rnn as rnn_utils
import transformers
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import CPMDataset
"""
模型蒸馏
"""


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=5)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def seed_everything(seed=42):
    """
    设置整个开发环境的seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def distill_loss(logits, labels, target_logits, ignore_index):
    # hard loss
    hard_loss = hard_cross_entropy_loss(logits, labels, ignore_index)
    # soft loss
    soft_loss = soft_cross_entropy_loss(logits, labels, target_logits, ignore_index)
    # 加权
    loss = 0.5 * hard_loss + 0.5 * soft_loss
    return loss


def hard_cross_entropy_loss(logits, labels, ignore_index):
    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    return loss


def soft_cross_entropy_loss(logits, labels, target_logits, ignore_index):
    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)
    target_probs = torch.softmax(target_logits, axis=-1)
    target_probs = target_probs[..., :-1, :].contiguous().view(-1, target_probs.size(-1))

    # 计算每个位置的loss
    loss = F.cross_entropy(logits, target_probs, reduction='none')

    # 选出非padding的loss，求平均
    loss_mask = (labels == ignore_index)
    loss = torch.masked_select(loss, ~loss_mask)
    loss = loss.mean()

    return loss


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


def train(teacher, student, train_dataset, writer, args):
    teacher.eval()
    student.train()
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(student.parameters(), lr=args.lr, eps=args.eps)
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
            with torch.no_grad():
                target_logits = teacher(input_ids=input_ids).logits
            logits = student(input_ids=input_ids).logits

            # 计算loss
            loss = soft_cross_entropy_loss(logits, labels, target_logits, args.ignore_index)
            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            batch_acc = batch_correct_num/batch_total_num
            train_loss += loss
            train_acc += batch_acc

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
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
                logger.info('Epoch {} step {} train Loss {:.4f}, train ACC {:.4f}'.format(epoch + 1, step, train_loss, train_acc))
                writer.add_scalar('train loss', train_loss, step)
                writer.add_scalar('train acc', train_acc, step)
                train_loss = 0
                train_acc = 0

            if step % save_step == 0:
                logger.info('Saving model at Epoch {} step {}'.format(epoch + 1, step))
                model_path = join(args.output_path, 'epoch_{}-step_{}'.format(epoch + 1, step))
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_to_save = student.module if hasattr(student, 'module') else student
                model_to_save.save_pretrained(model_path)

    logger.info('training finished')


def main():
    # 参数设置
    args = set_args()
    # 设置随机种子
    seed_everything(args.seed)
    # 设置显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 创建输出目录
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # 日志输出位置
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'distill-{}.log'.format(cur_time)))
    # 初始化tensorboard
    writer = SummaryWriter(args.output_path)
    # 加载tokenizer
    # tokenizer = CpmTokenizer(vocab_file=args.vocab_path)
    # args.eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    # args.pad_id = tokenizer.pad_token_id
    # 加载teacher模型
    teacher = GPT2LMHeadModel.from_pretrained(args.teacher_checkpoint)
    teacher = teacher.to(args.device)
    # 初始化student模型
    student_config = GPT2Config.from_pretrained(args.student_config_path)
    student = GPT2LMHeadModel(student_config)
    student = student.to(args.device)
    logger.info('student model config:{}'.format(student_config))

    # 计算模型参数量
    params_teacher = sum([param.nelement() for param in teacher.parameters()])
    logger.info("Number of teacher parameter: %.2fM" % (params_teacher / 1e6))
    params_student = sum([param.nelement() for param in student.parameters()])
    logger.info("Number of student parameter: %.2fM" % (params_student / 1e6))
    # 记录参数设置
    logger.info(args)

    # 加载训练集
    train_dataset = load_dataset(logger, args)
    train(teacher, student, train_dataset, writer, args)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", type=str, default='3', help="")
    parser.add_argument("--output_path", type=str, default='output/distill')
    parser.add_argument('--vocab_path', default='vocab/chinese_vocab.model', type=str, required=False,
                        help='sp模型路径')
    parser.add_argument("--teacher_checkpoint", type=str, default="model/zuowen_epoch40", help='teacher模型的路径')
    parser.add_argument("--student_config_path", type=str, default="config/cpm-one-layer.json", help='student模型的配置')
    parser.add_argument('--train_path', default='data/train.pkl', type=str, required=False, help='经过预处理之后的数据存放路径')
    parser.add_argument('--max_len', default=200, type=int, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False,
                        help='对于ignore_index的label token不计算梯度')

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False, help='梯度积累的步数')

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_step", type=int, default=100, help="every eval_step to save model")
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()


