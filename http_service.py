import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel, CpmTokenizer
from utils import top_k_top_p_filtering, set_logger
from os.path import join
from flask import Flask, redirect, url_for, request
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码


def generate_next_token(input_ids):
    """
    对于给定的上文，生成下一个单词
    """
    # 只根据当前位置的前context_len个token进行生成
    input_ids = input_ids[:, -args.context_len:]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[0, -1, :]
    next_token_logits = next_token_logits / args.temperature
    # 对于<unk>的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    next_token_logits[unk_id] = -float('Inf')
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
    # torch.multinomial表示从候选集合中选出无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
    next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    return next_token_id


@app.route('/zuowen', methods=['POST', 'GET'])
def zuowen():
    if request.method == 'POST':
        data = request.get_json()
        title = data['title']
        context = data['context']
        max_len = data['max_len']
    elif request.method == 'GET':
        title = request.args.get('title', type=str)
        context = request.args.get('context', type=str)
        max_len = request.args.get('max_len', type=int)

    # print("title:{}".format(title))
    # print("context:{}".format(context))
    logger.info("receive request,title:{}, context:{}".format(title, context))

    title_ids = tokenizer.encode(title, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    input_ids = title_ids + [sep_id] + context_ids
    cur_len = len(input_ids)
    last_token_id = input_ids[-1]  # 已生成的内容的最后一个token
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    while True:
        next_token_id = generate_next_token(input_ids)
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
        cur_len += 1
        word = tokenizer.convert_ids_to_tokens(next_token_id.item())
        # 超过最大长度，并且换行
        if cur_len >= max_len and last_token_id == 8 and next_token_id == 3:
            break
        # 超过最大长度，并且生成标点符号
        if cur_len >= max_len and word in [".", "。", "！", "!", "?", "？", ",", "，"]:
            break
        # 生成结束符
        if next_token_id == eod_id:
            break
    result = tokenizer.decode(input_ids.squeeze(0))
    content = result.split("<sep>")[1]  # 生成的最终内容
    result = {"title": title, "content": content}
    logger.info("generated result:{}".format(result))
    return result


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=0, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.85, type=float, required=False, help='最高积累概率')
    parser.add_argument('--context_len', default=200, type=int, required=False, help='作文生成中，每一步生成时，参考的上文的长度')
    # parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='重复惩罚参数')
    parser.add_argument('--port', type=int, default=8085, help='服务绑定的端口号')
    parser.add_argument('--log_path', default='log/http_service.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--model_path', type=str, default='model/zuowen_epoch40', help='模型存放位置')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    args.cuda = torch.cuda.is_available() and not args.no_cuda  # 当用户使用GPU,并且GPU可用时
    device = 'cuda:0' if args.cuda else 'cpu'
    # device = 'cpu'

    # 创建日志对象
    logger = set_logger(args.log_path)

    # 加载tokenizer
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id

    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.eval()
    model = model.to(device)

    app.run(debug=True, host="0.0.0.0", port=args.port)
