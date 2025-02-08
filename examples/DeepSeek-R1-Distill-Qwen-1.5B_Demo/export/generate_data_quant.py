from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-dir", help="Path to model directory cloned from HF Hub")
ap.add_argument("-o", "--output-file", help="Path to save calibration file", default='./data_quant.json')
ap.add_argument('--apply_chat_template', help="Whether to use the chat template or not.", default=True)
ap.add_argument('--max_new_tokens', default=128)
ap.add_argument('--temperature', default=0.6)
ap.add_argument('--repetition_penalty', default=1.1)
args = ap.parse_args()


if __name__ == '__main__':
    
    if not torch.cuda.is_available():
        dev = 'cpu'
    else:            
        dev = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)
    model = model.to(dev)
    model = model.eval()
    
    ## 请根据模型特点与使用场景准备用于量化的校准样本。
    input_text = [
        '在农业生产中被当作极其重要的劳动对象发挥作用，最主要的不可替代的基本生产资料是\nA. 农业生产工具\nB. 土地\nC. 劳动力\nD. 资金',
        '下列行为如满足规定条件，应认定为伪造货币罪的是\nA. 将英镑揭层一分为二\nB. 铸造珍稀古钱币\nC. 临摹欧元收藏\nD. 用黄金铸造流通的纪念金币',
        '设是 $f(x)$ 偶函数， $\varphi(x)$ 是奇函数， 则下列函数(假设都有意义)中是奇函数的是 ( ).\nA. $f[f(x)]$\nB. $\varphi[\varphi(x)]$\nC. $\varphi[f(x)]$\nD. $f[\varphi(x)]$',
        'def fizz_buzz(n: int): """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13. >>> fizz_buzz(50) 0 >>> fizz_buzz(78) 2 >>> fizz_buzz(79) 3 """'
        '已知 ${ }_s p_{10}=0.4$， 且 $\mu_x=0.01+b x， x \geqslant 0$， 则 $b$ 等于 $(\quad$ 。\nA. -0.05\nB. -0.014\nC. -0.005\nD. 0.014',
        "Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?",
        "你是一个手机内负责日程管理的智能助手，你要基于用户给定的目标日程信息，综合考虑日程库的行程，联想一些可以体现人文关怀，实用，并且可以给用户带来惊喜的子日程提醒。",
        "给定以下Python代码，请改写它，使用列表解析来完成相同的功能。\n\nsquares = []\n\nfor i in range(10):     \n\n    squares.append(i**2)\n\nprint(squares)",
        "Some people got on a bus at the terminal. At the first bus stop, half of the people got down and 4 more people got in. Then at the second bus stop, 6 people got down and 8 more got in. If there were a total of 25 people heading to the third stop, how many people got on the bus at the terminal?",
        "下列句子中存在歧义的一句是（）A：上级要求我们按时完成任务B：老师满意地朝他看了一眼C：我看见你那年才十岁D：她的一句话说得大家都笑了",
        "What is the coefficient of $x^2y^6$ in the expansion of $\left(\frac{3}{5}x-\frac{y}{2}\right)^8$? Express your answer as a common fraction.",
        "I love Woudenberg Koffiemok Trots op Goedkoop I love Woudenberg Koffiemok Trots op Een te gekke koffiemok die je niet mag missen als je in Woudenberg woont. Productcode: 29979 - bbweb",
        "Aussie Speedo Guy is a Bisexual Aussie Guy who loves speedos. » Indoor Pool TwinksAussieSpeedoGuy.org: AussieSpeedoGuy.org Members Blog No User Responded in \" Indoor Pool Twinks \"",
        "在城市夜晚的霓虹灯下，车水马龙，您能为此创作七言绝句吗？关键词：夜晚，霓虹。",
        "以下是关于经济学的单项选择题，请从A、B、C、D中选择正确答案对应的选项。\n题目：当长期均衡时，完全竞争企业总是\nA. 经济利润大于零\nB. 正常利润为零\nC. 经济利润小于零\nD. 经济利润为零\n答案是: ",
        '下列句中，“是”充当前置宾语的一句是\nA. 如有不由此者，在執者去，衆以焉殃，是謂小康。\nB. 子曰：敏而好學，不下恥問，是以謂之文也。\nC. 是乃其所以千萬臣而無數者也。\nD. 鑄名器，藏寶財，固民之殄病是待。',
        'def is_multiply_prime(a): """Write a function that returns true if the given number is the multiplication of 3 prime numbers and false otherwise. Knowing that (a) is less then 100. Example: is_multiply_prime(30) == True 30 = 2 * 3 * 5 """',
        "What is the theory of general relativity?\n General relativity is a theory of gravitation developed by Albert Einstein. It describes gravity not as a force, but as a curvature of spacetime caused by mass and energy.",
        "Human: 请提取以下句子中的关健词信息，用JSON返回：句子：'我现在想感受一下不同的文化背景，看一部外国喜剧电影。'。",
        '《坛经》，是历史上除佛经外，唯一被尊称为“经”的佛教典籍。此书作者是\nA. 六祖慧能\nB. 南天竺菩提达摩\nC. 释迦牟尼\nD. 五祖宏忍',
        "把这句话翻译成英文: RK3588是新一代高端处理器，具有高算力、低功耗、超强多媒体、丰富数据接口等特点",
        "把这句话翻译成中文: Knowledge can be acquired from many sources. These include books, teachers and practical experience, and each has its own advantages. The knowledge we gain from books and formal education enables us to learn about things that we have no opportunity to experience in daily life. We can also develop our analytical skills and learn how to view and interpret the world around us in different ways."
    ]
    
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "top_k":1, "temperature": args.temperature, "do_sample": True, "repetition_penalty": args.repetition_penalty}
    calidata = []

    for idx, inp in enumerate(input_text):
        question = inp.strip()
        if args.apply_chat_template:
            datas = [{"role":"user","content":question}]
            messages = tokenizer.apply_chat_template(datas, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        else:
            messages = f"{question}"
        
        try:
            inputs = tokenizer(messages, return_tensors='pt').to(dev)
            outputs = model.generate(**inputs, **gen_kwargs)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(result, len(result))
            calidata.append({"input":messages, "target":result[len(messages):]})
        except:
            calidata.append({"input":messages, "target":''})

    
    with open(args.output_file, 'w') as f:
        json.dump(calidata, f)
