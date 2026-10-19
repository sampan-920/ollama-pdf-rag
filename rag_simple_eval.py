import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen:4b"

def ollama_ask(prompt: str):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    return resp.json()["response"]

# 评估数据集
eval_data = [
    {
        "question": "RAG的完整流程是什么？",
        "retrieved_context": ["RAG分为文档加载、文本切分、向量化存入向量库；用户提问，检索相关片段，把上下文+问题交给大模型生成答案。"],
        "model_answer": "RAG分为文档加载、文本切分、向量化存入向量库；用户提问，检索相关片段，把上下文+问题交给大模型生成答案。",
        "ground_truth": "RAG包含文档解析、分段、向量化存储、相似度检索、prompt拼接、大模型生成回答。"
    },
    {
        "question": "向量数据库的作用？",
        "retrieved_context": ["向量数据库用来存储文本向量，根据用户问题做相似度检索，召回相关文档片段。"],
        "model_answer": "向量数据库用来存储文本向量，根据用户问题做相似度检索，召回相关文档片段。",
        "ground_truth": "向量数据库保存文本向量，实现语义检索，找到和用户问题相关的文档上下文。"
    }
]

def calc_context_precision(question, contexts):
    prompt = f"""
任务：判断文档片段是否和用户问题相关。只输出0~1的小数，1代表完全相关，0代表无关。
问题：{question}
文档片段：{contexts[0]}
仅输出数字：
"""
    res = ollama_ask(prompt).strip()
    try:
        return float(res)
    except:  # noqa: E722
        return 0.0

def calc_context_recall(ground_truth, contexts):
    prompt = f"""
任务：判断文档片段能否覆盖标准答案的信息。输出0~1小数，1代表完全覆盖，0完全没覆盖。
标准答案：{ground_truth}
文档片段：{contexts[0]}
仅输出数字：
"""
    res = ollama_ask(prompt).strip()
    try:
        return float(res)
    except:  # noqa: E722
        return 0.0

def calc_faithfulness(answer, contexts):
    prompt = f"""
任务：判断模型回答是否全部来源于提供的上下文，有没有编造不存在的信息。
规则：模型回答所有句子，如果都能在上下文找到依据 → 高分；如果有编造内容则低分。输出0~1之间小数。
上下文：{contexts[0]}
模型回答：{answer}
仅输出数字：
"""
    res = ollama_ask(prompt).strip()
    try:
        return float(res)
    except:  # noqa: E722
        return 0.0

def calc_answer_relevancy(question, answer):
    prompt = f"""
任务：判断模型回答与用户问题的相关程度，输出0~1小数，越高越相关。
问题：{question}
模型回答：{answer}
仅输出数字：
"""
    res = ollama_ask(prompt).strip()
    try:
        return float(res)
    except:  # noqa: E722
        return 0.0

# 执行评估
result_list = []
for item in eval_data:
    cp = calc_context_precision(item["question"], item["retrieved_context"])
    cr = calc_context_recall(item["question"], item["retrieved_context"])
    fth = calc_faithfulness(item["model_answer"], item["retrieved_context"])
    ar = calc_answer_relevancy(item["question"], item["model_answer"])
    row = {
        "question": item["question"],
        "context_precision": cp,
        "context_recall": cr,
        "faithfulness": fth,
        "answer_relevancy": ar
    }
    result_list.append(row)

df = pd.DataFrame(result_list)
print("==== 简易RAG评估结果（对标RAGAS四大指标） ====")
print(df)
print("\n==== 各项指标平均值 ====")
print(df[["context_precision","context_recall","faithfulness","answer_relevancy"]].mean())
