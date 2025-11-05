import argparse
import json
import os

def read_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass  
    return data


def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_data_path', type=str,
                        default=None,
                        )
    parser.add_argument('--question_key', type=str,
                        default='question',
                        )
    parser.add_argument('--result_path', type=str,
                        default=None,
                        )
    parser.add_argument('--save_data_path', type=str,
                        default=None,
                        )                
    args = parser.parse_args()
    origin_data = read_jsonl(args.origin_data_path)
    result_data = read_json(args.result_path)
    for tool in result_data:
        if tool['step'] == 'benchmark.get_data':
            re_q_ls = tool['memory']['memory_q_ls']
        elif tool['step'] == 'retriever.retriever_search':
            ret_psg = tool['memory']['memory_ret_psg']
    for data, re_q, re_ret_psg in zip(origin_data,re_q_ls, ret_psg):
        q_ls = data[args.question_key]
        assert q_ls == re_q, f"Question mismatch: {q_ls} != {re_q}"
        data['retrieved_passages'] = re_ret_psg
    
    save_jsonl(origin_data, args.save_data_path)
    print(f"✅ Saved to {args.save_data_path}")



if __name__ == "__main__":
    main()