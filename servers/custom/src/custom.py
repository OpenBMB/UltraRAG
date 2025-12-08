import re
from typing import List, Dict, Any

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("custom")


@app.tool(output="ans_ls->extract_query_list")
def search_r1_query_extract(ans_ls: List[str]) -> Dict[str, List[str]]:

    def get_query(text):
        import re

        pattern = re.compile(r"<search>([^<]*)", re.DOTALL)
        matches = pattern.findall(text)

        if matches:
            query = matches[-1].strip()
            if not query.endswith("?"):
                query += "?"
            return query
        else:
            return "There is no query."

    query = [get_query(answer) for answer in ans_ls]

    return {"extract_query_list": query}


@app.tool(output="ans_ls->extract_query_list")
def r1_searcher_query_extract(ans_ls: List[str]) -> Dict[str, List[str]]:

    def get_query(text):
        import re

        pattern = re.compile(r"<|begin_of_query|>([^<]*)", re.DOTALL)
        matches = pattern.findall(text)

        if matches:
            query = matches[-1].strip()
            if not query.endswith("?"):
                query += "?"
            return query
        else:
            return "There is no query."

    query = [get_query(answer) for answer in ans_ls]

    return {"extract_query_list": query}


@app.tool(output="q_ls,ret_psg->nextq_ls")
def iterretgen_nextquery(
    q_ls: List[str],
    ans_ls: List[str | Any],
) -> Dict[str, List[str]]:
    ret = []
    for q, ans in zip(q_ls, ans_ls):
        next_query = f"{q} {ans}"
        ret.append(next_query)
    return {"nextq_ls": ret}


@app.tool(output="ans_ls->pred_ls")
def output_extract_from_boxed(ans_ls: List[str]) -> Dict[str, List[str]]:
    def extract(ans: str) -> str:
        start = ans.rfind(r"\boxed{")
        if start == -1:
            content = ans.strip()
        else:
            i = start + len(r"\boxed{")
            brace_level = 1
            end = i
            while end < len(ans) and brace_level > 0:
                if ans[end] == "{":
                    brace_level += 1
                elif ans[end] == "}":
                    brace_level -= 1
                end += 1
            content = ans[i : end - 1].strip()
            content = re.sub(r"^\$+|\$+$", "", content).strip()
            content = re.sub(r"^\\\(|\\\)$", "", content).strip()
            if content.startswith(r"\text{") and content.endswith("}"):
                content = content[len(r"\text{") : -1].strip()
            content = content.strip("()").strip()

        content = content.replace("\\", " ")
        content = content.replace("  ", " ")
        return content

    return {"pred_ls": [extract(ans) for ans in ans_ls]}


@app.tool(output="ans_ls->q_ls")
def ircot_get_first_sent(
    ans_ls: List[str],
) -> Dict[str, List[str]]:
    ret = []
    for ans in ans_ls:
        match = re.search(r"(.+?[。！？.!?])", ans)
        if match:
            ret.append(match.group(1))
        else:
            ret.append(ans.strip())
    return {"q_ls": ret}


@app.tool(output="ans_ls->pred_ls")
def ircot_extract_ans(ans_ls: List[str]) -> Dict[str, List[str]]:
    ret = []
    pattern = re.compile(r"so the answer is[\s:]*([^\n]*)", re.IGNORECASE)
    for ans in ans_ls:
        match = pattern.search(ans)
        if match:
            ret.append(match.group(1).strip())
        else:
            ret.append(ans.strip())
    return {"pred_ls": ret}


@app.tool(output="q_ls->total_subq_list,total_reason_list,total_final_info_list")
def search_o1_init_list(q_ls: List[str]) -> Dict[str, List[Any]]:
    n = len(q_ls)

    return {
        "total_subq_list": [["<PAD>"] for _ in range(n)],
        "total_reason_list": [["<PAD>"] for _ in range(n)],
        "total_final_info_list": [["<PAD>"] for _ in range(n)],
    }

@app.tool(
    output="total_subq_list, extract_query_list, total_reason_list, extract_reason_list"
           "->total_subq_list, total_reason_list"
)
def search_o1_combine_list(
    total_subq_list: List[List[Any]],
    extract_query_list: List[str],
    total_reason_list: List[List[Any]],
    extract_reason_list: List[str],
) -> Dict[str, List[Any]]:
    
    PAD = "<PAD>"

    for q, bucket in zip(extract_query_list, total_subq_list):
        if len(bucket) == 1 and bucket[0] == PAD:
            bucket[0] = q            
        else:
            bucket.append(q)

    for c, bucket in zip(extract_reason_list, total_reason_list):
        if len(bucket) == 1 and bucket[0] == PAD:
            bucket[0] = c           
        else:
            bucket.append(c)

    return {
        "total_subq_list": total_subq_list,
        "total_reason_list": total_reason_list,
    }

@app.tool(output="ans_ls->extract_query_list")
def search_o1_query_extract(ans_ls: List[str]) -> Dict[str, List[str]]:
    import re

    BEGIN = "<|begin_search_query|>"
    END = "<|end_search_query|>"
    PATTERN = re.escape(BEGIN) + r"(.*?)" + re.escape(END)

    def get_query(text):
        matches = re.findall(PATTERN, text, flags=re.DOTALL)
        if not matches:
            return ""  
        q = matches[-1].strip()
        q = re.sub(r"\s+", " ", q).strip(' "\'')
        return q

    query = [get_query(answer) for answer in ans_ls]

    return {"extract_query_list": query}

@app.tool(output="ans_ls->extract_reason_list")
def search_o1_reasoning_extract(ans_ls: List[str]) -> Dict[str, List[str]]:

    BEGIN = "<|begin_search_query|>"

    def get_content_before(text):
        if BEGIN not in text:
            return text.strip()
        

        return text.split(BEGIN, 1)[0].strip()

    content_list = [get_content_before(answer) for answer in ans_ls]

    return {"extract_reason_list": content_list}

@app.tool(output="ans_ls->extract_final_infor_list")
def search_o1_extract_final_information(ans_ls: List[str]) -> Dict[str, List[str]]:

    BEGIN = "**Final Information**"

    def get_content_after(text):
        if BEGIN not in text:
            return ""
    
        return BEGIN + "\n" + text.split(BEGIN, 1)[1].strip()

    content_list = [get_content_after(answer) for answer in ans_ls]

    return {"extract_final_infor_list": content_list}

@app.tool(output="total_final_info_list, extract_final_infor_list->total_final_info_list")
def search_o1_combine_final_information(
    total_final_info_list: List[List[str]],
    extract_final_infor_list: List[str],
) -> Dict[str, List[Any]]:
    
    PAD = "<PAD>"

    for c, bucket in zip(extract_final_infor_list, total_final_info_list):
        if len(bucket) == 1 and bucket[0] == PAD:
            bucket[0] = c           
        else:
            bucket.append(c)

    app.logger.warning(f"len total_final_info_list: {len(total_final_info_list)}")
    app.logger.warning(f"total_final_info_list: {total_final_info_list}")

    return {
        "total_final_info_list": total_final_info_list,
    }

@app.tool(output="temp_psg,ret_psg->ret_psg")
def merge_passages(
    temp_psg: List[str | Any],
    ret_psg: List[str | Any],
) -> Dict[str, List[str | Any]]:
    for t_psg, psg in zip(temp_psg, ret_psg):
        psg.extend(t_psg)

    return {"ret_psg": ret_psg}


@app.tool(output="ans_ls->pred_ls")
def evisrag_output_extract_from_special(ans_ls: List[str]) -> Dict[str, List[str]]:
    def extract(ans: str) -> str:
        try:
            content = ans.split('<answer>')[1].split('</answer>')[0].strip()
        except:
            content = ans.strip()
        return content

    return {"pred_ls": [extract(ans) for ans in ans_ls]}


if __name__ == "__main__":
    app.run(transport="stdio")
