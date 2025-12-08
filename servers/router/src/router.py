from typing import List, Dict, Any
from ultrarag.server import UltraRAG_MCP_Server


app = UltraRAG_MCP_Server("router")


@app.tool(output="query_list")
def route1(query_list: List[str]) -> Dict[str, List[Dict[str, str]]]:
    query = [
        {"data": query, "state": "state1" if int(query) == 1 else "state2"}
        for query in query_list
    ]
    return {"query_list": query}


@app.tool(output="query_list")
def route2(query_list: List[str]) -> Dict[str, List[Dict[str, str]]]:
    query = [{"data": query, "state": "state2"} for query in query_list]
    return {"query_list": query}


@app.tool(output="ans_ls->ans_ls")
def ircot_check_end(ans_ls: List[str]) -> Dict[str, List[Dict[str, str]]]:
    ans_ls = [
        {
            "data": ans,
            "state": "complete" if "so the answer is" in ans.lower() else "incomplete",
        }
        for ans in ans_ls
    ]
    return {"ans_ls": ans_ls}


@app.tool(output="ans_ls->ans_ls")
def search_r1_check(ans_ls: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Check if the answer is complete or incomplete.
    Args:
        ans_ls (list): List of answers to check.
    Returns:
        dict: Dictionary containing the list of answers with their states.
    """

    def get_eos(text):
        import re

        if "<|endoftext|>" in text or "<|im_end|>" in text:
            return True
        else:
            return False

    ans_ls = [
        {
            "data": answer,
            "state": "complete" if get_eos(answer) else "incomplete",
        }
        for answer in ans_ls
    ]
    return {"ans_ls": ans_ls}


@app.tool(output="page_ls->page_ls")
def webnote_check_page(page_ls: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Check if the page is complete or incomplete.
    Args:
        page_ls (list): List of pages to check.
    Returns:
        dict: Dictionary containing the list of pages with their states.
    """
    page_ls = [
        {
            "data": page,
            "state": "incomplete" if "to be filled" in page.lower() else "complete",
        }
        for page in page_ls
    ]
    return {"page_ls": page_ls}


@app.tool(output="ans_ls->ans_ls")
def r1_searcher_check(ans_ls: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Check if the answer is complete or incomplete.
    Args:
        ans_ls (list): List of answers to check.
    Returns:
        dict: Dictionary containing the list of answers with their states.
    """

    def get_eos(text):
        import re

        if "<|endoftext|>" in text or "<|im_end|>" in text or "</answer>" in text:
            return True
        else:
            return False

    ans_ls = [
        {
            "data": answer,
            "state": "complete" if get_eos(answer) else "incomplete",
        }
        for answer in ans_ls
    ]
    return {"ans_ls": ans_ls}


@app.tool(
    output=("ans_ls,q_ls,total_subq_list,total_reason_list,total_final_info_list->ans_ls,q_ls,total_subq_list,total_reason_list,total_final_info_list")
)
def search_o1_check(
    ans_ls: List[str],
    q_ls: List[str],
    total_subq_list: List[List[Any]],
    total_reason_list: List[List[Any]],
    total_final_info_list: List[List[Any]],
) -> Dict[str, List[Dict[str, Any]]]:

    def get_eos(text: str) -> bool:
        if "<|im_end|>" in text:
            return True
        elif "<|end_search_query|>" in text:
            return False
        else:
            return True

    ans_out: List[Dict[str, Any]] = []
    q_out: List[Dict[str, Any]] = []
    subq_out: List[Dict[str, Any]] = []
    reason_out: List[Dict[str, Any]] = []
    info_out: List[Dict[str, Any]] = []

    for ans, q, subq, reason, info in zip(
        ans_ls, q_ls, total_subq_list, total_reason_list, total_final_info_list
    ):
        state = "stop" if get_eos(ans) else "retrieve"

        ans_out.append({"data": ans, "state": state})
        q_out.append({"data": q, "state": state})
        subq_out.append({"data": subq, "state": state})
        reason_out.append({"data": reason, "state": state})
        info_out.append({"data": info, "state": state})

    return {
        "ans_ls": ans_out,
        "q_ls": q_out,
        "total_subq_list": subq_out,
        "total_reason_list": reason_out,
        "total_final_info_list": info_out,
    }


@app.tool(output="ans_ls->ans_ls")
def check_model_state(ans_ls: List[str]) -> Dict[str, List[Dict[str, str]]]:

    def check_state(text):
        if "<search>" in text:
            return True
        else:
            return False

    ans_ls = [
        {
            "data": answer,
            "state": "continue" if check_state(answer) else "stop",
        }
        for answer in ans_ls
    ]
    return {"ans_ls": ans_ls}


if __name__ == "__main__":
    app.run(transport="stdio")
