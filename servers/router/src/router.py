import re
from typing import Any, Dict, List, Optional, Union

from ultrarag.server import UltraRAG_MCP_Server


app = UltraRAG_MCP_Server("router")


def _check_eos_token(text: str, tokens: List[str]) -> bool:
    """Check if text contains any end-of-sequence tokens.

    Args:
        text: Text to check
        tokens: List of EOS tokens to look for

    Returns:
        True if any token is found, False otherwise
    """
    return any(token in text for token in tokens)


@app.tool(output="query_list")
def route1(query_list: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Route queries to state1 or state2 based on query value.

    Args:
        query_list: List of query strings

    Returns:
        Dictionary with 'query_list' containing queries with routing states

    Raises:
        ValueError: If query cannot be converted to integer
    """
    query = []
    for q in query_list:
        try:
            state = "state1" if int(q) == 1 else "state2"
        except ValueError:
            state = "state2"
        query.append({"data": q, "state": state})
    return {"query_list": query}


@app.tool(output="query_list")
def route2(query_list: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Route all queries to state2.

    Args:
        query_list: List of query strings

    Returns:
        Dictionary with 'query_list' containing queries with state2
    """
    query = [{"data": query, "state": "state2"} for query in query_list]
    return {"query_list": query}


@app.tool(output="ans_ls->ans_ls")
def ircot_check_end(ans_ls: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Check if IRCoT answers are complete based on completion phrase.

    Args:
        ans_ls: List of answer strings

    Returns:
        Dictionary with 'ans_ls' containing answers with completion states
    """
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
    """Check if search-r1 answers are complete based on EOS tokens.

    Args:
        ans_ls: List of answer strings to check

    Returns:
        Dictionary with 'ans_ls' containing answers with completion states
    """
    eos_tokens = ["<|endoftext|>", "<|im_end|>"]

    ans_ls = [
        {
            "data": answer,
            "state": (
                "complete" if _check_eos_token(answer, eos_tokens) else "incomplete"
            ),
        }
        for answer in ans_ls
    ]
    return {"ans_ls": ans_ls}


@app.tool(output="page_ls->page_ls")
def webnote_check_page(page_ls: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Check if WebNote pages are complete or incomplete.

    Args:
        page_ls: List of page strings to check

    Returns:
        Dictionary with 'page_ls' containing pages with completion states
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
    """Check if r1_searcher answers are complete based on EOS tokens.

    Args:
        ans_ls: List of answer strings to check

    Returns:
        Dictionary with 'ans_ls' containing answers with completion states
    """
    eos_tokens = ["<|endoftext|>", "<|im_end|>", "</answer>"]

    ans_ls = [
        {
            "data": answer,
            "state": (
                "complete" if _check_eos_token(answer, eos_tokens) else "incomplete"
            ),
        }
        for answer in ans_ls
    ]
    return {"ans_ls": ans_ls}


@app.tool(
    output=(
        "ans_ls,q_ls,total_subq_list,total_reason_list,total_final_info_list->"
        "ans_ls,q_ls,total_subq_list,total_reason_list,total_final_info_list"
    )
)
def search_o1_check(
    ans_ls: List[str],
    q_ls: List[str],
    total_subq_list: List[List[Any]],
    total_reason_list: List[List[Any]],
    total_final_info_list: List[List[Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Check if Search O1 should stop or continue retrieving.

    Args:
        ans_ls: List of answer strings
        q_ls: List of query strings
        total_subq_list: List of sub-question lists
        total_reason_list: List of reasoning lists
        total_final_info_list: List of final info lists

    Returns:
        Dictionary with all input lists wrapped with routing states
    """
    ans_out: List[Dict[str, Any]] = []
    q_out: List[Dict[str, Any]] = []
    subq_out: List[Dict[str, Any]] = []
    reason_out: List[Dict[str, Any]] = []
    info_out: List[Dict[str, Any]] = []

    for ans, q, subq, reason, info in zip(
        ans_ls, q_ls, total_subq_list, total_reason_list, total_final_info_list
    ):
        # Stop if answer contains end token, continue if it contains search query marker
        if "<|end_search_query|>" in ans:
            state = "retrieve"
        elif "<|im_end|>" in ans:
            state = "stop"
        else:
            state = "stop"

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
    """Check if model should continue or stop based on search token.

    Args:
        ans_ls: List of answer strings

    Returns:
        Dictionary with 'ans_ls' containing answers with continue/stop states
    """
    ans_ls = [
        {
            "data": answer,
            "state": "continue" if "<search>" in answer else "stop",
        }
        for answer in ans_ls
    ]
    return {"ans_ls": ans_ls}


@app.tool(
    output=(
        "state_ls,cursor_ls,survey_ls,step_ls,extend_time_ls,extend_result_ls->"
        "state_ls,cursor_ls,survey_ls,step_ls,extend_time_ls,extend_result_ls"
    )
)
def surveycpm_state_router(
    state_ls: List[str],
    cursor_ls: List[Optional[str]],
    survey_ls: List[str],
    step_ls: List[int],
    extend_time_ls: List[int],
    extend_result_ls: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Route SurveyCPM state variables based on current state.

    Args:
        state_ls: List of state strings
        cursor_ls: List of cursor positions (may be None)
        survey_ls: List of survey JSON strings
        step_ls: List of step numbers
        extend_time_ls: List of extend time values
        extend_result_ls: List of extend result strings

    Returns:
        Dictionary with all input lists wrapped with routing states
    """
    routed_state_ls = []
    routed_cursor_ls = []
    routed_survey_ls = []
    routed_step_ls = []
    routed_extend_time_ls = []
    routed_extend_result_ls = []

    for state, cursor, survey, step, extend_time, extend_result in zip(
        state_ls, cursor_ls, survey_ls, step_ls, extend_time_ls, extend_result_ls
    ):
        routed_state_ls.append({"data": state, "state": state})
        routed_cursor_ls.append({"data": cursor, "state": state})
        routed_survey_ls.append({"data": survey, "state": state})
        routed_step_ls.append({"data": step, "state": state})
        routed_extend_time_ls.append({"data": extend_time, "state": state})
        routed_extend_result_ls.append({"data": extend_result, "state": state})

    return {
        "state_ls": routed_state_ls,
        "cursor_ls": routed_cursor_ls,
        "survey_ls": routed_survey_ls,
        "step_ls": routed_step_ls,
        "extend_time_ls": routed_extend_time_ls,
        "extend_result_ls": routed_extend_result_ls,
    }


if __name__ == "__main__":
    app.run(transport="stdio")
