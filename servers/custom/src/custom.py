import re
import json
import copy
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


@app.tool(output="ret_psg->ret_psg")
def assign_citation_ids(
    ret_psg: List[List[str]],
) -> Dict[str, Any]:
    result_psg = []
    
    for docs_list in ret_psg:
        cited_docs = []
        for idx, doc in enumerate(docs_list, start=1):
            doc_text = str(doc).strip()
            cited_docs.append(f"[{idx}] {doc_text}")
        result_psg.append(cited_docs)
    
    return {
        "ret_psg": result_psg,
    }


class CitationRegistry:
    _instances: Dict[int, Dict[str, Any]] = {}
    
    @classmethod
    def reset(cls):
        cls._instances = {}
    
    @classmethod
    def get_or_create(cls, query_index: int) -> Dict[str, Any]:
        if query_index not in cls._instances:
            cls._instances[query_index] = {
                "registry": {},
                "counter": 0
            }
        return cls._instances[query_index]
    
    @classmethod
    def assign_id(cls, query_index: int, doc_text: str) -> int:
        state = cls.get_or_create(query_index)
        doc_hash = doc_text.strip()
        
        if doc_hash in state["registry"]:
            return state["registry"][doc_hash]
        else:
            state["counter"] += 1
            state["registry"][doc_hash] = state["counter"]
            return state["counter"]


@app.tool(output="q_ls->q_ls")
def init_citation_registry(q_ls: List[str]) -> Dict[str, Any]:
    CitationRegistry.reset()
    return {"q_ls": q_ls}


@app.tool(output="ret_psg->ret_psg")
def assign_citation_ids_stateful(
    ret_psg: List[List[str]],
) -> Dict[str, Any]:
    result_psg = []
    
    for i, docs_list in enumerate(ret_psg):
        cited_docs = []
        for doc in docs_list:
            doc_text = str(doc).strip()
            doc_id = CitationRegistry.assign_id(i, doc_text)
            cited_docs.append(f"[{doc_id}] {doc_text}")
        result_psg.append(cited_docs)
    
    return {
        "ret_psg": result_psg,
    }


# ==================== SurveyCPM Citation Tools ====================


class SurveyCPMCitationRegistry:
    """Citation registry for SurveyCPM pipeline.
    
    Maintains unique citation IDs across multiple search rounds for each query.
    """
    _instances: Dict[int, Dict[str, Any]] = {}
    
    @classmethod
    def reset(cls):
        cls._instances = {}
    
    @classmethod
    def get_or_create(cls, query_index: int) -> Dict[str, Any]:
        if query_index not in cls._instances:
            cls._instances[query_index] = {
                "registry": {},
                "counter": 0
            }
        return cls._instances[query_index]
    
    @classmethod
    def assign_id(cls, query_index: int, doc_text: str) -> str:
        """Assign a unique ID to a document, deduplicating across rounds."""
        state = cls.get_or_create(query_index)
        doc_hash = doc_text.strip()
        
        if doc_hash in state["registry"]:
            return state["registry"][doc_hash]
        else:
            state["counter"] += 1
            state["registry"][doc_hash] = f'textid{str(state["counter"])}'
            return f'textid{str(state["counter"])}'


@app.tool(output="instruction_ls->instruction_ls")
def surveycpm_init_citation_registry(instruction_ls: List[str]) -> Dict[str, Any]:
    """Initialize citation registry for SurveyCPM pipeline."""
    SurveyCPMCitationRegistry.reset()
    return {"instruction_ls": instruction_ls}


@app.tool(output="ret_psg_ls,survey_ls->retrieved_info_ls,ret_psg")
def surveycpm_process_passages_with_citation(
    ret_psg_ls: List[List[List[str]]],
    survey_ls: List[str],
) -> Dict[str, Any]:
    """Process passages and assign unique citation IDs for SurveyCPM.
    
    Similar to surveycpm_process_passages, but adds [id] prefix to each passage
    for citation tracking.
    
    Returns:
        - retrieved_info_ls: List of formatted strings for prompts
        - ret_psg: List of lists of cited passages for frontend rendering
                   (same length as ret_psg_ls, each element is a list of cited docs)
    """
    retrieved_info_ls = []
    ret_psg_output = []  # List of lists for frontend sources (same length as input)
    
    for query_idx, (ret_psg, survey_json) in enumerate(zip(ret_psg_ls, survey_ls)):
        survey = json.loads(survey_json) if survey_json and survey_json != "<PAD>" else {}
        top_k = 10 if survey else 20

        if not ret_psg:
            retrieved_info_ls.append("")
            ret_psg_output.append([])
            continue
        
        num_queries = len(ret_psg)
        per_query_limit = max(1, top_k // num_queries)
        
        seen = set()
        all_passages = []
        
        # First pass: get unique passages up to per_query_limit from each query
        for passages in ret_psg:
            for psg in passages[:per_query_limit]:
                if psg not in seen:
                    seen.add(psg)
                    # Assign unique citation ID
                    citation_id = SurveyCPMCitationRegistry.assign_id(query_idx, psg)
                    # cited_psg = f"[{citation_id}] {psg}"
                    cited_psg = f"bibkey:{citation_id}\n{psg}"
                    all_passages.append(cited_psg)
        
        # Second pass: fill remaining slots
        remaining_slots = top_k - len(all_passages)
        if remaining_slots > 0:
            for passages in ret_psg:
                for psg in passages[per_query_limit:]:
                    if psg not in seen and remaining_slots > 0:
                        seen.add(psg)
                        citation_id = SurveyCPMCitationRegistry.assign_id(query_idx, psg)
                        cited_psg = f"bibkey:{citation_id}\n{psg}"
                        all_passages.append(cited_psg)
                        remaining_slots -= 1
        
        info = "\n\n".join(all_passages).strip()
        retrieved_info_ls.append(info)
        ret_psg_output.append(all_passages)  # List of cited docs for this query
    
    return {
        "retrieved_info_ls": retrieved_info_ls,
        "ret_psg": ret_psg_output  # List of lists, same length as input
    }


# ==================== SurveyCPM Custom Tools ====================


def _surveycpm_abbr_one_line(string, abbr=True, tokenizer=None):
    """Abbreviate content to one line."""
    if isinstance(string, dict):
        if "content" in string and string["content"]:
            return _surveycpm_abbr_one_line(string["content"], abbr=abbr, tokenizer=tokenizer)
        elif "plan" in string:
            return "[PLAN] " + string["plan"].replace("\n", " ").strip()
        else:
            return ""
    else:
        if not string:
            return ""
        else:
            if abbr and tokenizer:
                tokens = tokenizer(string, return_tensors="pt")
                if tokens.input_ids.size(1) > 150:
                    decoded_prefix = tokenizer.decode(tokens.input_ids[0][:100], skip_special_tokens=True)
                    decoded_suffix = tokenizer.decode(tokens.input_ids[0][-50:], skip_special_tokens=True)
                    decoded = decoded_prefix + " ... " + decoded_suffix
                    return "[OK] " + decoded.replace("\n", " ").strip()
                else:
                    return "[OK] " + string.replace("\n", " ").strip()
            else:
                return "[OK] " + string.replace("\n", " ").strip()


def _surveycpm_to_one_line(string):
    """Convert content to one line."""
    if isinstance(string, dict):
        if "content" in string:
            if not string["content"]:
                return ""
            return "[OK] " + string["content"].replace("\n", " ").strip() + _surveycpm_to_one_line(string["content"])
        elif "plan" in string:
            return "[PLAN] " + string["plan"].replace("\n", " ").strip()
        else:
            return ""
    if not string:
        return ""
    else:
        return string.replace("\n", " ")


def _surveycpm_check_progress_postion(current_survey):
    """Check the current progress position in the survey."""
    if current_survey == {}:
        return "outline"
    else:
        if "sections" in current_survey:
            for i, section in enumerate(current_survey["sections"]):
                if "content" not in section:
                    return f"section-{i+1}"
                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        if "content" not in subsection:
                            return f"section-{i+1}.{j+1}"
                        if "subsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsections"]):
                                if "content" not in subsubsection:
                                    return f"section-{i+1}.{j+1}.{k+1}"
    return None


def _surveycpm_check_progress_postion_last_detail(current_survey):
    """Check the last completed position with detail."""
    if current_survey == {}:
        return "outline"
    else:
        titles = ["outline"]
        if "sections" in current_survey:
            for i, section in enumerate(current_survey["sections"]):
                if "content" not in section:
                    return titles[-1]
                else:
                    titles.append(f"section-{i+1}")
                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        if "content" not in subsection:
                            return titles[-1]
                        else:
                            titles.append(f"section-{i+1}.{j+1}")
                        if "subsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsections"]):
                                if "content" not in subsubsection:
                                    return titles[-1]
                                else:
                                    titles.append(f"section-{i+1}.{j+1}.{k+1}")
    return titles[-1]


def _surveycpm_print_tasknote_hire(current_survey, last_detail=False):
    """Print survey structure with hierarchical detail."""
    string = ""
    if current_survey == {}:
        return "There is no survey."
    
    # title
    try:
        content = _surveycpm_abbr_one_line(current_survey["title"], abbr=False)
        string += f"# Title: {content}\n\n"
    except:
        string += f"# Title: None\n\n"

    # sections
    if last_detail:
        now_section = _surveycpm_check_progress_postion_last_detail(current_survey)
    else:
        now_section = _surveycpm_check_progress_postion(current_survey)
    
    now_hire = now_section.count(".") if now_section else 0
    
    if "sections" in current_survey:
        for i, section in enumerate(current_survey["sections"]):
            title_key = "name" if "name" in section else "title"
            if now_section and (now_hire == 0 or (now_section.startswith(f"section-{i+1}") and now_hire == 1)):
                to_line_func = _surveycpm_to_one_line
            else:
                to_line_func = _surveycpm_abbr_one_line
            name, content = section[title_key], to_line_func(section)
            string += f"## Section-{i+1} [{name}]\n\n{content}\n\n"

            if "subsections" in section:
                for j, subsection in enumerate(section["subsections"]):
                    if now_section and ((now_section.startswith(f"section-{i+1}") and now_hire == 1) or \
                       (now_section.startswith(f"section-{i+1}.{j+1}") and now_hire == 2)):
                        to_line_func = _surveycpm_to_one_line
                    else:
                        to_line_func = _surveycpm_abbr_one_line
                    
                    name, content = subsection[title_key], to_line_func(subsection)
                    string += f"### Section-{i+1}.{j+1} [{name}]\n\n{content}\n\n"

                    if "subsections" in subsection:
                        for k, subsubsection in enumerate(subsection["subsections"]):
                            if now_section and now_section.startswith(f"section-{i+1}.{j+1}"):
                                to_line_func = _surveycpm_to_one_line
                            else:
                                to_line_func = _surveycpm_abbr_one_line
                            
                            name, content = subsubsection[title_key], to_line_func(subsubsection)
                            string += f"#### Section-{i+1}.{j+1}.{k+1} [{name}]\n\n{content}\n\n"
    
    return string.strip()


def _surveycpm_match_reference(text: str) -> List[str]:
    """Extract citation keys from LaTeX text."""
    reg = r"\\\w*cite(?!style)\w*\{(.+?)\}"
    placeholder_reg = re.compile(r"^#\d+$")
    reg_bibkeys = re.findall(reg, text)
    bibkeys = set()
    for bibkey in reg_bibkeys:
        single_bib = bibkey.split(",")
        for bib in single_bib:
            if not placeholder_reg.match(bib):
                bib = bib.strip()
                if bib and bib != "*":
                    bibkeys.add(bib)
                    
    reg = r"\\nocite{(.+?)\}"
    reg_bibkeys = re.findall(reg, text)
    for bibkey in reg_bibkeys:
        single_bib = bibkey.split(",")
        for bib in single_bib:
            if not placeholder_reg.match(bib):
                bib = bib.strip()
                if bib and bib != "*":
                    if bib in bibkeys:
                        bibkeys.remove(bib)
        
    ref_key_list = list(bibkeys)
    return ref_key_list


def _surveycpm_check_language_consistency(item: Any, user_instruction: str) -> bool:
    """Check if text language matches user instruction language."""
    if isinstance(item, str):
        text = item
    elif isinstance(item, dict):
        text = ""
        for v in item.values():
            if isinstance(v, str):
                text += v + "\n"
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, str):
                        text += vv + "\n"
                    elif isinstance(vv, dict):
                        for vvv in vv.values():
                            if isinstance(vvv, str):
                                text += vvv + "\n"
    elif isinstance(item, list):
        text = ""
        for v in item:
            if isinstance(v, str):
                text += v + "\n"
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str):
                        text += vv + "\n"
                    elif isinstance(vv, list):
                        for vvv in vv:
                            if isinstance(vvv, str):
                                text += vvv + "\n"
    else:
        return False
    
    text = text.strip()
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    text = re.sub(r'(\\\\cite)\{(.*?)\}', '', text, flags=re.DOTALL)
    text = re.sub(r'(\\cite)\{(.*?)\}', '', text, flags=re.DOTALL)
    comma_english = r'[!"#$%&\'()\*\+,-./:;<=>\?@\\\[\]^_`{\|}~1234567890]'
    text = re.sub(comma_english, "", text)
    if len(text) == 0:
        return True
    
    is_chinese = re.search(r'[\u4e00-\u9fff]', user_instruction) is not None
    
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_count = len(chinese_chars)
    total_chars = len(text)
    if is_chinese:
        return chinese_count / total_chars > 0.6
    else:
        return chinese_count / total_chars < 0.01


def surveycpm_parse_response(
    response_text: str,
    is_json: bool = True,
    valid_actions: List[str] | None = None,
    hard_mode: bool = True,
    **kwargs
) -> Dict[str, Any]:
    extracted_result = {}
    
    if valid_actions is None:
        valid_actions = ["search", "init-plan", "extend-plan", "nop", "write"]
    
    think_pattern = r"<thought>(.*?)</thought>"
    action_pattern = r"<action>(.*?)</action>"
    
    think_is_valid, action_is_valid = False, False
    
    think_match = re.search(think_pattern, response_text, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
        think_is_valid = True
    else:
        think = ""
    extracted_result["thought"] = think
    
    if is_json:
        action_match = re.search(action_pattern, response_text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            try:
                action = json.loads(action)
                # action_is_valid = True
                action_is_valid = surveycpm_validate_action(
                    action, 
                    valid_actions=valid_actions, 
                    hard_mode=hard_mode, # You can use hard mode for better performance
                    **kwargs
                )
            except:
                action_is_valid = False
                action = {}
        else:
            action_is_valid = False
            action = {}
    else:
        action_match = re.search(action_pattern, response_text, re.DOTALL | re.MULTILINE)
        if action_match:
            action = action_match.group(1).strip()
            # action_is_valid = True
            
        else:
            action = ""
        action = {"name": "write", "content": action}
        action_is_valid = surveycpm_validate_action(
            action, 
            valid_actions=valid_actions, 
            hard_mode=hard_mode, # You can use hard mode for better performance
            **kwargs
        )
    
    extracted_result["action"] = action
    extracted_result["parse_success"] = action_is_valid
    
    score = 0.0
    if not think_is_valid:
        score -= 1.0
    if not action_is_valid:
        score -= 2.0
    
    extracted_result["step_format"] = {
        "score": score,
        "thought": think_is_valid,
        "action": action_is_valid,
    }
    
    return extracted_result


def surveycpm_validate_action(
    action: Dict[str, Any],
    valid_actions: List[str],
    current_survey: Dict[str, Any] | None = None,
    cursor: str | None = None,
    user_instruction: str | None = None,
    hard_mode: bool = False,
    retrieved_bibkeys: List[str] | None = None
) -> bool:
    """Validate if a survey action is properly formatted."""
    if not isinstance(action, dict):
        return False
    if "name" not in action:
        return False
    
    if action["name"] not in valid_actions:
        return False
    
    try:
        if action["name"] == "search":
            assert "keywords" in action
            assert isinstance(action["keywords"], list)
            assert len(action["keywords"]) > 0
            assert action.keys() == {"name", "keywords"}
            for kw in action["keywords"]:
                assert isinstance(kw, str) and len(kw) > 0
            if hard_mode:
                assert len(action["keywords"]) <= 5
                
        elif action["name"] == "init-plan":
            assert "title" in action
            assert "sections" in action
            assert isinstance(action["title"], str) and len(action["title"]) > 0
            assert isinstance(action["sections"], list) and len(action["sections"]) > 0
            assert action.keys() == {"name", "title", "sections"}
            for sec in action["sections"]:
                assert isinstance(sec, dict)
                assert "title" in sec and "plan" in sec
                assert isinstance(sec["title"], str) and len(sec["title"]) > 0
                assert isinstance(sec["plan"], str) and len(sec["plan"]) > 0
                assert sec.keys() == {"title", "plan"}
            if hard_mode:
                assert 3 <= len(action["sections"]) <= 12
                if user_instruction:
                    assert _surveycpm_check_language_consistency(
                        {"title": action["title"], "sections": action["sections"]}, 
                        user_instruction
                    )
                
        elif action["name"] == "extend-plan":
            assert "position" in action
            assert "subsections" in action
            assert isinstance(action["position"], str) and len(action["position"]) > 0
            assert isinstance(action["subsections"], list) and len(action["subsections"]) > 0
            assert action.keys() == {"name", "position", "subsections"}
            if cursor is not None:
                assert action["position"] == cursor
            assert action["position"].count(".") < 2
            
            # Check if already extended
            if current_survey:
                try:
                    section_node = surveycpm_get_position(current_survey, action["position"], tag="outline")
                    if "subsections" in section_node:
                        return False
                except:
                    return False

            for sec in action["subsections"]:
                assert isinstance(sec, dict)
                assert "title" in sec and "plan" in sec
                assert isinstance(sec["title"], str) and len(sec["title"]) > 0
                assert isinstance(sec["plan"], str) and len(sec["plan"]) > 0
                assert sec.keys() == {"title", "plan"}
            if hard_mode:
                assert 2 <= len(action["subsections"]) <= 7
                if user_instruction:
                    assert _surveycpm_check_language_consistency(
                        {"subsections": action["subsections"]}, 
                        user_instruction
                    )
                
        elif action["name"] == "nop":
            assert action.keys() == {"name"}
            
        elif action["name"] == "write":
            assert "content" in action
            assert action.keys() == {"name", "content"}
            if hard_mode:
                assert "#" not in action["content"]
                assert "bibkey" not in action["content"].lower()
                assert len(action["content"].strip()) > 50
                if user_instruction:
                    assert _surveycpm_check_language_consistency(action["content"], user_instruction)
                ref_key_list = _surveycpm_match_reference(action["content"])
                if retrieved_bibkeys:
                    for ref_key in ref_key_list:
                        if ref_key not in retrieved_bibkeys:
                            return False
                assert action["content"].count("\\cite") <= 12
                assert "\\cite" in action["content"]
                
    except:
        return False
    
    return True


def surveycpm_update_position(
    survey: Dict[str, Any],
    position: str,
    update_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Update survey content at a specific position."""
    survey = copy.deepcopy(survey)
    
    current = survey
    if position == "outline":
        for key, value in update_data.items():
            current[key] = value
    else:
        parts = position.split('-')[1].split('.')
        indices = [int(part) - 1 for part in parts]
        for i, idx in enumerate(indices):
            if i == 0:
                current = current['sections'][idx]
            else:
                current = current['subsections'][idx]
        
        for key, value in update_data.items():
            current[key] = value
    
    return survey


def surveycpm_get_position(
    survey: Dict[str, Any],
    position: str,
    tag: str = "content"
) -> Any:
    """Get content at a specific position in the survey."""
    parts = position.split('-')[1].split('.')
    indices = [int(part) - 1 for part in parts]
    current = survey
    
    for i, idx in enumerate(indices):
        if i == 0:
            current = current['sections'][idx]
        else:
            current = current['subsections'][idx]
    
    if tag == "outline":
        return current
    elif tag == "content":
        return current.get('content', "")
    else:
        raise ValueError(f"Invalid tag: {tag}")


@app.tool(output="instruction_ls->state_ls,cursor_ls,survey_ls,step_ls,extend_time_ls,extend_result_ls,retrieved_info_ls,parsed_ls")
def surveycpm_state_init(
    instruction_ls: List[str],
) -> Dict[str, List]:

    n = len(instruction_ls)
    return {
        "state_ls": ["search"] * n,
        "cursor_ls": ["outline"] * n,
        "survey_ls": ["<PAD>"] * n,
        "step_ls": [0] * n,
        "extend_time_ls": [0] * n,
        "extend_result_ls": ["<PAD>"] * n,
        "retrieved_info_ls": ["<PAD>"] * n,
        "parsed_ls": [True] * n,
    }


@app.tool(output="response_ls,surveycpm_hard_mode->keywords_ls,parsed_ls")
def surveycpm_parse_search_response(
    response_ls: List[str],
    surveycpm_hard_mode: bool = True
) -> Dict[str, List]:
    keywords_ls = []
    parsed_ls = []
    
    for response in response_ls:
        result = surveycpm_parse_response(
            response_text=response, 
            is_json=True,
            valid_actions=["search"],
            hard_mode=surveycpm_hard_mode
        )
        keywords = result.get("action", {}).get("keywords", [])
        keywords_ls.append(keywords)
        parsed_ls.append(result.get("parse_success", False))
    
    return {"keywords_ls": keywords_ls, "parsed_ls": parsed_ls}


@app.tool(output="ret_psg_ls->retrieved_info_ls")
def surveycpm_process_passages(
    ret_psg_ls: List[List[List[str]]],
) -> Dict[str, List[str]]:

    top_k = 20
    retrieved_info_ls = []
    
    for ret_psg in ret_psg_ls:
        if not ret_psg:
            retrieved_info_ls.append("")
            continue
        
        num_queries = len(ret_psg)
        per_query_limit = max(1, top_k // num_queries)
        
        seen = set()
        all_passages = []
        
        for passages in ret_psg:
            for psg in passages[:per_query_limit]:
                if psg not in seen:
                    seen.add(psg)
                    all_passages.append(psg)
        
        remaining_slots = top_k - len(all_passages)
        if remaining_slots > 0:
            for passages in ret_psg:
                for psg in passages[per_query_limit:]:
                    if psg not in seen and remaining_slots > 0:
                        seen.add(psg)
                        all_passages.append(psg)
                        remaining_slots -= 1
        
        info = "\n\n".join(all_passages).strip()
        retrieved_info_ls.append(info)
    
    return {"retrieved_info_ls": retrieved_info_ls}


@app.tool(output="response_ls,survey_ls,instruction_ls,surveycpm_hard_mode->survey_ls,cursor_ls,parsed_ls")
def surveycpm_after_init_plan(
    response_ls: List[str],
    survey_ls: List[str], 
    instruction_ls: List[str],
    surveycpm_hard_mode: bool = True
) -> Dict[str, List]:

    import json
    new_survey_ls = []
    new_cursor_ls = []
    parsed_ls = []
    
    for response, survey_json, instruction in zip(response_ls, survey_ls, instruction_ls):
        result = surveycpm_parse_response(
            response_text=response, 
            is_json=True,
            user_instruction=instruction,
            valid_actions=["init-plan"],
            hard_mode=surveycpm_hard_mode
        )
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        parsed_ls.append(parse_success)
        
        if parse_success and action.get("name") == "init-plan":
            new_survey = {
                "title": action.get("title", ""),
                "sections": action.get("sections", [])
            }
            new_survey_ls.append(json.dumps(new_survey, ensure_ascii=False))
            new_cursor_ls.append(_surveycpm_check_progress_postion(new_survey))
        else:
            new_survey_ls.append(survey_json)
            new_cursor_ls.append("outline")
    
    return {
        "survey_ls": new_survey_ls,
        "cursor_ls": new_cursor_ls,
        "parsed_ls": parsed_ls,
    }


@app.tool(output="response_ls,survey_ls,cursor_ls,instruction_ls,retrieved_info_ls,surveycpm_hard_mode->survey_ls,cursor_ls,parsed_ls")
def surveycpm_after_write(
    response_ls: List[str],
    survey_ls: List[str],
    cursor_ls: List[str | None],
    instruction_ls: List[str],
    retrieved_info_ls: List[str],
    surveycpm_hard_mode: bool = True
) -> Dict[str, List]:

    import json
    new_survey_ls = []
    new_cursor_ls = []
    parsed_ls = []
    
    for response, survey_json, cursor, instruction, retrieved_info in zip(response_ls, survey_ls, cursor_ls, instruction_ls, retrieved_info_ls):
        survey = json.loads(survey_json) if survey_json and survey_json != "<PAD>" else {}
        
        retrieved_bibkeys = None
        if retrieved_info and retrieved_info != "<PAD>":
             retrieved_bibkeys = list(_surveycpm_match_reference(retrieved_info))
             
        result = surveycpm_parse_response(
            response_text=response, 
            is_json=False,
            current_survey=survey,
            cursor=cursor,
            user_instruction=instruction,
            retrieved_bibkeys=retrieved_bibkeys,
            valid_actions=["write"],
            hard_mode=surveycpm_hard_mode
        )
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        parsed_ls.append(parse_success)
        
        if parse_success and action.get("name") == "write":
            content = action.get("content", "")
            if content:
                new_survey = surveycpm_update_position(
                    survey=survey,
                    position=cursor,
                    update_data={"content": content}
                )
                new_survey_ls.append(json.dumps(new_survey, ensure_ascii=False))
                new_cursor = _surveycpm_check_progress_postion(new_survey)
                new_cursor_ls.append(new_cursor)
            else:
                new_survey_ls.append(survey_json)
                new_cursor_ls.append(cursor)
        else:
            new_survey_ls.append(survey_json)
            new_cursor_ls.append(cursor)
    
    return {
        "survey_ls": new_survey_ls,
        "cursor_ls": new_cursor_ls,
        "parsed_ls": parsed_ls,
    }


@app.tool(output="response_ls,survey_ls,cursor_ls,instruction_ls,surveycpm_hard_mode->survey_ls,cursor_ls,extend_result_ls,parsed_ls")
def surveycpm_after_extend(
    response_ls: List[str],
    survey_ls: List[str],  # JSON strings
    cursor_ls: List[str | None],
    instruction_ls: List[str],
    surveycpm_hard_mode: bool = True
) -> Dict[str, List]:

    import json
    new_survey_ls = []
    new_cursor_ls = []
    new_extend_result_ls = []
    parsed_ls = []
    
    for response, survey_json, cursor, instruction in zip(response_ls, survey_ls, cursor_ls, instruction_ls):
        survey = json.loads(survey_json) if survey_json and survey_json != "<PAD>" else {}
        result = surveycpm_parse_response(
            response_text=response, 
            is_json=True,
            current_survey=survey,
            cursor=cursor,
            user_instruction=instruction,
            valid_actions=["extend-plan", "nop"],
            hard_mode=surveycpm_hard_mode
        )
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        action_name = action.get("name", "")
        parsed_ls.append(parse_success)
        
        if parse_success and action_name == "extend-plan":
            position = action.get("position", "")
            subsections = action.get("subsections", [])
            
            if position and subsections:
                new_survey = surveycpm_update_position(
                    survey=survey,
                    position=position,
                    update_data={"subsections": copy.deepcopy(subsections)}
                )
                new_survey_ls.append(json.dumps(new_survey, ensure_ascii=False))
                new_cursor_ls.append(_surveycpm_check_progress_postion(new_survey))
                new_extend_result_ls.append("extended")
            else:
                new_survey_ls.append(survey_json)
                new_cursor_ls.append(cursor)
                new_extend_result_ls.append("retry")
                
        elif parse_success and action_name == "nop":
            new_survey_ls.append(survey_json)
            new_cursor_ls.append(cursor)
            new_extend_result_ls.append("nop")
        else:
            new_survey_ls.append(survey_json)
            new_cursor_ls.append(cursor)
            new_extend_result_ls.append("retry")
    
    return {
        "survey_ls": new_survey_ls,
        "cursor_ls": new_cursor_ls,
        "extend_result_ls": new_extend_result_ls,
        "parsed_ls": parsed_ls,
    }

@app.tool(output="state_ls,cursor_ls,extend_time_ls,extend_result_ls,step_ls,parsed_ls,surveycpm_max_step,surveycpm_max_extend_step->state_ls,extend_time_ls,step_ls")
def surveycpm_update_state(
    state_ls: List[str],
    cursor_ls: List[str | None],
    extend_time_ls: List[int],
    extend_result_ls: List[str],
    step_ls: List[int],
    parsed_ls: List[bool],
    surveycpm_max_step: int = 140,
    surveycpm_max_extend_step: int = 5,
) -> Dict[str, List]:
    """Update state based on cursor and extend results.
    
    This runs OUTSIDE of branch, so all items are processed together.
    Handles all state transitions in one place to avoid length mismatch issues.
    
    State transition logic:
    - If step >= surveycpm_max_step: -> done
    - If parsed_ls is False: -> keep current state (retry)
    - If current state is 'search':
        - cursor == 'outline': -> analyst-init_plan
        - cursor is section-X: -> write
        - cursor is None: -> done (shouldn't happen in search)
    - If current state is 'analyst-init_plan':
        - cursor != 'outline': -> search (init plan succeeded)
        - cursor == 'outline': -> analyst-init_plan (retry)
    - If current state is 'write':
        - cursor is not None: -> search (continue writing)
        - cursor is None and extend_time < surveycpm_max_extend_step: -> analyst-extend_plan
        - cursor is None and extend_time >= surveycpm_max_extend_step: -> done
    - If current state is 'analyst-extend_plan':
        - extend_result == 'extended': -> search
        - extend_result == 'nop': -> done
        - extend_result == 'retry': -> analyst-extend_plan
    - If current state is 'done': -> done
    """
    new_state_ls = []
    new_extend_time_ls = []
    new_step_ls = []
    
    # if parsed_ls is empty, pad with True
    if not parsed_ls:
        parsed_ls = [True] * len(state_ls)
    
    for i, (state, cursor, extend_time, step, parsed) in enumerate(
        zip(state_ls, cursor_ls, extend_time_ls, step_ls, parsed_ls)
    ):
        extend_result = extend_result_ls[i] if i < len(extend_result_ls) else "<PAD>"
        if extend_result == "<PAD>":
            extend_result = ""
        
        if step >= surveycpm_max_step:
            new_state_ls.append("done")
            new_extend_time_ls.append(extend_time)
            # still increment step
            new_step_ls.append(step + 1)
            continue

        if not parsed:
            
            if state != "analyst-extend_plan":
                new_extend_time_ls.append(extend_time)
                new_state_ls.append(state)
            else:
                new_extend_time_ls.append(extend_time + 1)
                if extend_time < surveycpm_max_extend_step:
                    new_state_ls.append("analyst-extend_plan")
                else:
                    new_state_ls.append("done")
            new_step_ls.append(step + 1)
            continue
        
        if state == "search":
            if cursor == "outline":
                new_state_ls.append("analyst-init_plan")
            elif cursor is not None:
                new_state_ls.append("write")
            else:
                new_state_ls.append("done")
            new_extend_time_ls.append(extend_time)
            
        elif state == "analyst-init_plan":
            if cursor != "outline" and cursor is not None:
                new_state_ls.append("search")
            else:
                new_state_ls.append("analyst-init_plan")
            new_extend_time_ls.append(extend_time)
            
        elif state == "write":
            if cursor is not None:
                new_state_ls.append("search")
                new_extend_time_ls.append(extend_time)
            elif extend_time < surveycpm_max_extend_step:
                new_state_ls.append("analyst-extend_plan")
                new_extend_time_ls.append(extend_time + 1)
            else:
                new_state_ls.append("done")
                new_extend_time_ls.append(extend_time)
                
        elif state == "analyst-extend_plan":
            if extend_result == "extended":
                new_state_ls.append("search")
            elif extend_result == "nop":
                new_state_ls.append("done")
            elif extend_time < surveycpm_max_extend_step:
                new_state_ls.append("analyst-extend_plan")
            else:
                new_state_ls.append("done")
            new_extend_time_ls.append(extend_time)
            
        else:  # done or unknown
            new_state_ls.append("done")
            new_extend_time_ls.append(extend_time)
        
        step = step + 1
        new_step_ls.append(step)
    
    return {
        "state_ls": new_state_ls,
        "extend_time_ls": new_extend_time_ls,
        "step_ls": new_step_ls,
    }

@app.tool(output="step_ls,state_ls,surveycpm_max_step->state_ls")
def surveycpm_check_completion(
    step_ls: List[int],
    state_ls: List[str],
    surveycpm_max_step: int = 140
) -> Dict[str, List]:
    """Check completion status based on step count only.
    
    This runs OUTSIDE of branch, so all items are processed together.
    
    Note: The extend logic is now handled in surveycpm_after_write.
    This function only checks if surveycpm_max_step is reached.
    
    Logic:
    - If step >= surveycpm_max_step: -> done
    - Otherwise: keep current state (which may already be analyst-extend_plan from surveycpm_after_write)
    """
    # If inputs are empty (due to framework filtering), return empty to avoid overwriting
    if not step_ls or not state_ls:
        app.logger.warning("[surveycpm_check_completion] Empty inputs, returning empty to avoid overwriting")
        return {"state_ls": state_ls}
    
    new_state_ls = []
    
    for step, state in zip(step_ls, state_ls):
        if step >= surveycpm_max_step:
            new_state_ls.append("done")
        else:
            # Keep current state (analyst-extend_plan, search, write, done, etc.)
            new_state_ls.append(state)
    
    return {"state_ls": new_state_ls}


def _surveycpm_clean_content(content: str) -> str:
    """Clean up content text for proper Markdown formatting.
    
    This function:
    - Removes "Section-X.X.X" style references from content
    - Fixes improper markdown headers (headers with leading spaces)
    - Normalizes multiple consecutive blank lines to single blank line
    - Ensures proper spacing around headers
    """
    if not content:
        return ""
    
    # Remove "Section-X.X.X" patterns (e.g., "Section-1", "Section-2.3", "Section-4.3.1")
    # But preserve the context around them
    content = re.sub(r'\bSection-(\d+(?:\.\d+)*)\b', '', content)
    
    # Fix headers that have leading spaces (not valid markdown)
    # Match lines like "    ## Title" or "        ### Title" and remove leading spaces
    content = re.sub(r'^[ \t]+(#{1,6})\s+', r'\1 ', content, flags=re.MULTILINE)
    
    # Normalize multiple consecutive blank lines to at most two newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Ensure headers have a blank line before them (except at start)
    content = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', content)
    
    # Ensure headers have a blank line after them
    content = re.sub(r'(#{1,6}\s[^\n]+)\n([^\n#])', r'\1\n\n\2', content)
    
    return content.strip()

def _surveycpm_clean_title(title:str) -> str:
    # remove any thing like section 1., Section-1., 1., (1), 一、, （一）, 一. ，
    # and keep the original title
    # Some sources use full-width dots "．" or "。" in section numbers.
    _dot = r'[.\uFF0E\u3002]'

    title = re.sub(rf'(?i)^\s*section[-\s]*[\d{_dot}]*\s*', '', title)
    # remove headings like "第一章 第一节 第一部 ..." / "第1章: ..." (can repeat)
    title = re.sub(
        r'^(?:第\s*[0-9一二三四五六七八九十百千]+\s*(?:章|节|部(?:分)?|篇|卷)\s*[-—:：\.、，,]?\s*)+',
        '',
        title,
    )
    # remove headings like "3.4.2，" / "3.4.2." / "3.4.2 "
    title = re.sub(rf'^\s*\d+(?:{_dot}\d+)+[\.、，,]?\s*', '', title)
    title = re.sub(r'^\s*\d+[.、，,]\s*', '', title)
    title = re.sub(r'[\(（]\d+[\)）]\s*', '', title)
    title = re.sub(rf'^\s*[一二三四五六七八九十]+(?:{_dot}\d+)+[\.、，,]?\s*', '', title)
    title = re.sub(r'^\s*[一二三四五六七八九十]+[.、，,]\s*', '', title)
    title = re.sub(r'[\(（][一二三四五六七八九十]+[\)）]\s*', '', title)
    title = re.sub(r'^[一二三四五六七八九十]+\s+', '', title)
    return title.strip()


def _surveycpm_to_one_line_old(string):
    if isinstance(string, dict):
        if "content" in string and string["content"]:
            return _surveycpm_to_one_line_old(string["content"])
        elif "plan" in string and string["plan"]:
            return _surveycpm_to_one_line_old(string["plan"])
        else:
            return ""
    if not string:
        return ""
    else:
        return string


def _surveycpm_to_one_line_old(string):
    if isinstance(string, dict):
        if "content" in string and string["content"]:
            return _surveycpm_to_one_line_old(string["content"])
        elif "plan" in string and string["plan"]:
            return _surveycpm_to_one_line_old(string["plan"])
        else:
            return ""
    if not string:
        return ""
    else:
        return string


def _surveycpm_format_survey_markdown(survey: Dict[str, Any]) -> str:
    """Format survey as clean Markdown for final output.
    
    Unlike _surveycpm_print_tasknote_hire, this function:
    - Does NOT add [OK] or [PLAN] prefixes
    - Preserves proper paragraph breaks and formatting
    - Produces clean, renderable Markdown
    - Cleans up "Section-X.X.X" references in content
    - Ensures proper markdown header formatting
    - Adds section numbers (Section 1. xxx, Section 1.1 xxx, etc.)
    """
    if not survey or survey == {}:
        return "No survey generated."
    
    lines = []
    
    # Title
    title = survey.get("title", "Untitled Survey")
    lines.append(f"# {title}")
    lines.append("")
    
    # Sections
    sections = survey.get("sections", [])
    for i, section in enumerate(sections):
        title_key = "name" if "name" in section else "title"
        section_title = section.get(title_key, "")
        section_title = _surveycpm_clean_title(section_title)
        section_num = i + 1
        
        lines.append(f"## {section_num} {section_title}")
        # lines.append(f"## **{section_title}** ")
        lines.append("")
        
        # Section content
        if "content" in section and section["content"]:
            cleaned_content = _surveycpm_clean_content(section["content"])
            if cleaned_content:
                lines.append(cleaned_content)
                lines.append("")
        elif "plan" in section and section["plan"]:
            lines.append(f"*{section['plan'].strip()}*")
            lines.append("")
        
        # Subsections
        if "subsections" in section:
            for j, subsection in enumerate(section["subsections"]):
                subsection_title = subsection.get(title_key, "")
                subsection_title = _surveycpm_clean_title(subsection_title)
                subsection_num = f"{section_num}.{j + 1}"
                
                lines.append(f"### {subsection_num} {subsection_title}")
                # lines.append(f"### **{subsection_title}**")
                lines.append("")
                
                if "content" in subsection and subsection["content"]:
                    cleaned_content = _surveycpm_clean_content(subsection["content"])
                    if cleaned_content:
                        lines.append(cleaned_content)
                        lines.append("")
                elif "plan" in subsection and subsection["plan"]:
                    lines.append(f"*{subsection['plan'].strip()}*")
                    lines.append("")
                
                # Sub-subsections
                if "subsections" in subsection:
                    for k, subsubsection in enumerate(subsection["subsections"]):
                        subsubsection_title = subsubsection.get(title_key, "")
                        subsubsection_num = f"{section_num}.{j + 1}.{k + 1}"
                        subsubsection_title = _surveycpm_clean_title(subsubsection_title)
                        lines.append(f"#### {subsubsection_num} {subsubsection_title}")
                        # lines.append(f"#### **{subsubsection_title}**")
                        lines.append("")
                        
                        if "content" in subsubsection and subsubsection["content"]:
                            cleaned_content = _surveycpm_clean_content(subsubsection["content"])
                            if cleaned_content:
                                lines.append(cleaned_content)
                                lines.append("")
                        elif "plan" in subsubsection and subsubsection["plan"]:
                            lines.append(f"*{subsubsection['plan'].strip()}*")
                            lines.append("")
    
    # Final cleanup: normalize multiple blank lines
    result = "\n".join(lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    # change citation format from bibkey to []
    def extract_num_from_textid(textid):
        try:
            if textid.startswith("textid"):
                return int(textid.split("textid")[-1].strip())
        except:
            return None
        return None
    def replace_bibkey(match): # \\cite{textid2, textid1} -> [1,2]
        bibkey_group = match.group(1)
        single_bibs = [bib.strip() for bib in bibkey_group.split(",")]
        new_bibs = []
        for bib in single_bibs:
            if bib.startswith("bibkey: "):
                bib = bib[len("bibkey: "):].strip()
            num = extract_num_from_textid(bib)
            if num is not None:
                new_bibs.append(str(num))
        new_bibs = sorted(set(new_bibs), key=lambda x: int(x))
        if len(new_bibs) > 0:
            return f'{"".join([f"[{i}]" for i in new_bibs])}'
        else:
            return ""
    reg = r"\\cite\{(.+?)\}"
    result = re.sub(reg, replace_bibkey, result)
    return result.strip()


@app.tool(output="survey_ls,instruction_ls->ans_ls")
def surveycpm_format_output(
    survey_ls: List[str],  # JSON strings
    instruction_ls: List[str]
) -> Dict[str, List[str]]:
    """Format final survey output as clean Markdown.
    survey_ls contains JSON strings that are parsed.
    """
    import json
    ans_ls = []
    for survey_json, instruction in zip(survey_ls, instruction_ls):
        survey = json.loads(survey_json) if survey_json and survey_json != "<PAD>" else {}
        output = _surveycpm_format_survey_markdown(survey)
        ans_ls.append(output)
    
    return {"ans_ls": ans_ls}


if __name__ == "__main__":
    app.run(transport="stdio")
