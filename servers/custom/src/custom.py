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
    comma_english = r'[!"#$%&\'()\*\+,-./:;<=>\?@\\\[\]^_`{\|}~]'
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
        return chinese_count / total_chars < 0.3


def surveycpm_parse_response(
    response_text: str,
    is_json: bool = True
) -> Dict[str, Any]:
    """Parse LLM response for survey generation."""
    extracted_result = {}
    
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
                action_is_valid = True
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
            action_is_valid = True
        else:
            action = ""
        action = {"name": "write", "content": action}
    
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
                assert len(action["content"].strip()) > 100
                if user_instruction:
                    assert _surveycpm_check_language_consistency(action["content"], user_instruction)
                ref_key_list = _surveycpm_match_reference(action["content"])
                if retrieved_bibkeys:
                    for ref_key in ref_key_list:
                        if ref_key not in retrieved_bibkeys:
                            return False
                assert action["content"].count("\\cite") < 10
                
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


@app.tool(output="instruction_ls->state_ls,cursor_ls,survey_ls,retrieved_info_ls,step_ls,extend_time_ls,no_check_ls,no_extend_ls")
def surveycpm_state_init(
    instruction_ls: List[str]
) -> Dict[str, List]:
    """Initialize survey state for all instances.
    
    Note: survey_ls is stored as JSON strings to avoid being filtered by
    UltraRAG's branch filtering logic which filters dict lists by branch state.
    """
    import json
    n = len(instruction_ls)
    return {
        "state_ls": ["search"] * n,
        "cursor_ls": ["outline"] * n,
        # Store as JSON strings to avoid branch filtering (dict lists get filtered)
        "survey_ls": [json.dumps({}) for _ in range(n)],
        "retrieved_info_ls": [""] * n,
        "step_ls": [0] * n,
        "extend_time_ls": [0] * n,
        "no_check_ls": [True] * n,
        "no_extend_ls": [True] * n,
    }


@app.tool(output="response_ls->keywords_ls,parse_success_ls")
def surveycpm_parse_search_response(
    response_ls: List[str]
) -> Dict[str, List]:
    """Parse search responses and extract keywords.
    
    Returns keywords_ls as 2D list: [[kw1, kw2], [kw3], ...]
    Each inner list contains keywords for one batch item.
    """
    keywords_ls = []
    parse_success_ls = []
    
    for response in response_ls:
        result = surveycpm_parse_response(
            response_text=response,
            is_json=True
        )
        
        keywords = result.get("action", {}).get("keywords", [])
        parse_success = result.get("parse_success", False) and len(keywords) > 0
        
        keywords_ls.append(keywords)
        parse_success_ls.append(parse_success)
    
    return {
        "keywords_ls": keywords_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool(output="retrieved_info_ls,cursor_ls->retrieved_info_ls,state_ls")
def surveycpm_after_search(
    retrieved_info_ls: List[str],
    cursor_ls: List[str | None],
) -> Dict[str, List]:
    """Process search results and determine next state.
    
    State transitions based on cursor:
    - cursor == "outline": -> analyst-init_plan (need to create survey structure)
    - cursor == section-X: -> write (need to write content for this section)
    - cursor is None: -> done (all sections completed)
    """
    new_retrieved_info_ls = []
    state_ls = []
    
    for info, cursor in zip(retrieved_info_ls, cursor_ls):
        # Handle edge case: cursor is None means all sections are done
        if cursor is None:
            state_ls.append("done")
            new_retrieved_info_ls.append("")
            continue
        
        # Keep the retrieved info (may be empty string if retrieval failed)
        new_retrieved_info_ls.append(info if info else "")
        
        # Transition based on cursor state
        if cursor == "outline":
            state_ls.append("analyst-init_plan")
        else:
            state_ls.append("write")
    
    return {
        "retrieved_info_ls": new_retrieved_info_ls,
        "state_ls": state_ls
    }


@app.tool(output="response_ls,survey_ls->survey_ls,state_ls,cursor_ls,parse_success_ls")
def surveycpm_after_init_plan(
    response_ls: List[str],
    survey_ls: List[str]  # JSON strings
) -> Dict[str, List]:
    """Parse init_plan responses and create survey structures.
    survey_ls contains JSON strings that are parsed/serialized.
    """
    import json
    new_survey_ls = []
    state_ls = []
    cursor_ls = []
    parse_success_ls = []
    
    for response, survey_json in zip(response_ls, survey_ls):
        survey = json.loads(survey_json) if survey_json else {}
        
        result = surveycpm_parse_response(
            response_text=response,
            is_json=True
        )
        
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        
        if parse_success and action.get("name") == "init-plan":
            new_survey = {
                "title": action.get("title", ""),
                "sections": action.get("sections", [])
            }
            new_survey_ls.append(json.dumps(new_survey))
            state_ls.append("search")
            cursor_ls.append(_surveycpm_check_progress_postion(new_survey))
            parse_success_ls.append(True)
        else:
            new_survey_ls.append(survey_json)
            state_ls.append("analyst-init_plan")
            cursor_ls.append("outline")
            parse_success_ls.append(False)
    
    return {
        "survey_ls": new_survey_ls,
        "state_ls": state_ls,
        "cursor_ls": cursor_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool(output="response_ls,survey_ls,cursor_ls,no_check_ls,no_extend_ls->survey_ls,state_ls,cursor_ls,parse_success_ls")
def surveycpm_after_write(
    response_ls: List[str],
    survey_ls: List[str],  # JSON strings
    cursor_ls: List[str | None],
    no_check_ls: List[bool],
    no_extend_ls: List[bool]
) -> Dict[str, List]:
    """Parse write responses and update survey content.
    survey_ls contains JSON strings that are parsed/serialized.
    
    State transitions after successful write:
    - no_check=True, no_extend=True: -> search
    - no_check=True, no_extend=False: -> analyst-extend_plan (if cursor depth < 2)
    - no_check=False: -> analyst-check_para (not implemented yet)
    """
    import json
    new_survey_ls = []
    state_ls = []
    new_cursor_ls = []
    parse_success_ls = []
    
    for response, survey_json, cursor, no_check, no_extend in zip(
        response_ls, survey_ls, cursor_ls, no_check_ls, no_extend_ls
    ):
        survey = json.loads(survey_json) if survey_json else {}
        
        result = surveycpm_parse_response(
            response_text=response,
            is_json=False
        )
        
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        
        if parse_success and action.get("name") == "write":
            content = action.get("content", "")
            if content:
                new_survey = surveycpm_update_position(
                    survey=survey,
                    position=cursor,
                    update_data={"content": content}
                )
                new_survey_ls.append(json.dumps(new_survey))
                new_cursor = _surveycpm_check_progress_postion(new_survey)
                new_cursor_ls.append(new_cursor)
                
                # Determine next state based on no_check and no_extend flags
                if no_check:
                    if no_extend:
                        # Disable check and extend: go directly to search
                        state_ls.append("search")
                    else:
                        # Disable check but allow extend: 
                        # extend if cursor depth < 2, otherwise search
                        # Note: cursor should not be None here, but handle it safely
                        if cursor is not None and cursor.count(".") < 2:
                            state_ls.append("analyst-extend_plan")
                        else:
                            state_ls.append("search")
                else:
                    # Check enabled (not implemented, default to search)
                    state_ls.append("search")
                    
                parse_success_ls.append(True)
            else:
                new_survey_ls.append(survey_json)
                state_ls.append("write")
                new_cursor_ls.append(cursor)
                parse_success_ls.append(False)
        else:
            new_survey_ls.append(survey_json)
            state_ls.append("write")
            new_cursor_ls.append(cursor)
            parse_success_ls.append(False)
    
    return {
        "survey_ls": new_survey_ls,
        "state_ls": state_ls,
        "cursor_ls": new_cursor_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool(output="response_ls,survey_ls,cursor_ls,extend_time_ls->survey_ls,state_ls,cursor_ls,extend_time_ls,parse_success_ls")
def surveycpm_after_extend(
    response_ls: List[str],
    survey_ls: List[str],  # JSON strings
    cursor_ls: List[str | None],
    extend_time_ls: List[int]
) -> Dict[str, List]:
    """Parse extend responses and handle extend-plan/nop actions.
    survey_ls contains JSON strings that are parsed/serialized.
    """
    import json
    new_survey_ls = []
    state_ls = []
    new_cursor_ls = []
    new_extend_time_ls = []
    parse_success_ls = []
    
    for response, survey_json, cursor, extend_time in zip(
        response_ls, survey_ls, cursor_ls, extend_time_ls
    ):
        survey = json.loads(survey_json) if survey_json else {}
        
        result = surveycpm_parse_response(
            response_text=response,
            is_json=True
        )
        
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        action_name = action.get("name", "")
        
        if parse_success and action_name == "extend-plan":
            position = action.get("position", "")
            subsections = action.get("subsections", [])
            
            if position and subsections:
                new_survey = surveycpm_update_position(
                    survey=survey,
                    position=position,
                    update_data={"subsections": copy.deepcopy(subsections)}
                )
                new_survey_ls.append(json.dumps(new_survey))
                state_ls.append("search")
                new_cursor_ls.append(_surveycpm_check_progress_postion(new_survey))
                new_extend_time_ls.append(extend_time)
                parse_success_ls.append(True)
            else:
                new_survey_ls.append(survey_json)
                state_ls.append("analyst-extend_plan")
                new_cursor_ls.append(cursor)
                new_extend_time_ls.append(extend_time)
                parse_success_ls.append(False)
                
        elif parse_success and action_name == "nop":
            new_survey_ls.append(survey_json)
            state_ls.append("search")
            new_cursor_ls.append(cursor)
            new_extend_time_ls.append(12)
            parse_success_ls.append(True)
        else:
            new_survey_ls.append(survey_json)
            state_ls.append("analyst-extend_plan")
            new_cursor_ls.append(cursor)
            new_extend_time_ls.append(extend_time)
            parse_success_ls.append(False)
    
    return {
        "survey_ls": new_survey_ls,
        "state_ls": state_ls,
        "cursor_ls": new_cursor_ls,
        "extend_time_ls": new_extend_time_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool(output="step_ls->step_ls")
def surveycpm_increment_step(
    step_ls: List[int]
) -> Dict[str, List[int]]:
    """Increment step counter for all instances."""
    return {"step_ls": [step + 1 for step in step_ls]}


@app.tool(output="cursor_ls,extend_time_ls,no_extend_ls,state_ls,step_ls->state_ls,extend_time_ls,done_ls")
def surveycpm_check_completion(
    cursor_ls: List[str | None],
    extend_time_ls: List[int],
    no_extend_ls: List[bool],
    state_ls: List[str],
    step_ls: List[int],
    max_step: int = 140
) -> Dict[str, List]:
    """Check if survey generation is complete.
    
    Logic based on source code:
    - If step >= max_step: done (incomplete)
    - If cursor is None (all sections written):
      - If extend_time < 10 and no_extend: allow more extend attempts
      - Else: done (complete)
    """
    new_state_ls = []
    new_extend_time_ls = []
    done_ls = []
    
    for cursor, extend_time, no_extend, state, step in zip(
        cursor_ls, extend_time_ls, no_extend_ls, state_ls, step_ls
    ):
        if step >= max_step:
            new_state_ls.append("done")
            new_extend_time_ls.append(extend_time)
            done_ls.append(False)  # Incomplete due to max step
            continue
        
        if cursor is None:
            # All sections have content, check if we should extend
            if extend_time < 10 and no_extend:
                # Still have extend attempts in no_extend mode
                new_state_ls.append("analyst-extend_plan")
                new_extend_time_ls.append(extend_time + 1)
                done_ls.append(False)
            else:
                # No more extend attempts or not in no_extend mode
                new_state_ls.append("done")
                new_extend_time_ls.append(extend_time)
                done_ls.append(True)  # Successfully complete
        else:
            new_state_ls.append(state)
            new_extend_time_ls.append(extend_time)
            done_ls.append(False)
    
    return {
        "state_ls": new_state_ls,
        "extend_time_ls": new_extend_time_ls,
        "done_ls": done_ls
    }


@app.tool(output="survey_ls,instruction_ls->ans_ls")
def surveycpm_format_output(
    survey_ls: List[str],  # JSON strings
    instruction_ls: List[str]
) -> Dict[str, List[str]]:
    """Format final survey output.
    survey_ls contains JSON strings that are parsed.
    """
    import json
    ans_ls = []
    for survey_json, instruction in zip(survey_ls, instruction_ls):
        survey = json.loads(survey_json) if survey_json else {}
        if not survey or survey == {}:
            ans_ls.append("No survey generated.")
        else:
            output = _surveycpm_print_tasknote_hire(survey, last_detail=False)
            ans_ls.append(output)
    
    return {"ans_ls": ans_ls}


if __name__ == "__main__":
    app.run(transport="stdio")
