import os
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jinja2 import Template
from jinja2.sandbox import SandboxedEnvironment
from markupsafe import escape

from fastmcp.prompts import PromptMessage
from ultrarag.server import UltraRAG_MCP_Server


app = UltraRAG_MCP_Server("prompt")

# Create a sandboxed Jinja2 environment for security
_sandboxed_env = SandboxedEnvironment(autoescape=True)


def _validate_template_path(template_path: Union[str, Path]) -> Path:
    """Validate template path to prevent path traversal.
    
    Args:
        template_path: Path to template file
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or contains traversal attempts
    """
    path = Path(template_path)
    
    # Check for path traversal
    if ".." in str(path):
        raise ValueError(f"Path traversal detected in template path: {template_path}")
    
    # Resolve to absolute path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid template path: {template_path}") from e
    
    return resolved


def load_prompt_template(template_path: Union[str, Path]) -> Template:
    """Load Jinja2 template from file with security validation.

    Args:
        template_path: Path to template file

    Returns:
        Jinja2 Template object from sandboxed environment

    Raises:
        FileNotFoundError: If template file doesn't exist
        ValueError: If template path is invalid
    """
    # Validate path to prevent traversal
    safe_path = _validate_template_path(template_path)
    
    if not safe_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Load template using sandboxed environment
    with open(safe_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    
    # Use sandboxed environment to prevent code injection
    return _sandboxed_env.from_string(template_content)


def _safe_render(template: Template, **kwargs: Any) -> str:
    """Safely render a template with escaped user inputs.
    
    Args:
        template: Jinja2 Template object
        **kwargs: Template variables (will be escaped if strings)
        
    Returns:
        Rendered template string
    """
    # Escape all string inputs to prevent XSS and injection
    safe_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            safe_kwargs[key] = escape(value)
        elif isinstance(value, list):
            # Escape string items in lists
            safe_kwargs[key] = [escape(str(item)) if isinstance(item, str) else item for item in value]
        else:
            safe_kwargs[key] = value
    
    return template.render(**safe_kwargs)


@app.prompt(output="q_ls,template->prompt_ls")
def qa_boxed(q_ls: List[str], template: Union[str, Path]) -> List[PromptMessage]:
    """Generate prompts for QA boxed format.

    Args:
        q_ls: List of questions
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    for q in q_ls:
        p = _safe_render(template, question=q)
        ret.append(p)
    return ret


@app.prompt(output="q_ls,choices_ls,template->prompt_ls")
def qa_boxed_multiple_choice(
    q_ls: List[str],
    choices_ls: List[List[str]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for multiple choice QA boxed format.

    Args:
        q_ls: List of questions
        choices_ls: List of choice lists (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    CHOICES: List[str] = list(string.ascii_uppercase)  # A, B, ..., Z
    for q, choices in zip(q_ls, choices_ls):
        choices_text = "\n".join(f"{CHOICES[i]}: {c}" for i, c in enumerate(choices))
        p = _safe_render(template, question=q, choices=choices_text)
        ret.append(p)
    return ret


@app.prompt(output="q_ls,ret_psg,template->prompt_ls")
def qa_rag_boxed(
    q_ls: List[str], ret_psg: List[Union[str, Any]], template: Union[str, Path]
) -> List[PromptMessage]:
    """Generate prompts for QA RAG boxed format.

    Args:
        q_ls: List of questions
        ret_psg: List of retrieved passages (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = _safe_render(template, question=q, documents=passage_text)
        ret.append(p)
    return ret


@app.prompt(output="q_ls,choices_ls,ret_psg,template->prompt_ls")
def qa_rag_boxed_multiple_choice(
    q_ls: List[str],
    choices_ls: List[List[str]],
    ret_psg: List[List[str]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for QA RAG boxed format with multiple choice.

    Args:
        q_ls: List of questions
        choices_ls: List of choice lists (one per question)
        ret_psg: List of retrieved passage lists (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    CHOICES: List[str] = list(string.ascii_uppercase)  # A, B, ..., Z
    for q, psg, choices in zip(q_ls, ret_psg, choices_ls):
        passage_text = "\n".join(psg)
        choices_text = "\n".join(f"{CHOICES[i]}: {c}" for i, c in enumerate(choices))
        p = _safe_render(template, question=q, documents=passage_text, choices=choices_text)
        ret.append(p)
    return ret


@app.prompt(output="q_ls,ret_psg,kr_template->prompt_ls")
def RankCoT_kr(
    q_ls: List[str],
    ret_psg: List[Union[str, Any]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for RankCoT knowledge retrieval step.

    Args:
        q_ls: List of questions
        ret_psg: List of retrieved passages (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = _safe_render(template, question=q, documents=passage_text)
        ret.append(p)
    return ret


@app.prompt(output="q_ls,kr_ls,qa_template->prompt_ls")
def RankCoT_qa(
    q_ls: List[str],
    kr_ls: List[str],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for RankCoT QA step.

    Args:
        q_ls: List of questions
        kr_ls: List of knowledge retrieval results (CoT reasoning)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    for q, cot in zip(q_ls, kr_ls):
        p = _safe_render(template, question=q, CoT=cot)
        ret.append(p)
    return ret


@app.prompt(output="memory_q_ls,memory_ret_psg,template->prompt_ls")
def ircot_next_prompt(
    memory_q_ls: List[List[Optional[str]]],
    memory_ret_psg: List[Optional[List[List[str]]]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for IRCoT (Iterative Retrieval Chain-of-Thought) pipeline.

    Args:
        memory_q_ls: List of question lists (one per round)
        memory_ret_psg: List of retrieved passage lists (one per round)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret: List[PromptMessage] = []
    # Single turn
    if len(memory_q_ls) == 1:
        for q, psg in zip(memory_q_ls[0], memory_ret_psg[0]):  # type: ignore[arg-type]
            if q is None:
                continue
            passage_text = "" if psg is None else "\n".join(psg)
            ret.append(
                _safe_render(template, documents=passage_text, question=q, cur_answer="")
            )
        return ret
    # Multi turn
    data_num = len(memory_q_ls[0])
    round_cnt = len(memory_q_ls)
    for i in range(data_num):
        if memory_q_ls[0][i] is None:
            continue
        all_passages, all_cots = [], []
        for r in range(round_cnt):
            psg = None
            if memory_ret_psg is not None and r < len(memory_ret_psg):
                round_psg = memory_ret_psg[r]
                if round_psg is not None and i < len(round_psg):
                    psg = round_psg[i]
            if psg:
                all_passages.extend(psg)
            if r > 0:
                cot = memory_q_ls[r][i]
                if cot:
                    all_cots.append(cot)
        passage_text = "\n".join(all_passages)
        cur_answer = " ".join(all_cots).strip()
        q = memory_q_ls[0][i]
        ret.append(
            _safe_render(template, documents=passage_text, question=q, cur_answer=cur_answer)
        )
    return ret


@app.prompt(output="q_ls,plan_ls,webnote_init_page_template->prompt_ls")
def webnote_init_page(
    q_ls: List[str],
    plan_ls: List[str],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for WebNote initial page creation.

    Args:
        q_ls: List of questions
        plan_ls: List of plans (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, plan in zip(q_ls, plan_ls):
        p = _safe_render(template, question=q, plan=plan)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,webnote_gen_plan_template->prompt_ls")
def webnote_gen_plan(
    q_ls: List[str],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for WebNote plan generation.

    Args:
        q_ls: List of questions
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q in q_ls:
        p = _safe_render(template, question=q)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,plan_ls,page_ls,webnote_gen_subq_template->prompt_ls")
def webnote_gen_subq(
    q_ls: List[str],
    plan_ls: List[str],
    page_ls: List[str],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for WebNote sub-question generation.

    Args:
        q_ls: List of questions
        plan_ls: List of plans (one per question)
        page_ls: List of page contents (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, plan, page in zip(q_ls, plan_ls, page_ls):
        p = _safe_render(template, question=q, plan=plan, page=page)
        all_prompts.append(p)
    return all_prompts


@app.prompt(
    output="q_ls,plan_ls,page_ls,subq_ls,psg_ls,webnote_fill_page_template->prompt_ls"
)
def webnote_fill_page(
    q_ls: List[str],
    plan_ls: List[str],
    page_ls: List[str],
    subq_ls: List[str],
    psg_ls: List[Any],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for WebNote page filling.

    Args:
        q_ls: List of questions
        plan_ls: List of plans (one per question)
        page_ls: List of page contents (one per question)
        subq_ls: List of sub-questions (one per question)
        psg_ls: List of passages (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, plan, page, subq, psg in zip(q_ls, plan_ls, page_ls, subq_ls, psg_ls):
        p = _safe_render(
            template, question=q, plan=plan, sub_question=subq, docs_text=psg, page=page
        )
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,page_ls,webnote_gen_answer_template->prompt_ls")
def webnote_gen_answer(
    q_ls: List[str],
    page_ls: List[str],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for WebNote answer generation.

    Args:
        q_ls: List of questions
        page_ls: List of page contents (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, page in zip(q_ls, page_ls):
        p = _safe_render(template, page=page, question=q)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="prompt_ls,ans_ls,ret_psg,search_r1_gen_template->prompt_ls")
def search_r1_gen(
    prompt_ls: List[PromptMessage],
    ans_ls: List[str],
    ret_psg: List[Union[str, Any]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for search-r1 pipeline.

    Args:
        prompt_ls: List of previous prompts
        ans_ls: List of previous answers
        ret_psg: List of retrieved passages (one per prompt)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    for prompt, ans, psg in zip(prompt_ls, ans_ls, ret_psg):
        passages = psg[:3]
        passage_text = "\n".join(passages)
        _pro = prompt.content.text
        p = _safe_render(template, history=_pro, answer=ans, passages=passage_text)
        ret.append(p)
    return ret


@app.prompt(output="prompt_ls,ans_ls,ret_psg,r1_searcher_gen_template->prompt_ls")
def r1_searcher_gen(
    prompt_ls: List[PromptMessage],
    ans_ls: List[str],
    ret_psg: List[Union[str, Any]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for r1_searcher pipeline.

    Args:
        prompt_ls: List of previous prompts
        ans_ls: List of previous answers
        ret_psg: List of retrieved passages (one per prompt)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    for prompt, ans, psg in zip(prompt_ls, ans_ls, ret_psg):
        passages = psg[:5]
        passage_text = "\n".join(passages)
        _pro = prompt.content.text
        p = _safe_render(template, history=_pro, answer=ans, passages=passage_text)
        ret.append(p)
    return ret


@app.prompt(output="q_ls,searcho1_reasoning_template->prompt_ls")
def search_o1_init(
    q_ls: List[str],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for Search O1 initialization.

    Args:
        q_ls: List of questions
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)

    ret = []
    for q in q_ls:
        p = _safe_render(template, question=q)
        ret.append(p)
    return ret


@app.prompt(
    output="extract_query_list, ret_psg, total_reason_list, searcho1_refine_template -> prompt_ls"
)
def search_o1_reasoning_indocument(
    extract_query_list: List[str],
    ret_psg: List[List[str]],
    total_reason_list: List[List[str]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for Search O1 reasoning in document step.

    Args:
        extract_query_list: List of extracted search queries
        ret_psg: List of retrieved passage lists (one per query)
        total_reason_list: List of reasoning history lists (one per query)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []

    for squery, psg_list, history_steps in zip(
        extract_query_list, ret_psg, total_reason_list
    ):
        passage_text = "\n".join(psg_list)

        if len(history_steps) <= 3:
            selected_history = history_steps[:]
        else:
            selected_history = [history_steps[0]] + history_steps[-3:]

        formatted_history_parts = [
            f"Step {i}: {reason}" for i, reason in enumerate(selected_history, 1)
        ]
        formatted_history_str = "\n\n".join(formatted_history_parts)

        p = _safe_render(
            template,
            prev_reasoning=formatted_history_str,
            search_query=squery,
            document=passage_text,
        )
        ret.append(p)

    return ret


@app.prompt(
    output="q_ls,total_subq_list,total_final_info_list,searcho1_reasoning_template->prompt_ls"
)
def search_o1_insert(
    q_ls: List[str],
    total_subq_list: List[List[str]],
    total_final_info_list: List[List[str]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for Search O1 by inserting sub-queries and results.

    Args:
        q_ls: List of questions
        total_subq_list: List of sub-query lists (one per question)
        total_final_info_list: List of final info lists (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    prompt_ls = []
    for q in q_ls:
        p = _safe_render(template, question=q)
        prompt_ls.append(p)

    ret = []
    for prompt, sub_queries, sub_reasons in zip(
        prompt_ls, total_subq_list, total_final_info_list
    ):
        for query, reason in zip(sub_queries, sub_reasons):
            part = (
                "<|begin_search_query|>"
                + str(query)
                + "<|end_search_query|>"
                + "\n"
                + "<|begin_search_result|>"
                + str(reason)
                + "<|end_search_result|>"
            )
            prompt += part

        ret.append(prompt)

    return ret


@app.prompt(output="q_ls,ret_psg,gen_subq_template->prompt_ls")
def gen_subq(
    q_ls: List[str],
    ret_psg: List[Union[str, Any]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for sub-question generation (loop and branch demo).

    Args:
        q_ls: List of questions
        ret_psg: List of retrieved passages (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = _safe_render(template, question=q, documents=passage_text)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,ret_psg,check_psg_template->prompt_ls")
def check_passages(
    q_ls: List[str],
    ret_psg: List[Union[str, Any]],
    template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for passage checking (loop and branch demo).

    Args:
        q_ls: List of questions
        ret_psg: List of retrieved passages (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = _safe_render(template, question=q, documents=passage_text)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,ret_psg,evisrag_template->prompt_ls")
def evisrag_vqa(
    q_ls: List[str], ret_psg: List[Union[str, Any]], template: Union[str, Path]
) -> List[PromptMessage]:
    """Generate prompts for EVisRAG visual question answering.

    Args:
        q_ls: List of questions
        ret_psg: List of image path lists (one per question)
        template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(template)
    ret = []
    for q, psg in zip(q_ls, ret_psg):
        p = _safe_render(template, question=q)
        p = p.replace("<image>", "<image>" * len(psg))
        ret.append(p)
    return ret


# ==================== SurveyCPM Prompts ====================


def _abbr_one_line(
    string: Union[str, Dict[str, Any]], abbr: bool = True, tokenizer: Any = None
) -> str:
    """Abbreviate content to one line for SurveyCPM prompts.

    Args:
        string: String or dict to abbreviate
        abbr: Whether to abbreviate long content
        tokenizer: Optional tokenizer for length checking

    Returns:
        One-line string representation
    """
    if isinstance(string, dict):
        if "content" in string and string["content"]:
            return _abbr_one_line(string["content"], abbr=abbr, tokenizer=tokenizer)
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
                    decoded_prefix = tokenizer.decode(
                        tokens.input_ids[0][:100], skip_special_tokens=True
                    )
                    decoded_suffix = tokenizer.decode(
                        tokens.input_ids[0][-50:], skip_special_tokens=True
                    )
                    decoded = decoded_prefix + " ... " + decoded_suffix
                    return "[OK] " + decoded.replace("\n", " ").strip()
                else:
                    return "[OK] " + string.replace("\n", " ").strip()
            else:
                return "[OK] " + string.replace("\n", " ").strip()


def _to_one_line(string: Union[str, Dict[str, Any]]) -> str:
    """Convert content to one line for SurveyCPM prompts.

    Args:
        string: String or dict to convert

    Returns:
        One-line string representation
    """
    if isinstance(string, dict):
        if "content" in string:
            if not string["content"]:
                return ""
            return "[OK] " + string["content"].replace("\n", " ").strip()
        elif "plan" in string:
            return "[PLAN] " + string["plan"].replace("\n", " ").strip()
        else:
            return ""
    if not string:
        return ""
    else:
        return string.replace("\n", " ")


def _check_progress_postion(current_survey: Dict[str, Any]) -> Optional[str]:
    """Check the current progress position in the survey.

    Args:
        current_survey: Survey dictionary

    Returns:
        Position string (e.g., "outline", "section-1", "section-1.2") or None
    """
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
                            for k, subsubsection in enumerate(
                                subsection["subsections"]
                            ):
                                if "content" not in subsubsection:
                                    return f"section-{i+1}.{j+1}.{k+1}"
    return None


def _check_progress_postion_last_detail(current_survey: Dict[str, Any]) -> str:
    """Check the last completed position with detail.

    Args:
        current_survey: Survey dictionary

    Returns:
        Position string of the last completed section
    """
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
                            for k, subsubsection in enumerate(
                                subsection["subsections"]
                            ):
                                if "content" not in subsubsection:
                                    return titles[-1]
                                else:
                                    titles.append(f"section-{i+1}.{j+1}.{k+1}")
    return titles[-1]


def _print_tasknote(current_survey: dict, abbr: bool = True) -> str:
    """Print survey structure as a formatted string.

    Args:
        current_survey: Survey dictionary
        abbr: Whether to abbreviate content

    Returns:
        Formatted string representation of survey
    """
    string = ""
    if current_survey == {}:
        return "There is no survey."

    # Title
    try:
        content = _abbr_one_line(current_survey["title"], abbr=False)
        string += f"# Title: {content}\n"
    except Exception:
        string += f"# Title: None\n"

    to_line_func = _abbr_one_line

    if "sections" in current_survey:
        for i, section in enumerate(current_survey["sections"]):
            title_key = "name" if "name" in section else "title"
            name, content = section[title_key], to_line_func(section, abbr)
            string += f"# Section-{i+1} [{name}]: {content}\n"

            if "subsections" in section:
                for j, subsection in enumerate(section["subsections"]):
                    name, content = subsection[title_key], to_line_func(
                        subsection, abbr
                    )
                    string += f"    ## Section-{i+1}.{j+1} [{name}]: {content}\n"

                    if "subsections" in subsection:
                        for k, subsubsection in enumerate(subsection["subsections"]):
                            name, content = subsubsection[title_key], to_line_func(
                                subsubsection, abbr
                            )
                            string += f"        ### Section-{i+1}.{j+1}.{k+1} [{name}]: {content}\n"

    return string


def _print_tasknote_hire(
    current_survey: Dict[str, Any], last_detail: bool = False
) -> str:
    """Print survey structure with hierarchical detail.

    Args:
        current_survey: Survey dictionary
        last_detail: Whether to use last detail mode

    Returns:
        Formatted string representation of survey with hierarchical detail
    """
    string = ""
    if current_survey == {}:
        return "There is no survey."

    # Title
    try:
        content = _abbr_one_line(current_survey["title"], abbr=False)
        string += f"# Title: {content}\n"
    except Exception:
        string += f"# Title: None\n"

    # Sections
    if last_detail:
        now_section = _check_progress_postion_last_detail(current_survey)
    else:
        now_section = _check_progress_postion(current_survey)

    now_hire = now_section.count(".") if now_section else 0

    if "sections" in current_survey:
        for i, section in enumerate(current_survey["sections"]):
            title_key = "name" if "name" in section else "title"
            if now_section and (
                now_hire == 0
                or (now_section.startswith(f"section-{i+1}") and now_hire == 1)
            ):
                to_line_func = _to_one_line
            else:
                to_line_func = _abbr_one_line
            name, content = section[title_key], to_line_func(section)
            string += f"# Section-{i+1} [{name}]: {content}\n"

            if "subsections" in section:
                for j, subsection in enumerate(section["subsections"]):
                    if now_section and (
                        (now_section.startswith(f"section-{i+1}") and now_hire == 1)
                        or (
                            now_section.startswith(f"section-{i+1}.{j+1}")
                            and now_hire == 2
                        )
                    ):
                        to_line_func = _to_one_line
                    else:
                        to_line_func = _abbr_one_line

                    name, content = subsection[title_key], to_line_func(subsection)
                    string += f"    ## Section-{i+1}.{j+1} [{name}]: {content}\n"

                    if "subsections" in subsection:
                        for k, subsubsection in enumerate(subsection["subsections"]):
                            if now_section and now_section.startswith(
                                f"section-{i+1}.{j+1}"
                            ):
                                to_line_func = _to_one_line
                            else:
                                to_line_func = _abbr_one_line

                            name, content = subsubsection[title_key], to_line_func(
                                subsubsection
                            )
                            string += f"        ### Section-{i+1}.{j+1}.{k+1} [{name}]: {content}\n"

    return string


@app.prompt(
    output="instruction_ls,survey_ls,cursor_ls,surveycpm_search_template->prompt_ls"
)
def surveycpm_search(
    instruction_ls: List[str],
    survey_ls: List[str],
    cursor_ls: List[Optional[str]],
    surveycpm_search_template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for SurveyCPM search step.

    Args:
        instruction_ls: List of user instructions
        survey_ls: List of survey JSON strings
        cursor_ls: List of cursor positions
        surveycpm_search_template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    import json

    template: Template = load_prompt_template(surveycpm_search_template)
    ret = []
    for instruction, survey_json, cursor in zip(instruction_ls, survey_ls, cursor_ls):
        survey = (
            json.loads(survey_json) if survey_json and survey_json != "<PAD>" else {}
        )

        if not survey:
            survey_str = "There is no survey."
        else:
            survey_str = _print_tasknote(survey, abbr=True)

        p = _safe_render(
            template,
            user_query=instruction,
            current_outline=survey_str,
            current_instruction=f"You need to update {cursor}",
        )
        ret.append(p)
    return ret


@app.prompt(
    output="instruction_ls,retrieved_info_ls,surveycpm_init_plan_template->prompt_ls"
)
def surveycpm_init_plan(
    instruction_ls: List[str],
    retrieved_info_ls: List[str],
    surveycpm_init_plan_template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for SurveyCPM initial plan step.

    Args:
        instruction_ls: List of user instructions
        retrieved_info_ls: List of retrieved information strings
        surveycpm_init_plan_template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    template: Template = load_prompt_template(surveycpm_init_plan_template)
    ret = []
    for instruction, retrieved_info in zip(instruction_ls, retrieved_info_ls):
        info = retrieved_info if retrieved_info != "<PAD>" else ""
        p = _safe_render(template, user_query=instruction, current_information=info)
        ret.append(p)
    return ret


@app.prompt(
    output="instruction_ls,survey_ls,cursor_ls,retrieved_info_ls,surveycpm_write_template->prompt_ls"
)
def surveycpm_write(
    instruction_ls: List[str],
    survey_ls: List[str],
    cursor_ls: List[Optional[str]],
    retrieved_info_ls: List[str],
    surveycpm_write_template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for SurveyCPM write step.

    Args:
        instruction_ls: List of user instructions
        survey_ls: List of survey JSON strings
        cursor_ls: List of cursor positions
        retrieved_info_ls: List of retrieved information strings
        surveycpm_write_template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    import json

    template: Template = load_prompt_template(surveycpm_write_template)
    ret = []
    for instruction, survey_json, cursor, retrieved_info in zip(
        instruction_ls, survey_ls, cursor_ls, retrieved_info_ls
    ):
        survey = (
            json.loads(survey_json) if survey_json and survey_json != "<PAD>" else {}
        )
        info = retrieved_info if retrieved_info != "<PAD>" else ""
        survey_str = _print_tasknote_hire(survey, last_detail=True)
        p = _safe_render(
            template,
            user_query=instruction,
            current_survey=survey_str,
            current_instruction=f"You need to update {cursor}",
            current_information=info,
        )
        ret.append(p)
    return ret


@app.prompt(output="instruction_ls,survey_ls,surveycpm_extend_plan_template->prompt_ls")
def surveycpm_extend_plan(
    instruction_ls: List[str],
    survey_ls: List[str],
    surveycpm_extend_plan_template: Union[str, Path],
) -> List[PromptMessage]:
    """Generate prompts for SurveyCPM extend plan step.

    Args:
        instruction_ls: List of user instructions
        survey_ls: List of survey JSON strings
        surveycpm_extend_plan_template: Path to Jinja2 template file

    Returns:
        List of PromptMessage objects
    """
    import json

    template: Template = load_prompt_template(surveycpm_extend_plan_template)

    ret = []
    for instruction, survey_json in zip(instruction_ls, survey_ls):
        survey = (
            json.loads(survey_json) if survey_json and survey_json != "<PAD>" else {}
        )
        survey_str = _print_tasknote(survey, abbr=False)
        p = _safe_render(template, user_query=instruction, current_survey=survey_str)
        ret.append(p)
    return ret


if __name__ == "__main__":
    app.run(transport="stdio")
