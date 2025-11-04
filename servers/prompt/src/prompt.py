import os
import string
from pathlib import Path
from typing import Any, List

from jinja2 import Template

from fastmcp.prompts import PromptMessage
from ultrarag.server import UltraRAG_MCP_Server


app = UltraRAG_MCP_Server("prompt")


def load_prompt_template(template_path: str | Path) -> Template:
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    return Template(template_content)


# prompt for QA boxed
@app.prompt(output="q_ls,template->prompt_ls")
def qa_boxed(q_ls: List[str], template: str | Path) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for q in q_ls:
        p = template.render(question=q)
        ret.append(p)
    return ret


# prompt for Multiple Choice QA boxed
@app.prompt(output="q_ls,choices_ls,template->prompt_ls")
def qa_boxed_multiple_choice(
    q_ls: List[str],
    choices_ls: List[List[str]],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    CHOICES: List[str] = list(string.ascii_uppercase)  # A, B, ..., Z
    for q, choices in zip(q_ls, choices_ls):
        choices_text = "\n".join(f"{CHOICES[i]}: {c}" for i, c in enumerate(choices))
        p = template.render(question=q, choices=choices_text)
        ret.append(p)
    return ret


# prompt for QA RAG boxed
@app.prompt(output="q_ls,ret_psg,template->prompt_ls")
def qa_rag_boxed(
    q_ls: List[str], ret_psg: List[str | Any], template: str | Path
) -> list[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = template.render(question=q, documents=passage_text)
        ret.append(p)
    return ret


# prompt for QA RAG boxed with multiple choice
@app.prompt(output="q_ls,choices_ls,ret_psg,template->prompt_ls")
def qa_rag_boxed_multiple_choice(
    q_ls: List[str],
    choices_ls: List[List[str]],
    ret_psg: List[List[str]],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    CHOICES: List[str] = list(string.ascii_uppercase)  # A, B, ..., Z
    for q, psg, choices in zip(q_ls, ret_psg, choices_ls):
        passage_text = "\n".join(psg)
        choices_text = "\n".join(f"{CHOICES[i]}: {c}" for i, c in enumerate(choices))
        p = template.render(question=q, documents=passage_text, choices=choices_text)
        ret.append(p)
    return ret


# prompt for RankCoT
@app.prompt(output="q_ls,ret_psg,kr_template->prompt_ls")
def RankCoT_kr(
    q_ls: List[str],
    ret_psg: List[str | Any],
    template: str | Path,
) -> list[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = template.render(question=q, documents=passage_text)
        ret.append(p)
    return ret


@app.prompt(output="q_ls,kr_ls,qa_template->prompt_ls")
def RankCoT_qa(
    q_ls: List[str],
    kr_ls: List[str],
    template: str | Path,
) -> list[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for q, cot in zip(q_ls, kr_ls):
        p = template.render(question=q, CoT=cot)
        ret.append(p)
    return ret


# prompt for IRCOT
@app.prompt(output="memory_q_ls,memory_ret_psg,template->prompt_ls")
def ircot_next_prompt(
    memory_q_ls: List[List[str | None]],
    memory_ret_psg: List[List[List[str]] | None],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret: List[PromptMessage] = []
    # ---------- single turn ----------
    if len(memory_q_ls) == 1:
        for q, psg in zip(memory_q_ls[0], memory_ret_psg[0]):  # type: ignore[arg-type]
            if q is None:
                continue
            passage_text = "" if psg is None else "\n".join(psg)
            ret.append(
                template.render(documents=passage_text, question=q, cur_answer="")
            )
        return ret
    # ---------- multi turn ----------
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
            template.render(documents=passage_text, question=q, cur_answer=cur_answer)
        )
    return ret


# prompt for WebNote
@app.prompt(output="q_ls,plan_ls,webnote_init_page_template->prompt_ls")
def webnote_init_page(
    q_ls: List[str],
    plan_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, plan in zip(q_ls, plan_ls):
        p = template.render(question=q, plan=plan)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,webnote_gen_plan_template->prompt_ls")
def webnote_gen_plan(
    q_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q in q_ls:
        p = template.render(question=q)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,plan_ls,page_ls,webnote_gen_subq_template->prompt_ls")
def webnote_gen_subq(
    q_ls: List[str],
    plan_ls: List[str],
    page_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, plan, page in zip(q_ls, plan_ls, page_ls):
        p = template.render(question=q, plan=plan, page=page)
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
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, plan, page, subq, psg in zip(q_ls, plan_ls, page_ls, subq_ls, psg_ls):
        p = template.render(question=q, plan=plan, sub_question=subq, docs_text=psg, page=page)
        all_prompts.append(p)
    return all_prompts


@app.prompt(output="q_ls,page_ls,webnote_gen_answer_template->prompt_ls")
def webnote_gen_answer(
    q_ls: List[str],
    page_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, page in zip(q_ls, page_ls):
        p = template.render(page=page, question=q)
        all_prompts.append(p)
    return all_prompts


# prompt for search-r1
@app.prompt(output="prompt_ls,ans_ls,ret_psg,search_r1_gen_template->prompt_ls")
def search_r1_gen(
    prompt_ls: List[PromptMessage],
    ans_ls: List[str],
    ret_psg: List[str | Any],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for prompt, ans, psg in zip(prompt_ls, ans_ls, ret_psg):
        # passages = [psg[index]["segment"] for index in range(min(5, len(psg)))]
        passages = psg[:3]
        passage_text = "\n".join(passages)
        _pro = prompt.content.text
        p = template.render(history=_pro, answer=ans, passages=passage_text)
        ret.append(p)
    return ret


# prompt for r1_searcher
@app.prompt(output="prompt_ls,ans_ls,ret_psg,r1_searcher_gen_template->prompt_ls")
def r1_searcher_gen(
    prompt_ls: List[PromptMessage],
    ans_ls: List[str],
    ret_psg: List[str | Any],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for prompt, ans, psg in zip(prompt_ls, ans_ls, ret_psg):
        # passages = [psg[index]["segment"] for index in range(min(5, len(psg)))]
        passages = psg[:5]
        passage_text = "\n".join(passages)
        _pro = prompt.content.text
        p = template.render(history=_pro, answer=ans, passages=passage_text)
        ret.append(p)
    return ret


# prompt for search-o1
@app.prompt(output="q_ls,searcho1_reasoning_template->prompt_ls")
def search_o1_init(
    q_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)

    ret = []
    for q in q_ls:
        p = template.render(question=q)
        ret.append(p)
    return ret


@app.prompt(
    output="prompt_ls,extract_query_list,ret_psg,searcho1_refine_template->prompt_ls"
)
def searcho1_reasoning_indocument(
    prompt_ls: List[PromptMessage],
    extract_query_list: List[str],
    ret_psg: List[str | Any],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for prompt, squery, psg in zip(prompt_ls, extract_query_list, ret_psg):
        passage_text = "\n".join(psg)
        _pro = prompt.content.text
        p = template.render(
            prev_reasoning=_pro, search_query=squery, document=passage_text
        )
        ret.append(p)
    return ret


@app.prompt(output="prompt_ls,ans_ls->prompt_ls")
def search_o1_insert(
    prompt_ls: List[PromptMessage],
    ans_ls: List[str],
) -> List[PromptMessage]:
    ret = []
    for prompt, ans in zip(prompt_ls, ans_ls):
        _pro = prompt.content.text
        p = _pro + "<|begin_search_result|>" + ans + "<|end_search_result|>"
        ret.append(p)
    return ret


# prompt for loop and branch demo
@app.prompt(output="q_ls,ret_psg,gen_subq_template->prompt_ls")
def gen_subq(
    q_ls: List[str],
    ret_psg: List[str | Any],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = template.render(question=q, documents=passage_text)
        all_prompts.append(p)
    return all_prompts

@app.prompt(output="q_ls,ret_psg,check_psg_template->prompt_ls")
def check_passages(
    q_ls: List[str],
    ret_psg: List[str | Any],
    template: str | Path,
) -> List[PromptMessage]:
    template: Template = load_prompt_template(template)
    all_prompts = []
    for q, psg in zip(q_ls, ret_psg):
        passage_text = "\n".join(psg)
        p = template.render(question=q, documents=passage_text)
        all_prompts.append(p)
    return all_prompts


# prompt for EVisRAG
@app.prompt(output="q_ls,ret_psg,evisrag_template->prompt_ls")
def evisrag_vqa(
    q_ls: List[str], ret_psg: List[str | Any], template: str | Path
) -> list[PromptMessage]:
    template: Template = load_prompt_template(template)
    ret = []
    for q, psg in zip(q_ls, ret_psg):
        p = template.render(question=q)
        p = p.replace('<image>', '<image>' * len(psg))
        ret.append(p)
    return ret

if __name__ == "__main__":
    app.run(transport="stdio")
