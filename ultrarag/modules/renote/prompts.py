
RENOTE_PROMPTS = dict()

RENOTE_PROMPTS["define_notes"] = \
'''根据以下提供的文档内容撰写一篇笔记。笔记应整合所有能够帮助回答指定问题的相关原文信息，并形成一段连贯的文本。请确保笔记包含所有对回答问题有用的原文信息。

需要回答的问题： {query}
文档内容： {refs}

请提供你撰写的笔记：
'''


RENOTE_PROMPTS["gen_new_query"] = \
'''任务：根据笔记，提出两个新问题。这些新问题将用于检索文档，以补充笔记并帮助回答原始问题。新问题应简明扼要，并包括有利于检索的关键词。新问题应避免与现有问题列表重复。
输出格式：
1. 问题1？
2. 问题2？

原始问题：{query}
笔记：{note}

现有问题列表：{query_log}

提供两个新问题：
'''


RENOTE_PROMPTS["refine_notes"] = \
'''任务：根据检索到的文档，用尚未包含但对回答问题有用的内容补充笔记。补充内容应使用检索到的文档中的原始文本，并尽可能多的包含检索到的文档的信息。请确保笔记简洁凝练

问题：{query}
检索到的文档：{refs}

笔记：{note}

提供补充后的笔记：
'''


RENOTE_PROMPTS["update_notes"] = \
'''任务：请你帮我判断提供的哪个笔记更好，具体评价标准如下：
1. 包含与问题直接相关的关键信息。
2. 信息的全面性：是否涵盖了所有相关的方面和细节。
3. 信息的细节程度：是否提供了足够的细节来深入理解问题。
4. 实用性：笔记是否提供了实际的帮助和解决方案。

请严格按照以下要求进行判断：
- 如果笔记2没有在笔记1的基础上增加新的有意义的内容，或者只是增加了多余的信息，请直接返回: {{"status": false}}。
- 如果笔记2比笔记1在上述标准上有明显改进，请直接返回: {{"status": true}}，否则请直接返回: {{"status": false}}。

问题：{query}
提供的笔记1：{best_note}
提供的笔记2：{new_note}


请根据以上信息进行判断，不要解释，直接返回结果。
'''


RENOTE_PROMPTS["answer_by_notes"] = \
'''你是一位文字回复专家，尤其擅长根据笔记措辞回复，现在，你被提供了1个问题，和与问题相关的笔记。请你结合笔记和你自身的知识回答这些问题。
请在回答时遵循一个规则：如果问题是中文，请用中文回答；如果问题是英文，请用英文回答；务必遵循这条规则。

问题：{query}

与问题相关的笔记：{note}

请给出你的回答：
'''
