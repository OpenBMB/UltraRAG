# Example for ToolCall usage with benchmark and retriever servers

from ultrarag.api import initialize, ToolCall


initialize(["benchmark", "retriever"], server_root="servers") 

benchmark_param_dict = {
    "key_map":{
      "gt_ls": "golden_answers",
      "q_ls": "question"
    },
    "limit": -1,
    "seed": 42,
    "name": "nq",
    "path": "data/sample_nq_10.jsonl",
    
}
benchmark = ToolCall.benchmark.get_data(benchmark_param_dict)

query_list = benchmark['q_ls']


retriever_init_param_dict = {
    "model_name_or_path": "Qwen/Qwen3-Embedding-0.6B",
}

ToolCall.retriever.retriever_init(
    **retriever_init_param_dict
)

result = ToolCall.retriever.retriever_search(
    query_list=query_list,
    top_k=5,
)

retrieve_passages = result['ret_psg']


# Example for PipelineCall usage with rag_deploy.yaml

from ultrarag.api import PipelineCall

result = PipelineCall(
    pipeline_file="examples/rag_deploy.yaml",
    parameter_file="examples/parameter/rag_deploy_parameter.yaml",
)

final_step_result = result['final_result']
all_steps_result = result['all_results']



