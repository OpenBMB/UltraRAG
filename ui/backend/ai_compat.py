"""Token budget spelling for known reasoning models and compatible endpoints."""


def completion_limit(model: str, budget: int) -> dict[str, int]:
    family = model.lower().removeprefix("openai/")
    field = (
        "max_completion_tokens"
        if family.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
        else "max_tokens"
    )
    return {field: budget}
