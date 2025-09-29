from langchain_huggingface import HuggingFacePipeline

class HuggingFacePipeline(HuggingFacePipeline):
    """HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation`, `text2text-generation`, `image-text-to-text`,
    `summarization` and `translation`  for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain_huggingface import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 10},
            )
    Example passing pipeline in directly:
        .. code-block:: python

            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
            )
            hf = HuggingFacePipeline(pipeline=pipe)
    """