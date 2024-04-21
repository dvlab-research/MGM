from .language_model.mgm_llama import MGMLlamaForCausalLM
try:
    from .language_model.mgm_mistral import MGMMistralForCausalLM
    from .language_model.mgm_mixtral import MGMMixtralForCausalLM
    from .language_model.mgm_gemma import MGMGemmaForCausalLM
except:
    ImportWarning("New model not imported. Try to update Transformers.")