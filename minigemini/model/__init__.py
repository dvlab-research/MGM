from .language_model.mini_gemini_llama import MiniGeminiLlamaForCausalLM
try:
    from .language_model.mini_gemini_mistral import MiniGeminiMistralForCausalLM
    from .language_model.mini_gemini_mixtral import MiniGeminiMixtralForCausalLM
    from .language_model.mini_gemini_gemma import MiniGeminiGemmaForCausalLM
except:
    ImportWarning("New model not imported. Try to update Transformers.")