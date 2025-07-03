from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model
translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Load tokenizer with use_fast=False to use the slow tokenizer (supports lang_code_to_id)
translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=False)
lang_codes = {
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "malayalam": "mal_Mlym",
    "kannada": "kan_Knda",
    "gujarati": "guj_Gujr",
    "marathi": "mar_Deva",
    "hindi": "hin_Deva",
    "english": "eng_Latn"
}
def translate_text(text, target_lang="tam_Taml"):
    tokenizer = translator_tokenizer
    model = translator_model

    tokenizer.src_lang = "eng_Latn"  # Source language is always English summary

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Get the token ID for the target language
    lang_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    # Translate
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=lang_token_id,
        max_length=128
    )

    # Decode output
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated_text
# Sample review
review_text = """
This smartwatch has a beautiful display and great battery, but the strap is uncomfortable for long use.
"""

# Step 1: Summarize
summary = summarize_review(review_text)
print("Summary in English:", summary)

# Step 2: Translate
translated_summary = translate_text(summary, lang_codes["tamil"])
print("Summary in Tamil:", translated_summary)
