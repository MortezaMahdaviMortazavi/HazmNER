import config

def post_process_predictions(predictions, original_texts, original_tags):
    processed_predictions = []
    
    for pred, text, tags in zip(predictions, original_texts, original_tags):
        processed_prediction = []
        current_entity = []
        current_label = None

        for token, tag in zip(text, tags):
            sub_tokens = config.TOKENIZER.tokenize(token)
            sub_tokens_len = len(sub_tokens)

            # If the token is the start of a new entity
            if tag.startswith("B-"):
                if current_label is not None:
                    processed_prediction.append((current_label, " ".join(current_entity)))
                current_label = tag[2:]
                current_entity = sub_tokens
            # If the token is part of an ongoing entity
            elif tag.startswith("I-"):
                current_entity.extend(sub_tokens)
            # If the token is outside any entity
            else:
                if current_label is not None:
                    processed_prediction.append((current_label, " ".join(current_entity)))
                current_label = None
                current_entity = []

        # Handle the last entity if necessary
        if current_label is not None:
            processed_prediction.append((current_label, " ".join(current_entity)))

        processed_predictions.append(processed_prediction)

    return processed_predictions



"""
This post-processing function takes the model predictions (which are at the sub-token level) and the original texts and tags. It then reconstructs the entities from the sub-tokens and their corresponding labels. The function uses the BIO (Beginning, Inside, Outside) tagging scheme commonly used in NER tasks.

Here's a breakdown of what happens in the post-processing function:

For each example in the dataset:

Iterate over the original tokens, their tags, and the model predictions.
Track the current entity and its label as you iterate through the tokens.
When encountering a "B-" tag, start a new entity.
When encountering an "I-" tag, extend the current entity.
When encountering an "O" tag, add the current entity if it exists.
The function returns a list of processed predictions for each example in the dataset.

Remember to adapt the function if your specific use case requires a different tagging scheme or handling of special tokens.
"""