from transformers import StoppingCriteria
import torch

class GPT2SingleWordStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria for GPT-2 that stops generation immediately after
    completing a single word. Uses GPT-2's tokenizer characteristic where 'Ġ' (U+0120)
    indicates word boundaries.
    """
    def __init__(self, tokenizer, start_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.start_length = start_length
        # Special GPT-2 word boundary token
        self.word_boundary = ' '
        # Track if we've started a new word
        self.started_word = False
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check the most recently generated text
        if len(input_ids[0]) <= self.start_length:
            return False
            
        # Get the raw token strings for the generated sequence
        tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0][self.start_length:]]
        
        # If we haven't generated anything yet, continue
        if not tokens:
            return False
            
        for token in tokens:
            
            # If we see a word boundary token and we've already started a word,
            # that means we've completed a word
            
            if self.word_boundary in token and self.started_word:
                print(token)
                return True   

            if self.word_boundary in token:
                self.started_word = True
                
        # Also stop if we hit end punctuation
        if tokens[-1].strip() in ['.', '!', '?', '\n']:
            return True
            
        return False

def generate_with_word_stopping(model, tokenizer, prompt, max_length=50, **kwargs):
    """
    Generate text using GPT-2, stopping after completing exactly one word.
    
    Args:
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        prompt: The input prompt to generate from
        max_length: Maximum length of the generated text
        **kwargs: Additional arguments to pass to model.generate()
    
    Returns:
        Generated text containing exactly one word
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    start_length = len(inputs.input_ids[0])
    
    stopping_criteria = GPT2SingleWordStoppingCriteria(tokenizer, start_length)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        stopping_criteria=[stopping_criteria],
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    # print the transition scores as probabilities
    print(transition_scores)
    # decode each sequence
    outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    return outputs

# Example usage
#tokenizer.decode(outputs[0][start_length:], skip_special_tokens=True)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "The person that Mary thinks that John said that Larry spoke with"
outputs = generate_with_word_stopping(
    model, 
    tokenizer, 
    prompt,
    max_length=50,
    num_beams = 10,
    num_return_sequences=10,
    output_scores=True,
    return_dict_in_generate=True,
    do_sample=False,
)

print(outputs)  # Should print exactly one word
