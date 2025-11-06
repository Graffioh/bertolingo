"""
Inference and translation functions
"""

import torch


def translate(model, src_text, stoi, itos, device, max_len=128):
    """
    Translate English text to Italian using the trained model
    
    Args:
        model: Trained Seq2Seq model
        src_text: English text string
        stoi: String to index mapping
        itos: Index to string mapping
        device: Device to run on
        max_len: Maximum translation length
    Returns:
        Translated Italian text
    """
    model.eval()
    
    # Encode source text
    src_tokens = torch.tensor([[stoi.get(char, stoi['<PAD>']) for char in src_text]], device=device)
    src_mask = (src_tokens == stoi['<PAD>'])
    
    # Encode source
    with torch.no_grad():
        encoder_out = model.encoder(src_tokens, key_padding_mask=src_mask)
        
        # Start with first token (or padding if we had BOS)
        tgt_tokens = torch.tensor([[stoi.get(src_text[0], stoi['<PAD>'])]], device=device)
        
        # Autoregressive decoding
        for _ in range(max_len):
            tgt_mask = (tgt_tokens == stoi['<PAD>'])
            logits = model.decoder(
                tgt_tokens,
                encoder_out,
                tgt_key_padding_mask=tgt_mask,
                src_key_padding_mask=src_mask
            )
            
            # Get next token (greedy decoding)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
            
            # Stop if padding token (or could add EOS token)
            if next_token.item() == stoi['<PAD>']:
                break
    
    # Decode target tokens
    translated = ''.join([itos.get(idx.item(), '') for idx in tgt_tokens[0]])
    
    return translated

