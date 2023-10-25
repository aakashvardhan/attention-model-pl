import torch_tb_profiler
import torch
from data.dataset import casual_mask



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_token = torch.max(prob, dim=-1) # get the index of the greatest value (greedy approach)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_token.item())], dim=1
        )

        # break if eos token
        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)