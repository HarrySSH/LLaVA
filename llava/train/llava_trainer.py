import os
import torch

from transformers import Trainer
from typing import Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class LLaVATrainer(Trainer):
    def __init__(self,logic_classifier, *args, **kwargs):
        self.logtic_classifier_model = logic_classifier
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Not the default any more, I already fucked it up!
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # make the codes stop here so that I can keep understadning it
        # print the key values for inputs
        

        #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print(len(inputs['input_ids']))
        decoded_text = self.tokenizer.decode(inputs['input_ids'][0])
        #outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        print(decoded_text)

        print("out of curiosity how many sentences are in the input IDs")
        print(len(inputs['input_ids']))  # I found all of you, mother fucker!


        print('***********************')
        print('Try output')
        
        
        outputs = model(**inputs)
        print('The dimention of output logits:')
        print(outputs.logits.shape)
        # try to convert it the sentence
        probs = torch.nn.functional.softmax(outputs.logits
                                            , dim=-1)
        
        # Select the token with the highest probability or sample from the distribution  
        selected_tokens = torch.argmax(probs, dim=-1)  # Use torch.multinomial(probs, num_samples=1) for sampling  
        
        # Convert the selected tokens into words  
        output_text = self.tokenizer.decode(selected_tokens[0]) 
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(output_text)

        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        if 1==1:
            raise Exception("make the codes stop here so that I can keep understadning it")
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss



    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
