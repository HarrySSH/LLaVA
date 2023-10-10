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
    def __init__(self,  logic_classifier, *args, **kwargs,):
        super().__init__(*args, **kwargs,)
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

        def find_invalid_token_ids(input_ids, tokenizer):
            vocab_size = 25224#tokenizer.vocab.vocab_size
            invalid_ids = []
            for i, token_id in enumerate(input_ids):
                if token_id <0 in enumerate(input_ids):
                    invalid_ids.append((i, token_id))
            return invalid_ids
        
        

        
        # save the inputs['input_ids'][0] as .pt
        

        #outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        
        
        
        #print("out of curiosity how many sentences are in the input IDs")
        #print(len(inputs['input_ids']))  # I found all of you, mother fucker!


        #print('***********************')
        #print('Try output')
        
        
        outputs = model(**inputs)
        '''
        print('The dimention of output logits:')
        print(outputs.logits.shape)
        '''
        # try to convert it the sentence
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Select the token with the highest probability or sample from the distribution  
        
        
        selected_tokens = torch.argmax(probs, dim=-1)  # Use torch.multinomial(probs, num_samples=1) for sampling  
        
        # Convert the selected tokens into words  
        output_text = self.tokenizer.decode(selected_tokens[0], skip_special_tokens=False)
        
        
        # resg
        decoded_Lists = output_text.split('▶')
        print(len(decoded_Lists))
        assert len(decoded_Lists) ==11, 'there should be 11 elements'
        # 1,3,5,7,9 is what I need
        sentences = [x for x in [decoded_Lists[x] for x in [1,3,5,7,9]]]

        new_docs = [self.logtic_classifier_model['nlp']('\n'.join(cohort)) for cohort in sentences] 

        new_entities = [' '.join([ent.text for ent in doc.ents]) for doc in new_docs]  
        new_dependencies = [' '.join([token.dep_ for token in doc]) for doc in new_docs]  
        
        new_X_entities = self.logtic_classifier_model['entities_vectorizer'].transform(new_entities)  

        new_X_dependencies = self.logtic_classifier_model['dependencies_vectorizer'].transform(new_dependencies)  
        
        new_semantic_similarity = [self['doc2vec_model'].infer_vector([word for sentence in cohort for word in sentence.split()]) for cohort in sentences] 
            
        import numpy as np
        new_X = np.hstack((new_semantic_similarity, new_X_entities.toarray(), new_X_dependencies.toarray()))  
        
        new_y_pred =self['clf'].predict(new_X)  


        #print('***********************')
        if new_y_pred[0] == 1:
            print('The logic is consistent')
        else:
            print('The logic is inconsistent')
        

        

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
