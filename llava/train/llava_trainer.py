import os
import torch

from transformers import Trainer
from typing import Optional
import copy
#from transformers.src.transformers import generation
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import   StoppingCriteriaList
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


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

import math
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class LLaVATrainer(Trainer):
    def __init__(self,  logic_classifier, *args, **kwargs,):
        super().__init__(*args, **kwargs,)
        self.logtic_classifier_model = logic_classifier

    def evaluate_human_logics(self, model, inputs):
        
        question_list_list = inputs['question_list']
        # assert every question_list_list has 5 questions in
        assert len(question_list_list[0]) == 5, 'every question_list_list has 5 questions in'
        # convert this list of list to a list of strings    
        question_list = [x for y in question_list_list for x in y]

        assert len(question_list) == 160, 'every question_list has 160 questions, but it actually has {}'.format(len(question_list))
        questions = get_chunk(question_list,32,0)
        
        import time
        # record the start time
        start = time.time()
        # record the end time
        p = 0
        disable_torch_init()

        for qs in tqdm(questions):
            p = p + 1
            

            cur_prompt = qs
        
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates['llava_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            # stack two input_ids together
            

            
            print('Generate the output...')
            output_ids = model.generate(
                    input_ids,
                    images=inputs['images'],
                    do_sample=True,
                    temperature=0.2, #args.temperature,
                    top_p=None, #args.top_p,
                    num_beams=1, #args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=30,
                    use_cache=True)
            
            
            
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            #print(f'Input: {qs}')
            #print(f'Output: {outputs}')
        end = time.time()
        print(f'p: {p}')
        # calculate the difference
        print(f'Inference time: {end - start}')

        

    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        current_epoch = (self.state.global_step -1)// 35  # these two numberm ight not exist
        print(self.state.global_step)
        if current_epoch >=0:
            self.evaluate_human_logics(model, inputs)
            
        del inputs['question_list']
        outputs = model(**inputs)
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        
        

        #output_text = self.tokenizer.decode(labels[0], skip_special_tokens=False)
        #shift_logits = outputs['logits'][..., :-1, :].contiguous()
        # how many unique values are there in attention?
        import torch.nn.functional as F

        #selected_tokens =torch.argmax(shift_logits, dim=-1)  # Use torch.multinomial(probs, num_samples=1) for sampling  
        #probabilities = F.softmax(shift_logits[0], dim=-1) 
        #sampled_indices = torch.multinomial(probabilities, num_samples=1)  

        # Convert the selected tokens into words  
        #output_text = self.tokenizer.decode(sampled_indices.squeeze().tolist(), skip_special_tokens=False)
        
        
        
        return (loss, outputs) if return_outputs else loss

        
    def compute_loss_pass(self, model, inputs, return_outputs=False):
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

        
        

        
        # save the inputs['input_ids'][0] as .pt
        

        #outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        
        
        
        #print("out of curiosity how many sentences are in the input IDs")
        #print(len(inputs['input_ids']))  # I found all of you, mother fucker!


        #print('***********************')
        #print('Try output')
        generation_config = None
        

        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if not model.generation_config._from_model_config: #and model.generation_config._original_object_hash == hash(model.generation_config
             
                
                new_generation_config = GenerationConfig.from_model_config(model.config)
                
                model.generation_config = new_generation_config

            generation_config = model.generation_config

        import copy
        generation_config = copy.deepcopy(generation_config)
        
        

        
        kwargs = {'do_sample':True,
                'temperature':0.2,
                'top_p':None,
                'num_beams':1,
                'max_new_tokens':1024,
                'use_cache':True}

        
        
        
        
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        model._validate_model_kwargs(model_kwargs.copy())

        logits_processor = LogitsProcessorList()
        stopping_criteria =  StoppingCriteriaList()

        

        model_inputs = {"input_ids": inputs["input_ids"]}
        model_kwargs = model_inputs
        model_kwargs["attention_mask"] = None

        model_kwargs["output_attentions"] = None #generation_config.output_attentions
        model_kwargs["output_hidden_states"] = None #generation_config.output_hidden_states
        model_kwargs["use_cache"] = True

        

        #print('Loss value: ', loss)
        #print('Loss dimention: ', loss.shape)
        '''
        print('The dimention of output logits:')
        print(outputs.logits.shape)
        '''
        # try to convert it the sentence
        #probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        import copy
        

        

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                pass
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            
            generation_config.pad_token_id = eos_token_id

        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        print(model.config.is_encoder_decoder)
        dagbsd

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                pass
        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")



        ### --------------- me fucking up the models! -----------------###




        outputs = model(**inputs)

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
        
        # Select the token with the highest probability or sample from the distribution  
        
        
        selected_tokens = torch.argmax(probs, dim=-1)  # Use torch.multinomial(probs, num_samples=1) for sampling  
        
        # Convert the selected tokens into words  
        output_text = self.tokenizer.decode(selected_tokens[0], skip_special_tokens=False)
        
        
        # resg
        decoded_Lists = output_text.split('▶')
        #print(len(decoded_Lists))
        
        assert len(decoded_Lists) >=11, 'there should be 11 elements, but there are {}'.format(len(decoded_Lists))
        # 1,3,5,7,9 is what I need
        sentences = [x for x in [decoded_Lists[x] for x in [1,3,5,7,9]]]

        new_docs = [self.logtic_classifier_model['nlp']('\n'.join(cohort)) for cohort in sentences] 

        new_entities = [' '.join([ent.text for ent in doc.ents]) for doc in new_docs]  
        new_dependencies = [' '.join([token.dep_ for token in doc]) for doc in new_docs]  
        
        new_X_entities = self.logtic_classifier_model['entities_vectorizer'].transform(new_entities)  

        new_X_dependencies = self.logtic_classifier_model['dependencies_vectorizer'].transform(new_dependencies)  
        
        new_semantic_similarity = [self.logtic_classifier_model['doc2vec_model'].infer_vector([word for sentence in cohort for word in sentence.split()]) for cohort in sentences] 
            
        import numpy as np
        new_X = np.hstack((new_semantic_similarity, new_X_entities.toarray(), new_X_dependencies.toarray()))  
        
        new_y_pred =self.logtic_classifier_model['clf'].predict(new_X)  
        mlp 


        #print('***********************')
        
        

        
        
        #assert 1==2, 'stop here'
        if new_y_pred[0] == 0: # not align
            #double the loss
            loss = loss * 2
        else:
            pass
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

    def evaluate_human_logics(self, model, inputs):
        
        question_list_list = inputs['question_list']
        # assert every question_list_list has 5 questions in
        assert len(question_list_list[0]) == 5, 'every question_list_list has 5 questions in'
        # convert this list of list to a list of strings    
        question_list = [x for y in question_list_list for x in y]

        assert len(question_list) == 160, 'every question_list has 160 questions, but it actually has {}'.format(len(question_list))
        questions = get_chunk(question_list,32,0)
        
        import time
        # record the start time
        start = time.time()
        # record the end time
        p = 0
        

        for qs in questions:
            p = p + 1
            

            cur_prompt = qs
        
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates['llava_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            

            with torch.inference_mode():
            
                output_ids = model.generate(
                    input_ids,
                    images=inputs['images'],
                    do_sample=True,
                    temperature=0.2, #args.temperature,
                    top_p=None, #args.top_p,
                    num_beams=1, #args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=30,
                    use_cache=True)
            
            
            
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            #print(f'Input: {qs}')
            #print(f'Output: {outputs}')
        end = time.time()
        print(f'p: {p}')
        # calculate the difference
        print(f'Inference time: {end - start}')

        

    
    def compute_loss_backup(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.state.epoch > 0:
            self.evaluate_human_logics(model, inputs)
        del inputs['question_list']
        outputs = model(**inputs)
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        
        

        #output_text = self.tokenizer.decode(labels[0], skip_special_tokens=False)
        #shift_logits = outputs['logits'][..., :-1, :].contiguous()
        # how many unique values are there in attention?
        import torch.nn.functional as F

        #selected_tokens =torch.argmax(shift_logits, dim=-1)  # Use torch.multinomial(probs, num_samples=1) for sampling  
        #probabilities = F.softmax(shift_logits[0], dim=-1) 
        #sampled_indices = torch.multinomial(probabilities, num_samples=1)  

        # Convert the selected tokens into words  
        #output_text = self.tokenizer.decode(sampled_indices.squeeze().tolist(), skip_special_tokens=False)
        
        
        
        return (loss, outputs) if return_outputs else loss

        

        
        

