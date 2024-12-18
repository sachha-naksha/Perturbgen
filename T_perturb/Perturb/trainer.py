from typing import Any, List

import torch

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity

from T_perturb.Model.trainer import CellGenTrainer, CountDecoderTrainer
from T_perturb.Perturb.T_model import PerturberMasking
from T_perturb.src.utils import return_perturbation_adata  # WarmupScheduler,;


class PerturberTrainer(CellGenTrainer):
    def __init__(
        self,
        perturbation_mode: List[str],
        genes_to_perturb: List[int],
        perturbation_token: int = 0,
        generate: bool = False,
        sequence_length: int = 2048,
        temperature: float = 2.0,
        iterations: int = 18,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.transformer = PerturberMasking(
            tgt_vocab_size=kwargs['tgt_vocab_size'],
            d_model=kwargs['d_model'],
            num_heads=kwargs['num_heads'],
            num_layers=kwargs['num_layers'],
            d_ff=kwargs['d_ff'],
            max_seq_length=kwargs['max_seq_length'],
            dropout=kwargs['dropout'],
            pred_tps=kwargs['pred_tps'],
            context_tps=kwargs['context_tps'],
            n_total_tps=kwargs['n_total_tps'],
            encoder=kwargs['encoder'],
            mask_scheduler=kwargs['mask_scheduler'],
            pos_encoding_mode=kwargs['pos_encoding_mode'],
            return_attn=kwargs['return_attn'],
        )

        if perturbation_mode is not None:
            self.perturbation_mode = perturbation_mode
            self.genes_to_perturb = torch.tensor(genes_to_perturb, dtype=torch.long)
            self.perturbation_token = torch.tensor(
                perturbation_token, dtype=torch.long, device=self.device
            )

        self.generate = generate
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.iterations = iterations

        self.test_dict['cls_cosine_similarity'] = []
        self.test_dict['mean_cosine_similarity'] = []
        self.test_dict['delta_probs'] = []

    def forward(
        self,
        batch: Any,
        perturbation: bool = False,
        generate: bool = False,
    ):
        if perturbation:
            self.genes_to_perturb = self.genes_to_perturb.to(self.device)
            self.perturbation_token = self.perturbation_token.to(self.device)
        tgt_input_id_dict = {}
        for i in self.pred_tps:
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            if perturbation:
                if 'tgt' in self.perturbation_mode:
                    print('perturbating tgt')
                    perturbed_tgt = tgt_input_id_.clone()
                    print('perturbation', self.perturbation_token.device)
                    print('perturbed_tgt', perturbed_tgt.device)
                    mask = torch.isin(tgt_input_id_, self.genes_to_perturb)

                    perturbed_tgt[mask] = self.perturbation_token
                tgt_input_id_dict[f'tgt_input_ids_t{i}'] = perturbed_tgt
            else:
                tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if perturbation:
            if 'src' in self.perturbation_mode:
                print('perturbating src')
                perturbed_src = batch['src_input_ids'].clone()
                mask = torch.isin(batch['src_input_ids'], self.genes_to_perturb)
                perturbed_src[mask] = self.perturbation_token
        else:
            perturbed_src = batch['src_input_ids']
        if generate:
            outputs = None
        else:
            outputs = self.transformer(
                src_input_id=perturbed_src,
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=self.return_embeddings,
            )
        return outputs, perturbed_src, tgt_input_id_dict

    def test_step(self, batch, *args, **kwargs):
        if self.return_embeddings:
            true_outputs, _, _ = self.forward(batch, perturbation=False)
            perturbed_outputs, _, _ = self.forward(batch, perturbation=True)
            for t in self.pred_tps:
                true_cls = true_outputs[t]['dec_embedding'][:, 0, :]
                true_mean_cls = true_outputs[t]['mean_embedding']

                true_logits = true_outputs[t]['dec_logits']
                true_probs = torch.softmax(true_logits, dim=-1)
                true_probs = true_probs.sum(dim=1)

                perturbed_cls = perturbed_outputs[t]['dec_embedding'][:, 0, :]
                perturbed_mean_cls = perturbed_outputs[t]['mean_embedding']
                perturbed_logits = perturbed_outputs[t]['dec_logits']

                perturbed_probs = torch.softmax(perturbed_logits, dim=-1)
                perturbed_probs = perturbed_probs.sum(dim=1)
                delta_probs = perturbed_probs - true_probs
                delta_cls_cos_sim = cosine_similarity(
                    perturbed_cls,
                    true_cls,
                )
                delta_mean_cos_sim = cosine_similarity(
                    perturbed_mean_cls,
                    true_mean_cls,
                )
                delta_cls_cos_sim = delta_cls_cos_sim.detach().cpu()
                delta_mean_cos_sim = delta_mean_cos_sim.detach().cpu()
                cls_embeddings = true_cls.detach().cpu()
                delta_probs = delta_probs.detach().cpu()
                self.test_dict['cls_cosine_similarity'].append(delta_cls_cos_sim)
                self.test_dict['mean_cosine_similarity'].append(delta_mean_cos_sim)
                self.test_dict['cls_embeddings'].append(cls_embeddings)
                self.test_dict['delta_probs'].append(delta_probs)

                # return obs_key
                self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{t}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{t}'])
        if self.generate:
            (
                _,
                pert_src_input_ids,
                tgt_input_id_dict,
            ) = self.forward(batch, perturbation=True)
            decoder_kwargs = {
                'tgt_input_id_dict': tgt_input_id_dict,
                'mask_scheduler': self.mask_scheduler,
                'can_remask_prev_masked': False,
                'topk_filter_thres': 0.9,
                'temperature': self.temperature,
                'iterations': self.iterations,
                'sequence_length': self.sequence_length,
            }

            true_outputs, true_ids_dict = self.transformer.generate(
                src_input_id=batch['src_input_ids'],
                genes_to_perturb=self.genes_to_perturb,
                **decoder_kwargs,
            )
            perturbed_outputs, perturbed_ids_dict = self.transformer.generate(
                src_input_id=pert_src_input_ids,
                **decoder_kwargs,
            )
            for t, tgt_input_id in enumerate(true_ids_dict.keys()):
                print('tgt_input_id', tgt_input_id)
                print('perturbed_outputs', perturbed_outputs.keys())

                pred_ids = perturbed_ids_dict[tgt_input_id].detach().cpu().numpy()
                tgt_ids = true_ids_dict[tgt_input_id].detach().cpu().numpy()
                # compute cosine similarity between perturbed and true
                # t = i + 1
                print('t', t)

                # print('true_outputs', true_outputs.keys())
                # print('true_outputs', true_outputs)
                cls_cos_sim = cosine_similarity(
                    perturbed_outputs[t]['dec_embedding'][:, 0, :],
                    true_outputs[t]['dec_embedding'][:, 0, :],
                )
                mean_agg_cos_sim = cosine_similarity(
                    perturbed_outputs[t]['mean_embedding'],
                    true_outputs[t]['mean_embedding'],
                )
                self.test_dict['cls_cosine_similarity'].append(cls_cos_sim)
                self.test_dict['mean_cosine_similarity'].append(mean_agg_cos_sim)

                if self.return_rouge_score:
                    test_dict = self.compute_rouge_score(
                        pred_ids=pred_ids,
                        tgt_ids=tgt_ids,
                        rouge_len_list=self.rouge_seq_len_list,
                        max_seq_length=self.max_seq_length,
                        test_dict=self.test_dict,
                    )
                    self.test_dict = test_dict
                    # self.log(
                    #     'test/rouge1',
                    #     rouge_score['rouge1'],
                    #     on_step=False,
                    #     on_epoch=True,
                    #     prog_bar=True,
                    #     logger=True,
                    #     rank_zero_only=True,
                    #     sync_dist=True,
                    #     batch_size=batch['src_input_ids'].shape[0],
                    # )
            for time_step in self.pred_tps:
                self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{time_step}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])
                cls_embeddings = true_outputs[time_step]['cls_embedding'].detach().cpu()
                self.test_dict['cls_embeddings'].append(cls_embeddings)

    def on_test_epoch_end(self):
        if self.return_embeddings:
            obs_key = self.var_list if len(self.var_list) > 0 else []
            obs_key.extend(['cell_idx'])
            return_perturbation_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                output_dir=self.output_dir,
                file_name=(
                    f'{self.date}_prediction_embeddings'
                    f't{self.pred_tps}_lr{self.end_lr}_w{self.weight_decay}'
                ),
                mode='inference',
            )


class PerturberGenerationTrainer(CountDecoderTrainer):
    def __init__(
        self,
        genes_to_perturb: List[int] | None = None,
        perturbation_token: int | None = 0,
        cell_type_to_perturb: str | None = None,
        perturbation_mode: List[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        print('test kwargs', kwargs)

        self.decoder = PerturberMasking(
            pretrained_model=self.pretrained_model,
            loss_mode=kwargs['loss_mode'],
            d_model=kwargs['d_model'],
            dropout=kwargs['dropout'],
            pred_tps=kwargs['pred_tps'],
            context_tps=kwargs['context_tps'],
            context_mode=kwargs['context_mode'],
            n_genes=kwargs['n_genes'],
        )
        if perturbation_mode is not None:
            self.perturbation_mode = perturbation_mode
            self.genes_to_perturb = torch.tensor(genes_to_perturb, dtype=torch.long)
            self.perturbation_token = torch.tensor(perturbation_token, dtype=torch.long)
        else:
            self.perturbation_mode = []
        self.test_dict['cls_cosine_similarity'] = []
        self.test_dict['mean_cosine_similarity'] = []

    def test_step(self, batch, *args, **kwargs):
        tgt_input_id_dict = {}
        for i in self.total_tps:
            print(i)
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
            if len(self.perturbation_mode) > 0:
                if 'tgt' in self.perturbation_mode:
                    print('perturbating tgt')
                    perturbed_tgt = batch[f'tgt_input_ids_t{i}'].clone()
                    mask = torch.isin(
                        batch[f'tgt_input_ids_t{i}'], self.genes_to_perturb
                    )
                    perturbed_tgt[mask] = self.perturbation_token
        if len(self.perturbation_mode) > 0:
            if 'src' in self.perturbation_mode:
                print('perturbating src')
                perturbed_src = batch['src_input_ids'].clone()
                mask = torch.isin(batch['src_input_ids'], self.genes_to_perturb)
                perturbed_src[mask] = self.perturbation_token

    def on_test_epoch_end(self):
        if self.generate:
            obs_key = self.var_list if len(self.var_list) > 0 else []
            obs_key.extend(['cell_idx'])
            return_perturbation_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                output_dir=self.output_dir,
                file_name=(
                    f'{self.date}_generate_adata_'
                    f't{self.pred_tps}_{self.encoder}_s{self.seed}_'
                    f'l{self.loss_mode}_n{self.n_samples}'
                    f'_p{self.pos_encoding_mode}_'
                    f'm{self.mask_scheduler}_s{self.sequence_length}'
                ),
            )
