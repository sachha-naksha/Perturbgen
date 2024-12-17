from typing import List, Literal

from T_perturb.Dataloaders.datamodule import CellGenDataModule, CellGenDataset


class PerturberDataModule(CellGenDataModule):
    def __init__(
        self,
        celltype_to_perturb: List[str],
        filter_mode: Literal['include', 'exclude'] = 'include',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.celltype_to_perturb = celltype_to_perturb
        self.filter_mode = filter_mode

    # create a function filter huggingface dataset
    def filter_huggingface_dataset(
        self, dataset, celltype_to_perturb, filter_mode, num_proc=1
    ):
        if filter_mode == 'include':
            # Include samples where 'cell_type' is in the list
            dataset = dataset.filter(
                lambda example: example['cell_type'] in celltype_to_perturb,
                num_proc=num_proc,
            )
        elif filter_mode == 'exclude':
            # Exclude samples where 'cell_type' is in the list
            dataset = dataset.filter(
                lambda example: example['cell_type'] not in celltype_to_perturb,
                num_proc=num_proc,
            )
        else:
            raise ValueError(
                f"Invalid mode: {filter_mode}. Must be 'include' or 'exclude'."
            )

        return dataset

    def setup(self, stage=None):
        if self.context_tps is not None:
            all_modelling_tps = self.pred_tps + self.context_tps
            self.all_modelling_tps = list(set(all_modelling_tps))
        else:
            self.all_modelling_tps = self.pred_tps
        # filter the dataset
        if self.celltype_to_perturb is not None:
            src_dataset_filtered = self.filter_huggingface_dataset(
                self.src_dataset,
                self.celltype_to_perturb,
                self.filter_mode,
                num_proc=self.dataloader_kwargs['num_workers'],
            )
            print('src_dataset_filtered', src_dataset_filtered)
            print(src_dataset_filtered['length'])
            tgt_dataset_filtered = {}
            for name, tgt_dataset in self.tgt_datasets.items():
                tgt_dataset = self.filter_huggingface_dataset(
                    tgt_dataset,
                    self.celltype_to_perturb,
                    self.filter_mode,
                    num_proc=self.dataloader_kwargs['num_workers'],
                )
                tgt_dataset_filtered[name] = tgt_dataset

        else:
            src_dataset_filtered = self.src_dataset

            tgt_dataset_filtered = self.tgt_datasets
        dataset_args = {
            'src_dataset': src_dataset_filtered,
            'tgt_datasets': tgt_dataset_filtered,
            'src_counts': self.src_counts,
            'tgt_counts_dict': self.tgt_counts_dict,
            'time_steps': self.all_modelling_tps,
        }
        if stage == 'test' or stage is None:
            # use all time steps to provide as context
            self.all_modelling_tps = self.total_tps
            dataset_args['time_steps'] = self.all_modelling_tps
            dataset_args['split_indices'] = self.test_indices
            if self.condition_encodings is not None:
                dataset_args['conditions'] = (
                    self.conditions if self.condition_keys is not None else None
                )
                dataset_args['conditions_combined'] = (
                    self.conditions_combined
                    if self.condition_keys is not None
                    else None
                )
                self.test_dataset = CellGenDataset(**dataset_args)
            else:
                self.test_dataset = CellGenDataset(**dataset_args)
