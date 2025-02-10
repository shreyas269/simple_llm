from torch.utils.data import DataLoader

class LLMDataloader(DataLoader):
    def llm_dataloader(self, llm_dataset, batch_size, shuffle=True, drop_last=True, num_workers=0):
        dataloader = DataLoader(
        llm_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
        )

        return dataloader
    