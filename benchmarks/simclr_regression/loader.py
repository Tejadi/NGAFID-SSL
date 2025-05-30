class MaskedGADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        input_cols: list[str],
        masking_ratio=0.6,
        mean_mask_length=3,
        mode='separate',
        distribution='geometric',
        exclude_feats=None,
    ):
        self.paths = []
        dir_list = os.listdir(os.path.join(data_path, split))
        pbar = tqdm(dir_list, desc=f"Loading {split}", unit="file")
        for fname in pbar:
            full = os.path.join(data_path, split, fname)
            self.paths.append(full)
        self.input_cols      = input_cols
        self.mask_kwargs     = dict(
            masking_ratio=masking_ratio,
            mean_mask_length=mean_mask_length,
            mode=mode,
            distribution=distribution,
            exclude_feats=exclude_feats,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        df = pd.read_csv(self.paths[idx], na_values=[' NaN','NaN','NaN '])
        df = df[self.input_cols]
        n_rows = len(df)
        if n_rows < 4096:
            pad = pd.DataFrame(0, index=range(4096 - n_rows), columns=df.columns)
            df = pd.concat([df, pad], ignore_index=True)
        else:
            df = df.iloc[:4096]
        df = df.fillna(0)
        arr = df.values
        orig, masked = mask_transform(arr, **self.mask_kwargs)
        t_orig   = torch.from_numpy(orig).float()
        t_masked = torch.from_numpy(masked).float()
        return t_masked.unsqueeze(0), t_orig
