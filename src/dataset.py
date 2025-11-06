"""
Dataset loading, processing, and tokenization utilities
"""

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


def load_translation_dataset(dataset_name="Helsinki-NLP/opus-100", config="en-it"):
    """
    Load and process the translation dataset
    
    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration (e.g., "en-it")
    Returns:
        x_train, y_train, x_val, y_val: Lists of English and Italian texts
    """
    ds = load_dataset(dataset_name, config)
    
    def extract_translations(batch):
        return {
            'en_text': [t['en'] for t in batch['translation']],
            'it_text': [t['it'] for t in batch['translation']],
        }
    
    processed_ds_train = ds['train'].map(extract_translations, batched=True)
    processed_ds_val = ds['validation'].map(extract_translations, batched=True)
    
    x_train = processed_ds_train['en_text']
    y_train = processed_ds_train['it_text']
    x_val = processed_ds_val['en_text']
    y_val = processed_ds_val['it_text']
    
    return x_train, y_train, x_val, y_val


def create_tokenizer(x_train, y_train):
    """
    Create character-level tokenizer
    
    Args:
        x_train: List of English texts
        y_train: List of Italian texts
    Returns:
        stoi: String to index mapping
        itos: Index to string mapping
        vocab_size: Size of vocabulary
    """
    all_text = "".join(x_train) + "".join(y_train)
    chars = sorted(list(set(all_text)))
    
    stoi = {c: i for i, c in enumerate(chars)}
    stoi['<PAD>'] = len(stoi)
    
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    
    return stoi, itos, vocab_size


class TranslationDataset(Dataset):
    """Dataset for translation pairs"""
    
    def __init__(self, en_texts, it_texts, stoi, max_len=128):
        """
        Args:
            en_texts: List of English sentences
            it_texts: List of Italian sentences
            stoi: Character to index mapping
            max_len: Maximum sequence length
        """
        # Filter pairs that fit within max_len so that we can fit them into the context window
        filtered_pairs = [
            (en, it) for en, it in zip(en_texts, it_texts)
            if len(en) <= max_len and len(it) <= max_len
        ]
        
        if filtered_pairs:
            self.en_texts, self.it_texts = zip(*filtered_pairs)
        else:
            self.en_texts, self.it_texts = [], []
        
        self.stoi = stoi
        self.max_len = max_len
    
    def __len__(self):
        return len(self.en_texts)
    
    def __getitem__(self, idx):
        # Encode the texts
        en_encoded = torch.tensor([self.stoi[char] for char in self.en_texts[idx]])
        it_encoded = torch.tensor([self.stoi[char] for char in self.it_texts[idx]])
        
        return en_encoded, it_encoded


def collate_fn(batch, stoi):
    """
    Custom collate function to pad sequences in each batch
    
    Args:
        batch: List of tuples (en_tensor, it_tensor)
        stoi: String to index mapping (for padding token)
    Returns:
        x_batch: Padded English sequences
        y_batch: Padded Italian sequences
    """
    en_batch, it_batch = zip(*batch)
    
    # Pad sequences
    x_batch = pad_sequence(en_batch, batch_first=True, padding_value=stoi['<PAD>'])
    y_batch = pad_sequence(it_batch, batch_first=True, padding_value=stoi['<PAD>'])
    
    return x_batch, y_batch


def create_dataloaders(x_train, y_train, x_val, y_val, stoi, batch_size=32, 
                       sample_size=None, max_len=128, num_workers=0):
    """
    Create training and validation data loaders
    
    Args:
        x_train: List of English training texts
        y_train: List of Italian training texts
        x_val: List of English validation texts
        y_val: List of Italian validation texts
        stoi: String to index mapping
        batch_size: Batch size
        sample_size: Number of training samples to use (None for all)
        max_len: Maximum sequence length
        num_workers: Number of worker processes for data loading
    Returns:
        train_loader, val_loader: Data loaders
    """
    # Create datasets
    train_dataset = TranslationDataset(x_train, y_train, stoi, max_len=max_len)
    val_dataset = TranslationDataset(x_val, y_val, stoi, max_len=max_len)
    
    # For experimenting with smaller samples
    if sample_size is not None:
        train_dataset = Subset(train_dataset, range(min(sample_size, len(train_dataset))))
    
    # Create collate function with stoi
    def collate_fn_with_stoi(batch):
        return collate_fn(batch, stoi)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_stoi,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_stoi,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

