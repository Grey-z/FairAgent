import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class DenseBatchFCTanh(nn.Module):
    def __init__(self, input_dim, units, reg, do_norm=False):
        """
        A dense layer with optional batch normalization and tanh activation.

        Args:
            input_dim (int): Input dimension.
            units (int): Output dimension.
            reg (float): Regularization coefficient.
            do_norm (bool): Whether to apply batch normalization. Defaults to False.
        """
        super(DenseBatchFCTanh, self).__init__()
        self.do_norm = do_norm
        self.fc = nn.Linear(input_dim, units)  # Fully connected layer
        if do_norm:
            self.bn = nn.BatchNorm1d(units)  # Batch normalization layer
        self.reg = reg  # Regularization coefficient

    def forward(self, x):
        """
        Forward pass for the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the layer.
        """
        h1 = self.fc(x)  # Apply fully connected layer
        if self.do_norm:
            h1 = self.bn(h1) if self.training else h1  # Apply batch normalization if in training mode
        return torch.tanh(h1)  # Apply tanh activation


class DenseFC(nn.Module):
    def __init__(self, input_dim, units, reg):
        """
        A simple dense (fully connected) layer.

        Args:
            input_dim (int): Input dimension.
            units (int): Output dimension.
            reg (float): Regularization coefficient.
        """
        super(DenseFC, self).__init__()
        self.fc = nn.Linear(input_dim, units)  # Fully connected layer
        self.reg = reg  # Regularization coefficient

    def forward(self, x):
        """
        Forward pass for the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the layer.
        """
        return self.fc(x)  # Apply fully connected layer


class ALDI(nn.Module):
    def __init__(self, config, emb_dim, content_dim, alpha=0.9, beta=0.05, gama=0.1):
        """
        ALDI (Adaptive Learning for Domain-Invariant Representations) model.

        Args:
            config (dict): Configuration dictionary containing hyperparameters.
            emb_dim (int): Embedding dimension.
            content_dim (int): Content feature dimension.
            alpha (float): Weight for distillation loss. Defaults to 0.9.
            beta (float): Weight for frequency-based loss. Defaults to 0.05.
            gama (float): Weight for regularization. Defaults to 0.1.
        """
        super(ALDI, self).__init__()
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.transformed_layers = [emb_dim, emb_dim]  # Layer dimensions for transformation
        self.lr = config['lr']  # Learning rate
        self.reg = config['reg_1']  # Regularization coefficient
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta  # Weight for frequency-based loss
        self.gamma = gama  # Weight for regularization
        self.freq_coef_a = config['freq_coef_a']  # Coefficient for frequency scaling
        self.freq_coef_M = config['freq_coef_M']  # Maximum frequency scaling
        self.tws = config['tws']  # Whether to use frequency-based weighting

        # Initialize device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define layers for item and user transformation
        self.f_item_layers = nn.ModuleList()  # Layers for item transformation
        self.f_user_layers = nn.ModuleList()  # Layers for user transformation
        input_dim = content_dim
        for ihid, hid in enumerate(self.transformed_layers[:-1]):
            self.f_item_layers.append(DenseBatchFCTanh(input_dim, hid, self.reg, do_norm=True))
            self.f_user_layers.append(DenseBatchFCTanh(emb_dim, hid, self.reg, do_norm=True))
            input_dim = hid
        self.f_item_output = DenseFC(input_dim, self.transformed_layers[-1], self.reg)  # Output layer for items
        self.f_user_output = DenseFC(input_dim, self.transformed_layers[-1], self.reg)  # Output layer for users

        # Move model to the specified device
        self.to(self.device)

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, item_content, true_user_emb):
        """
        Forward pass for the ALDI model.

        Args:
            item_content (torch.Tensor): Item content features.
            true_user_emb (torch.Tensor): True user embeddings.

        Returns:
            tuple: A tuple containing:
                - gen_item_emb (torch.Tensor): Generated item embeddings.
                - user_emb (torch.Tensor): Transformed user embeddings.
        """
        # Ensure inputs are Float tensors and move to the correct device
        if item_content is not None:
            item_content = item_content.float().to(self.device)
        true_user_emb = true_user_emb.float().to(self.device)

        # Generate item embeddings
        gen_item_emb = item_content
        for layer in self.f_item_layers:
            gen_item_emb = layer(gen_item_emb)
        gen_item_emb = self.f_item_output(gen_item_emb)

        # Transform user embeddings
        user_emb = true_user_emb
        for layer in self.f_user_layers:
            user_emb = layer(user_emb)
        user_emb = self.f_user_output(user_emb)

        return gen_item_emb, user_emb

    def compute_loss(self, pos_item_content, pos_item_emb, neg_item_content, neg_item_emb, user_emb, pos_item_freq, neg_item_freq):
        """
        Compute the total loss for the model.

        Args:
            pos_item_content (torch.Tensor): Content features of positive items.
            pos_item_emb (torch.Tensor): Embeddings of positive items.
            neg_item_content (torch.Tensor): Content features of negative items.
            neg_item_emb (torch.Tensor): Embeddings of negative items.
            user_emb (torch.Tensor): User embeddings.
            pos_item_freq (torch.Tensor): Frequencies of positive items.
            neg_item_freq (torch.Tensor): Frequencies of negative items.

        Returns:
            torch.Tensor: Total loss.
        """
        # Concatenate positive and negative samples
        item_content = torch.cat([pos_item_content, neg_item_content], dim=0).to(self.device)
        true_item_emb = torch.cat([pos_item_emb, neg_item_emb], dim=0).to(self.device)
        freq = torch.cat([pos_item_freq, neg_item_freq], dim=0).to(self.device)

        # Forward pass
        gen_item_emb, map_user_emb = self.forward(item_content, user_emb)

        # Separate positive and negative samples
        pos_gen_item_emb = gen_item_emb[:pos_item_content.size(0)]
        neg_gen_item_emb = gen_item_emb[pos_item_content.size(0):]
        pos_true_item_emb = true_item_emb[:pos_item_content.size(0)]
        neg_true_item_emb = true_item_emb[pos_item_content.size(0):]

        # Compute supervised loss
        student_pos_logit = torch.sum(map_user_emb * pos_gen_item_emb, dim=1)
        student_neg_logit = torch.sum(map_user_emb * neg_gen_item_emb, dim=1)
        student_rank_distance = student_pos_logit - student_neg_logit
        supervised_loss = F.binary_cross_entropy_with_logits(student_rank_distance, torch.ones_like(student_rank_distance))

        # Compute distillation loss
        teacher_pos_logit = torch.sum(user_emb * pos_true_item_emb, dim=1)
        teacher_neg_logit = torch.sum(user_emb * neg_true_item_emb, dim=1)
        teacher_rank_distance = teacher_pos_logit - teacher_neg_logit
        if self.tws:
            pos_item_freq = torch.tanh(self.freq_coef_a * pos_item_freq).clamp(0, torch.tanh(torch.tensor(self.freq_coef_M)))
        else:
            pos_item_freq = torch.ones_like(student_rank_distance)
        distill_loss = self.alpha * F.binary_cross_entropy_with_logits(student_rank_distance, torch.sigmoid(teacher_rank_distance), weight=pos_item_freq)

        # Compute regularization loss
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)
        reg_loss = self.reg * reg_loss

        # Total loss
        total_loss = supervised_loss + distill_loss + reg_loss
        return total_loss

    def train_step(self, pos_item_content, pos_item_emb, neg_item_content, neg_item_emb, user_emb, pos_item_freq, neg_item_freq):
        """
        Perform a single training step.

        Args:
            pos_item_content (torch.Tensor): Content features of positive items.
            pos_item_emb (torch.Tensor): Embeddings of positive items.
            neg_item_content (torch.Tensor): Content features of negative items.
            neg_item_emb (torch.Tensor): Embeddings of negative items.
            user_emb (torch.Tensor): User embeddings.
            pos_item_freq (torch.Tensor): Frequencies of positive items.
            neg_item_freq (torch.Tensor): Frequencies of negative items.

        Returns:
            float: Loss value for the training step.
        """
        # Ensure data is on the correct device
        pos_item_content = pos_item_content.to(self.device)
        pos_item_emb = pos_item_emb.to(self.device)
        neg_item_content = neg_item_content.to(self.device)
        neg_item_emb = neg_item_emb.to(self.device)
        user_emb = user_emb.to(self.device)
        pos_item_freq = pos_item_freq.to(self.device)
        neg_item_freq = neg_item_freq.to(self.device)

        # Perform training step
        self.optimizer.zero_grad()
        loss = self.compute_loss(pos_item_content, pos_item_emb, neg_item_content, neg_item_emb, user_emb, pos_item_freq, neg_item_freq)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_user_rating(self, uemb, iemb):
        """
        Compute user-item ratings.

        Args:
            uemb (torch.Tensor): User embeddings.
            iemb (torch.Tensor): Item embeddings.

        Returns:
            torch.Tensor: User-item rating matrix.
        """
        # Ensure data is on the correct device
        uemb = uemb.to(self.device)
        iemb = iemb.to(self.device)
        return torch.matmul(uemb, iemb.t())

    def get_item_emb(self, item_content, item_emb, warm_item_ids, cold_item_ids):
        """
        Generate item embeddings for cold items.

        Args:
            item_content (torch.Tensor): Item content features.
            item_emb (torch.Tensor): Item embeddings.
            warm_item_ids (list): IDs of warm items.
            cold_item_ids (list): IDs of cold items.

        Returns:
            torch.Tensor: Updated item embeddings.
        """
        with torch.no_grad():
            # Ensure data is on the correct device
            item_content = item_content.to(self.device)
            item_emb = item_emb.to(self.device)
            gen_item_emb, _ = self.forward(item_content[cold_item_ids], torch.zeros_like(item_emb[cold_item_ids]))
            item_emb[cold_item_ids] = gen_item_emb
        return item_emb

    def get_user_emb(self, user_emb):
        """
        Transform user embeddings.

        Args:
            user_emb (torch.Tensor): User embeddings.

        Returns:
            torch.Tensor: Transformed user embeddings.
        """
        user_emb = user_emb.float().to(self.device)
        dummy_item_content = torch.zeros((user_emb.size(0), self.content_dim), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # Ensure data is on the correct device
            _, trans_user_emb = self.forward(dummy_item_content, user_emb)
        return torch.stack([user_emb, trans_user_emb], dim=0)

    def get_ranked_rating(self, ratings, k):
        """
        Get top-k ranked items.

        Args:
            ratings (torch.Tensor): User-item rating matrix.
            k (int): Number of top items to retrieve.

        Returns:
            tuple: A tuple containing:
                - top_score (torch.Tensor): Scores of top-k items.
                - top_item_index (torch.Tensor): Indices of top-k items.
        """
        # Ensure data is on the correct device
        ratings = ratings.to(self.device)
        top_score, top_item_index = torch.topk(ratings, k=k, dim=1)
        return top_score, top_item_index


class TrainDataset(Dataset):
    def __init__(self, data, content_data, item_emb, user_emb, item_freq):
        """
        Dataset class for training.

        Args:
            data (list): List of training data tuples (user_id, positive_item_id, negative_item_id).
            content_data (numpy array): Item content features.
            item_emb (numpy array): Item embeddings.
            user_emb (numpy array): User embeddings.
            item_freq (numpy array): Item frequencies.
        """
        self.data = data
        self.content_data = torch.tensor(content_data, dtype=torch.float32)  # Ensure CPU Tensor
        self.item_emb = torch.tensor(item_emb, dtype=torch.float32)  # Ensure CPU Tensor
        self.user_emb = torch.tensor(user_emb, dtype=torch.float32)  # Ensure CPU Tensor
        self.item_freq = torch.tensor(item_freq, dtype=torch.float32)  # Ensure CPU Tensor

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing:
                - pos_item_content (torch.Tensor): Content features of the positive item.
                - pos_item_emb (torch.Tensor): Embedding of the positive item.
                - neg_item_content (torch.Tensor): Content features of the negative item.
                - neg_item_emb (torch.Tensor): Embedding of the negative item.
                - user_emb (torch.Tensor): User embedding.
                - pos_item_freq (torch.Tensor): Frequency of the positive item.
                - neg_item_freq (torch.Tensor): Frequency of the negative item.
        """
        uid, pos1, neg1 = self.data[idx]
        return (
            self.content_data[pos1], self.item_emb[pos1],
            self.content_data[neg1], self.item_emb[neg1],
            self.user_emb[uid], self.item_freq[pos1], self.item_freq[neg1]
        )

from tqdm import tqdm


def train_ALDI_model(model, optimizer, train_loader, config, device='cuda:0'):
    """
    Train the ALDI model and save the best model.

    Args:
        model: The ALDI model to be trained.
        optimizer: The optimizer used for training.
        train_loader (DataLoader): DataLoader for the training data.
        config (dict): Configuration dictionary containing 'max_epoch'.
        device (str): Device to use for training (default: 'cuda:0').

    Returns:
        model: The trained model.
    """
    # Move the model to the specified device (e.g., GPU)
    model = model.to(device)

    # Training loop over epochs
    for epoch in range(1, config['max_epoch'] + 1):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0  # Initialize epoch loss

        # Use tqdm to display a progress bar for the training process
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}/{config['max_epoch']}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Unpack the batch
                pos_item_content, pos_item_emb, neg_item_content, neg_item_emb, user_emb_batch, pos_item_freq, neg_item_freq = batch

                # Move all tensors to the specified device
                pos_item_content = pos_item_content.to(device)
                pos_item_emb = pos_item_emb.to(device)
                neg_item_content = neg_item_content.to(device)
                neg_item_emb = neg_item_emb.to(device)
                user_emb_batch = user_emb_batch.to(device)
                pos_item_freq = pos_item_freq.to(device)
                neg_item_freq = neg_item_freq.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Compute the loss using the model's compute_loss method
                loss = model.compute_loss(
                    pos_item_content, pos_item_emb, neg_item_content, neg_item_emb, user_emb_batch, pos_item_freq, neg_item_freq
                )

                # Backpropagate the loss
                loss.backward()

                # Update the model parameters
                optimizer.step()

                # Accumulate the epoch loss
                epoch_loss += loss.item()

                # Update the progress bar with the current loss
                pbar.set_postfix(loss=loss.item())

        # Calculate the average loss for the epoch
        epoch_loss /= len(train_loader)
        print(f'Epoch [{epoch}/{config["max_epoch"]}], Average Loss: {epoch_loss:.4f}')

    # Print a message indicating the completion of training
    print("Finish training model at epoch {}.".format(epoch))
    return model