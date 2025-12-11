"""
Training Logger Module
A reusable logger for tracking training progress across different models.
"""
import logging
import torch
from tqdm import tqdm
TQDM_BAR_FORMAT = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}'
LOGGER = logging.getLogger("coiltech")
class TrainingLogger:
    """
    A flexible training logger that can be used with different models.
    Tracks metrics like losses, GPU memory, instances, and image sizes.
    """

    def __init__(self, rank=-1, world_size=1):
        """
        Initialize the training logger.

        Args:
            rank (int): Process rank for distributed training (-1 for single GPU)
            world_size (int): Number of processes in distributed training
        """
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank in {-1, 0}

    def print_header(self, loss_names=None):
        """
        Print the header for training progress.

        Args:
            loss_names (list): List of loss names to display.
                             Default: ['box_loss', 'cls_loss', 'dfl_loss']
        """
        if loss_names is None:
            loss_names = ['box_loss', 'cls_loss', 'dfl_loss']

        # Build header dynamically based on loss names
        num_columns = 3 + len(loss_names) + 1  # Epoch + GPU_mem + losses + Instances + Size
        header_format = '\n' + '%11s' * num_columns

        header_items = ['Epoch', 'GPU_mem'] + loss_names + ['Instances', 'Size']
        header_str = header_format % tuple(header_items)

        LOGGER.info(header_str)
        return num_columns

    def create_progress_bar(self, dataloader, total_batches):
        """
        Create a progress bar for the training loop.

        Args:
            dataloader: The training dataloader
            total_batches (int): Total number of batches

        Returns:
            Progress bar object
        """
        pbar = enumerate(dataloader)

        if self.is_main_process:
            pbar = tqdm(pbar, total=total_batches, bar_format=TQDM_BAR_FORMAT)

        return pbar

    def log_batch(self, epoch, epochs, mean_losses, targets_shape, imgs_shape,
                  pbar=None, batch_idx=0):
        """
        Log information for a single batch.

        Args:
            epoch (int): Current epoch number
            epochs (int): Total number of epochs
            mean_losses (torch.Tensor or list): Mean losses [box_loss, cls_loss, dfl_loss, ...]
            targets_shape (tuple): Shape of targets tensor (num_instances, ...)
            imgs_shape (tuple): Shape of images tensor (..., height, width)
            pbar: Progress bar object (if available)
            batch_idx (int): Current batch index
        """
        if not self.is_main_process:
            return

        # Calculate GPU memory
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'

        # Convert mean_losses to list if tensor
        if isinstance(mean_losses, torch.Tensor):
            mean_losses = mean_losses.cpu().numpy().tolist()

        # Get number of instances and image size
        num_instances = targets_shape[0] if len(targets_shape) > 0 else 0
        img_size = imgs_shape[-1] if len(imgs_shape) > 0 else 0

        # Build description string dynamically
        # num_losses = len(mean_losses)
        # desc_format = '%11s' * 2 + '%11.4g' * (num_losses + 2)
        # a = ['Epochs','GPU USAGE','l2_loss', 'ssim_loss', 'seg_loss', 'total_loss','Batch','Size','Bar','Time']
        # desc_items = [f'{epoch}/{epochs - 1}', mem] + list(mean_losses) + [num_instances, img_size]

        if pbar is not None:
            # Create a more descriptive string for tqdm
            loss_names = ['l2', 'ssim', 'seg', 'total']
            # Safely handle if more or fewer losses are passed
            loss_str = ' | '.join([f'{name}:{val:.4f}' for name, val in zip(loss_names, mean_losses)])
            desc = f'Epoch:{epoch+1}/{epochs} | Mem.Usag.:{mem} | {loss_str} | Bat:{num_instances} | Sz:{img_size}'
            pbar.set_description(desc)
        else:
            # Fallback for non-tqdm logging (keep tabular format if preferred, or use the descriptive one)
            num_losses = len(mean_losses)
            desc_format = '%11s' * 2 + '%11.4g' * (num_losses + 2)
            desc_items = [f'{epoch}/{epochs - 1}', mem] + list(mean_losses) + [num_instances, img_size]
            LOGGER.info(desc_format % tuple(desc_items))

    def log_epoch_end(self, epoch, lr_rates, mean_losses, results=None):
        """
        Log information at the end of an epoch.

        Args:
            epoch (int): Current epoch number
            lr_rates (list): Learning rates for each parameter group
            mean_losses (torch.Tensor or list): Mean losses for the epoch
            results (tuple): Validation results (P, R, mAP@.5, mAP@.5-.95, val_losses...)
        """
        if not self.is_main_process:
            return

        log_msg = f'\nEpoch {epoch} completed'

        if lr_rates:
            log_msg += f' | LR: {lr_rates[0]:.6f}'

        if isinstance(mean_losses, torch.Tensor):
            mean_losses = mean_losses.cpu().numpy().tolist()

        if mean_losses:
            loss_names = ['box_loss', 'cls_loss', 'dfl_loss'][:len(mean_losses)]
            losses_str = ' | '.join([f'{name}: {loss:.4f}'
                                    for name, loss in zip(loss_names, mean_losses)])
            log_msg += f' | {losses_str}'

        if results is not None:
            metrics_names = ['P', 'R', 'mAP@.5', 'mAP@.5-.95']
            metrics_str = ' | '.join([f'{name}: {val:.4f}'
                                     for name, val in zip(metrics_names, results[:4])])
            log_msg += f' | {metrics_str}'

        LOGGER.info(log_msg)


# Example usage functions
def example_usage_single_model():
    """
    Example: Using TrainingLogger with a single model training loop
    """
    logger = TrainingLogger(rank=-1, world_size=1)

    # Print header
    logger.print_header(['box_loss', 'cls_loss', 'dfl_loss'])

    # Simulate training loop
    epochs = 10
    for epoch in range(epochs):
        # Simulate mean losses
        mean_losses = torch.tensor([0.05, 0.03, 0.02])

        # Simulate batch info
        targets_shape = (32,)  # 32 instances in batch
        imgs_shape = (16, 3, 640, 640)  # batch_size, channels, height, width

        # Log batch information
        logger.log_batch(epoch, epochs, mean_losses, targets_shape, imgs_shape)

        # Simulate validation results
        results = (0.85, 0.80, 0.75, 0.65, 0.04, 0.03, 0.02)  # P, R, mAP@.5, mAP@.5-.95, losses
        lr_rates = [0.001]

        # Log epoch end
        logger.log_epoch_end(epoch, lr_rates, mean_losses, results)


def example_usage_with_progress_bar():
    """
    Example: Using TrainingLogger with progress bar (like in the original training script)
    """
    logger = TrainingLogger(rank=-1, world_size=1)

    # Print header
    logger.print_header()

    # Simulate dataloader
    class DummyDataLoader:
        def __init__(self):
            self.data = [(None, None, None, None) for _ in range(100)]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

    train_loader = DummyDataLoader()
    nb = len(train_loader)
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        # Create progress bar
        pbar = logger.create_progress_bar(train_loader, nb)

        # Initialize mean losses
        mloss = torch.zeros(3, device=device)

        for i, (imgs, targets, paths, _) in pbar:
            # Simulate loss calculation
            loss_items = torch.tensor([0.05, 0.03, 0.02], device=device)
            mloss = (mloss * i + loss_items) / (i + 1)

            # Simulate batch data
            targets_shape = (32,)
            imgs_shape = (16, 3, 640, 640)

            # Log batch
            logger.log_batch(epoch, epochs, mloss, targets_shape, imgs_shape, pbar, i)


if __name__ == "__main__":
    print("Training Logger Module - Example Usage\n")
    print("=" * 60)
    print("Example 1: Simple logging without progress bar")
    print("=" * 60)
    example_usage_single_model()

    print("\n" + "=" * 60)
    print("Example 2: Logging with progress bar")
    print("=" * 60)
    example_usage_with_progress_bar()
