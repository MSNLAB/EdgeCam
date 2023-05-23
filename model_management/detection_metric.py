import time

from numpy import average as avg
from tqdm import tqdm

class RetrainMetric:

    def __init__(self):
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {
            'loss_classifier': [], 'loss_box_reg': [],
            'loss_objectness': [], 'loss_rpn_box_reg': [],
            'total_loss': [],
        }

    def update(self, loss_dict, total_loss):
        self.metrics['loss_classifier'].append(loss_dict['loss_classifier'].detach().cpu().item())
        self.metrics['loss_box_reg'].append(loss_dict['loss_box_reg'].detach().cpu().item())
        self.metrics['loss_objectness'].append(loss_dict['loss_objectness'].detach().cpu().item())
        self.metrics['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].detach().cpu().item())
        self.metrics['total_loss'].append(total_loss.detach().cpu().item())

    def compute(self):
        return {
            'loss_classifier': avg(self.metrics['loss_classifier']),
            'loss_box_reg': avg(self.metrics['loss_box_reg']),
            'loss_objectness': avg(self.metrics['loss_objectness']),
            'loss_rpn_box_reg': avg(self.metrics['loss_rpn_box_reg']),
            'total_loss': avg(self.metrics['total_loss']),
        }

    def log_iter(self, epoch, num_epoch, data_loader):
        loop = tqdm(enumerate(data_loader, 1), total=len(data_loader),
                    desc=f'Epoch [{epoch}/{num_epoch}]')

        self.reset_metrics()
        data_load_time = []
        train_process_time = []

        end_time = time.time()
        for idx, (images, targets) in loop:
            start_time = time.time()
            data_load_time.append(start_time - end_time)

            yield images, targets

            end_time = time.time()
            train_process_time.append(end_time - start_time)

            loop.set_postfix(
                loss=f"{avg(self.metrics['total_loss']):.2f}",
                data=f"{avg(data_load_time):.3f}s/it",
                train=f"{avg(train_process_time):.3f}s/it",
            )
