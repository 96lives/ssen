import os
import torch
from torch.utils.tensorboard import SummaryWriter
from models.base import Model
from data import DataScheduler


def train_model(config, model: Model,
                scheduler: DataScheduler,
                writer: SummaryWriter):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')
    os.makedirs(saved_model_path, exist_ok=True)

    skip_batch = 0
    for step, (x, y, epoch) in enumerate(scheduler):

        x, y = x.to(config['device']), y.to(config['device'])

        # since number of points vary in the dataset,
        # we skip if gpu overflow occurs
        if config['skip_gpu_overflow']:
            try:
                train_loss = model.learn(x, y, step)
            except RuntimeError:
                skip_batch += 1
                continue
        else:
            train_loss = model.learn(x, y, step)

        # model learns
        print('\r[Epoch {:4}, Step {:7}, Overflow: {:7}, Loss {:5}]'.format(
            epoch, step, skip_batch, '%.3f' % train_loss), end=''
        )

        # evaluate
        if scheduler.check_eval_step(step):
            scheduler.eval(model, writer, step)

        if scheduler.check_vis_step(step):
            print("\nVisualizing...")
            scheduler.visualize(model, writer, step)
            writer.add_scalar('skip_batch', skip_batch, step)

        if (step + 1) % config['ckpt_step'] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    saved_model_path, 'ckpt-step-{}'
                    .format(str(step + 1).zfill(3))
                )
            )

        model.lr_scheduler.step()
