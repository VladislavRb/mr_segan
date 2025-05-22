import os
import json
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# POTENTIAL REFACTOR
from torch.cuda.amp import autocast, GradScaler

from model import Generator, Discriminator
from mr_waveform_dataset import MRWaveformDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--val_dataset', type=str)

    parser.add_argument("--early_stopping_patience", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    scaler = GradScaler()

    print('loading data...')

    train_dataset = MRWaveformDataset(dataset_path=args.train_dataset)
    val_dataset = MRWaveformDataset(dataset_path=args.val_dataset)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ref_batch = train_dataset.reference_batch(batch_size)

    discriminator = Discriminator()
    generator = Generator()

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)

    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))

    g_optimizer = optim.RMSprop(generator.parameters(), lr=args.lr)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=args.lr)

    if not os.path.exists('saved_checkpoints'):
        os.mkdir('saved_checkpoints')

    best_val_loss = float('inf')
    saved_checkpoint_counter = 0
    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience

    metrics_history = []

    train_dataset_length = len(train_data_loader.dataset)
    val_dataset_length = len(val_data_loader.dataset)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        train_bar = tqdm(train_data_loader)

        avg_train_d_clean_loss = 0
        avg_train_d_noisy_loss = 0
        avg_train_g_loss = 0
        avg_train_g_cond_loss = 0

        for train_batch, train_clean, train_noisy in train_bar:
            z = nn.init.normal(torch.Tensor(train_batch.size(0), 1024, 8))
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            discriminator.zero_grad()
            with autocast():
                outputs = discriminator(train_batch, ref_batch)
                clean_loss = torch.mean((outputs - 1.0) ** 2)

                generated_outputs = generator(train_noisy, z)
                outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
                noisy_loss = torch.mean(outputs ** 2)

                d_loss = clean_loss + noisy_loss

            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()

            generator.zero_grad()
            with autocast():
                generated_outputs = generator(train_noisy, z)
                gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
                outputs = discriminator(gen_noise_pair, ref_batch)

                g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
                l1_dist = torch.abs(generated_outputs - train_clean)
                g_cond_loss = 100 * torch.mean(l1_dist)
                g_loss = g_loss_ + g_cond_loss

            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()

            dcl_value = clean_loss.data.item()
            dnl_value = noisy_loss.data.item()
            gl_value = g_loss.data.item()
            gcl_value = g_cond_loss.data.item()

            avg_train_d_clean_loss += dcl_value
            avg_train_d_noisy_loss += dnl_value
            avg_train_g_loss += gl_value
            avg_train_g_cond_loss += gcl_value

            train_bar.set_description(f'Epoch {epoch + 1}: dcl {dcl_value:.4f}, dnl {dnl_value:.4f}, gl {gl_value:.4f}, gcl {gcl_value:.4f}')

        avg_train_d_clean_loss /= train_dataset_length
        avg_train_d_noisy_loss /= train_dataset_length
        avg_train_g_loss /= train_dataset_length
        avg_train_g_cond_loss /= train_dataset_length

        print(f'Train average losses: dcl {avg_train_d_clean_loss:.4f}, dnl {avg_train_d_noisy_loss:.4f}, gl {avg_train_g_loss:.4f}, gcl {avg_train_g_cond_loss:.4f}')

        avg_val_loss = 0.0
        generator.eval()
        with torch.no_grad():
            for _, clean, noisy in tqdm(val_data_loader, desc='Validation'):
                z = torch.randn(noisy.size(0), 1024, 8)
                if torch.cuda.is_available():
                    noisy, clean, z = noisy.cuda(), clean.cuda(), z.cuda()
                noisy, clean, z = Variable(noisy), Variable(clean), Variable(z)

                with autocast():
                    output = generator(noisy, z)
                    loss = nn.MSELoss()(output, clean)

                avg_val_loss += loss.item() * noisy.size(0)

        avg_val_loss /= val_dataset_length
        print(f'Validation MSE loss: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            g_path = os.path.join('saved_checkpoints', f'generator_{saved_checkpoint_counter}.pkl')
            d_path = os.path.join('saved_checkpoints', f'discriminator_{saved_checkpoint_counter}.pkl')

            torch.save(generator.state_dict(), g_path)
            torch.save(discriminator.state_dict(), d_path)

            patience_counter = 0
            saved_checkpoint_counter += 1

            print(f'Saved best generator model → {g_path}; discriminator model → {d_path}')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{early_stopping_patience}')
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping: no improvement for {early_stopping_patience} epochs')
                break

        metrics_history.append({
            'epoch': epoch + 1,
            'avg_train_d_clean_loss': avg_train_d_clean_loss,
            'avg_train_d_noisy_loss': avg_train_d_noisy_loss,
            'avg_train_g_loss': avg_train_g_loss,
            'avg_train_g_cond_loss': avg_train_g_cond_loss,
            'avg_val_loss': avg_val_loss,
            'best_val_loss_so_far': best_val_loss
        })

        with open('train_metrics.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)


if __name__ == '__main__':
    main()
