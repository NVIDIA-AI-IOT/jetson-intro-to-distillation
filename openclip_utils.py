# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import open_clip
import tqdm
import random
import glob
import numpy as np
import torch.nn.functional as F
import torch.utils.data
import PIL.Image
from torch.utils.tensorboard import SummaryWriter


class EmptyEmbeddingDataset(object):
    def __len__(self):
        return 0

    def __getitem__(self):
        raise NotImplementedError



def embedding_to_probs(embedding, text_embedding, temp=100.):
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    logits = embedding @ text_embedding.T
    logits = F.softmax(temp * logits, dim=-1)
    return logits


class EmbeddingDatasetWrapper(object):

    def __init__(self, dataset, embeddings_dir):
        self.dataset = dataset
        self.embeddings_dir = embeddings_dir
        for i in range(len(self.dataset)):
            embedding_path = os.path.join(self.embeddings_dir, f"embedding_{i}.pt")
            assert os.path.exists(embedding_path)
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        with torch.no_grad():
            image, label = self.dataset[index]
            embedding = self.get_embedding(index)
        return image, label, embedding

    def get_embedding(self, index):
        embedding = torch.load(os.path.join(self.embeddings_dir, f"embedding_{index}.pt"))
        embedding = embedding[0].detach()
        return embedding

class FilterTextEmbeddings():
    def __init__(self, dataset, text_embeddings, thresh=0.9):
        probs = []
        for i in tqdm.tqdm(range(len(dataset))):
            embedding = dataset.get_embedding(i)
            probs_i = embedding_to_probs(embedding, text_embeddings)
            probs.append(probs_i)
        probs = torch.stack(probs)
        indices = torch.nonzero(torch.amax(probs, dim=1) >= thresh).flatten()
        self.indices = [int(x) for x in indices]
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]

    def __len__(self):
        return len(self.indices)

    def get_embedding(self, index):
        return self.dataset.get_embedding(int(self.indices[index]))


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_clip_model():
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", 
        pretrained="laion2b_s34b_b79k"
    )

    return model, preprocess


def get_clip_tokenizer():
    return open_clip.get_tokenizer("ViT-B-32")


def precompute_clip_image_embeddings(output_dir, dataset, overwrite=False):

    model, preprocess = get_clip_model()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, (image, _) in tqdm.tqdm(enumerate(dataset)):

        output_path = os.path.join(output_dir, f"embedding_{index}.pt")

        if os.path.exists(output_path) and (not overwrite):
            continue

        input_tensor = preprocess(image).unsqueeze(0)

        embedding = model.encode_image(input_tensor)

        torch.save(embedding, output_path)


def compute_clip_text_embeddings(labels):
    tokenizer = get_clip_tokenizer()
    model, _ = get_clip_model()
    text = tokenizer(labels)
    embeddings = model.encode_text(text)
    return embeddings

def precompute_clip_text_embeddings(output_path, labels):

    embeddings = compute_clip_text_embeddings(labels)

    torch.save(embeddings, output_path)


def eval_dataset_clip_embeddings(embeddings_dataset, text_embeddings, batch_size=64):

    loader = torch.utils.data.DataLoader(
        dataset=embeddings_dataset,
        batch_size=batch_size
    )

    text_embeddings = text_embeddings.to("cuda")
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    num_correct = 0
    for image, label, image_embeddings in tqdm.tqdm(iter(loader)):
        with torch.no_grad():
            image = image.to("cuda")
            label = label.to("cuda")
            image_embeddings = image_embeddings.to("cuda")
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            logits = image_embeddings @ text_embeddings.T

            num_correct += int(torch.count_nonzero(logits.argmax(dim=-1) == label))
    
    return round(100. * num_correct / len(embeddings_dataset), 3)


def eval_embeddings_model(
        output_dir,
        dataset,
        model,
        probe_model=None,
        text_embeddings=None,
        batch_size=64
    ):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )

    assert (probe_model is not None) or (text_embeddings is not None)

    if text_embeddings is not None:
        text_embeddings = text_embeddings.to("cuda")
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    if probe_model is not None:
        probe_model = probe_model.cuda().eval()

    model = model.cuda().eval()

    num_correct = 0
    for image, label, _ in tqdm.tqdm(iter(loader)):
        with torch.no_grad():
            image = image.to("cuda")
            label = label.to("cuda")
            output_embeddings = model(image)
            if text_embeddings is not None:
                output_embeddings = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)
                output_embeddings = output_embeddings
                output_logits = output_embeddings @ text_embeddings.T
            else:
                output_logits = probe_model(output_embeddings)

            num_correct += int(torch.count_nonzero(output_logits.argmax(dim=-1) == label))
    

    acc = round(100. * num_correct / len(dataset), 3)

    logstr = f"TEST ACC: {acc}"
    print(logstr)
    with open(os.path.join(output_dir, "log.txt"), 'w') as f:
        f.write(logstr)


def eval_logits_model(
        output_dir,
        dataset,
        model,
        batch_size=64
    ):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )

    model = model.cuda().eval()

    num_correct = 0
    for image, label, _ in tqdm.tqdm(iter(loader)):
        with torch.no_grad():
            image = image.to("cuda")
            label = label.to("cuda")
            output_logits = model(image)

            num_correct += int(torch.count_nonzero(output_logits.argmax(dim=-1) == label))
    

    acc = round(100. * num_correct / len(dataset), 3)
    logstr = f"TEST ACC: {acc}"
    print(logstr)
    with open(os.path.join(output_dir, "log.txt"), 'w') as f:
        f.write(logstr)


def train_student_classification_model(
        output_dir,
        model,
        train_dataset,
        test_dataset,
        learning_rate,
        batch_size,
        num_workers,
        num_epochs,
        text_embeddings=None,
        probe_model=None,
        temperature=100.,
        seed=0,
    ):
    # Trains a classifier using zero-shot capabilities of a clip model with
    # text embeddings according to the label name.
    
    seed_all(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(output_dir)

    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    assert (probe_model is not None) or (text_embeddings is not None)

    if text_embeddings is not None:
        text_embeddings = text_embeddings.to("cuda")
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    if probe_model is not None:
        probe_model = probe_model.cuda().eval()

    best_acc = 0
    for epoch in range(num_epochs):

        # Train
        train_loss = 0.

        model.train()
        for image, label, clip_image_embeddings in tqdm.tqdm(iter(train_loader)):
            with torch.no_grad():
                image = image.cuda()
                label = label.cuda()
                clip_image_embeddings = clip_image_embeddings.cuda().detach()

                if text_embeddings is not None:
                    clip_image_embeddings = clip_image_embeddings / clip_image_embeddings.norm(dim=-1, keepdim=True)
                    clip_image_embeddings = clip_image_embeddings
                    clip_logits = clip_image_embeddings @ text_embeddings.T
                else:
                    clip_logits = probe_model(clip_image_embeddings)

            clip_logprob = F.log_softmax(temperature * clip_logits, dim=-1)
            optimizer.zero_grad()
            output_logits = model(image)

            output_logprob = F.log_softmax(temperature * output_logits, dim=-1)

            loss = F.kl_div(output_logprob, clip_logprob, log_target=True)

            loss.backward()
            optimizer.step()

            train_loss += float(loss)

        train_loss /= len(train_loader)

        # Eval
        model.eval()
        test_acc = 0.
        for image, label, clip_image_embeddings in tqdm.tqdm(iter(test_loader)):
            with torch.no_grad():
                image = image.cuda()
                label = label.cuda()
                output_logits = model(image)

                test_acc += int(torch.count_nonzero(output_logits.argmax(dim=-1) == label))
        test_acc = round(100 * test_acc / len(test_dataset), 3)

        logstr = f"| EPOCH {epoch} | TRAIN LOSS {train_loss} | TEST ACC {test_acc} |"
        print(logstr)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(logstr + "\n")

        writer.add_scalar("train_loss", train_loss, global_step=epoch)
        writer.add_scalar("test_acc", test_acc, global_step=epoch)

        if test_acc > best_acc:

            checkpoint_path = os.path.join(output_dir, f"checkpoint_{epoch}.pth")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            best_acc = test_acc


def train_student_embedding_model(
        output_dir,
        model,
        train_dataset,
        test_dataset,
        learning_rate,
        batch_size,
        num_workers,
        num_epochs,
        seed=0,
        text_embeddings=None,
        probe_model=None,
        loss_function = lambda x, y: F.mse_loss(x, y),
        include_test_accuracy: bool = False,
        weight_by_nearest_embedding = False,
        nearest_embedding_weight_std = 1.
    ):
    # Trains a classifier using zero-shot capabilities of a clip model with
    # text embeddings according to the label name.
    
    seed_all(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(output_dir)

    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    if text_embeddings is not None:
        text_embeddings = text_embeddings.to("cuda")
        text_embeddings_no_norm = text_embeddings
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    if probe_model is not None:
        probe_model = probe_model.cuda().eval()
    if include_test_accuracy:
        assert (probe_model is not None) or (text_embeddings is not None)
    if weight_by_nearest_embedding:
        assert (probe_model is not None) or (text_embeddings is not None)

    best_loss = 1e9
    for epoch in range(num_epochs):

        # Train
        train_loss = 0.

        model.train()
        for image, label, clip_image_embeddings in tqdm.tqdm(iter(train_loader)):
            with torch.no_grad():
                image = image.cuda()
                label = label.cuda()
                clip_image_embeddings = clip_image_embeddings.cuda().detach()

            optimizer.zero_grad()
            output_embeddings = model(image)

            if weight_by_nearest_embedding:
                clip_embed_norm = clip_image_embeddings / clip_image_embeddings.norm(dim=-1, keepdim=True)
                dist = torch.sum((clip_embed_norm[:, None, :] - text_embeddings_no_norm[None, :, :])**2, dim=-1)
                mindist = torch.amin(dist, dim=-1)
                weight = torch.exp(-mindist / nearest_embedding_weight_std**2)
                weight = F.softmax(weight, dim=0) * len(weight)
                loss = F.mse_loss(output_embeddings, clip_image_embeddings) * weight[:, None]
                loss = torch.mean(loss)
            else:
                loss = loss_function(output_embeddings, clip_image_embeddings)

            loss.backward()
            optimizer.step()

            train_loss += float(loss)

        train_loss /= len(train_loader)

        # Eval
        model.eval()
        test_loss = 0.
        test_acc = 0.
        for image, label, clip_image_embeddings in tqdm.tqdm(iter(test_loader)):
            with torch.no_grad():


                image = image.cuda()
                clip_image_embeddings = clip_image_embeddings.cuda()
                output_embeddings = model(image)
                test_loss += float(loss_function(output_embeddings, clip_image_embeddings))

                if include_test_accuracy:
                    label = label.cuda()
                    if text_embeddings is not None:
                        output_embeddings = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)
                        output_logits = output_embeddings @ text_embeddings.T
                    else:
                        output_logits = probe_model(output_embeddings)
                    test_acc += int(torch.count_nonzero(output_logits.argmax(dim=-1) == label))
                
        test_loss /= len(test_loader)

        logstr = f"| EPOCH {epoch} | TRAIN LOSS {train_loss} | TEST LOSS {test_loss} |"
        
        if include_test_accuracy:
            test_acc = round(100 * test_acc / len(test_dataset), 3)
            logstr += f" TEST ACC {test_acc} |"

        print(logstr)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(logstr + "\n")

        writer.add_scalar("train_loss", train_loss, global_step=epoch)
        writer.add_scalar("test_loss", test_loss, global_step=epoch)

        if test_loss < best_loss:

            checkpoint_path = os.path.join(output_dir, f"checkpoint_{epoch}.pth")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            best_loss = test_loss


def train_probe_model(
        output_dir,
        probe_model,
        train_dataset,
        test_dataset,
        learning_rate,
        batch_size,
        num_workers,
        num_epochs,
        temperature=100.,
        seed=0
    ):
    # Trains a probe model using a labeled training dataset. "Few-shot".
    # The probe model takes clip embeddings and generates a class label.
    seed_all(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(output_dir)

    probe_model = probe_model.cuda()
    
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    best_train_loss = 1e9
    for epoch in range(num_epochs):

        # Train
        train_loss = 0.

        probe_model.train()
        for _, label, clip_image_embeddings in tqdm.tqdm(iter(train_loader)):
            with torch.no_grad():
                label = label.cuda()
                clip_image_embeddings = clip_image_embeddings.cuda().detach()

            optimizer.zero_grad()
            output_logits = probe_model(clip_image_embeddings)

            output_logprob = F.log_softmax(temperature * output_logits, dim=-1)

            loss = F.nll_loss(output_logprob, label)

            loss.backward()
            optimizer.step()

            train_loss += float(loss)

        train_loss /= len(train_loader)

        # Eval
        probe_model.eval()
        test_acc = 0.
        for _, label, clip_image_embeddings in tqdm.tqdm(iter(test_loader)):
            with torch.no_grad():
                label = label.cuda()
                clip_image_embeddings = clip_image_embeddings.cuda()
                output_logits = probe_model(clip_image_embeddings)

                test_acc += int(torch.count_nonzero(output_logits.argmax(dim=-1) == label))
        test_acc = round(100 * test_acc / len(test_dataset), 3)

        logstr = f"| EPOCH {epoch} | TRAIN LOSS {train_loss} | TEST ACC {test_acc} |"
        print(logstr)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(logstr + "\n")
        writer.add_scalar("train_loss", train_loss, global_step=epoch)
        writer.add_scalar("test_acc", test_acc, global_step=epoch)

        if train_loss < best_train_loss:

            checkpoint_path = os.path.join(output_dir, f"checkpoint_{epoch}.pth")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(probe_model.state_dict(), checkpoint_path)
            best_train_loss = train_loss

def train_model_from_scratch(
        output_dir,
        model,
        train_dataset,
        test_dataset,
        learning_rate,
        batch_size,
        num_workers,
        num_epochs,
        seed=0
    ):
    # Trains a model from scratch using ground truth labels
    seed_all(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(output_dir)

    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    best_train_loss = 1e9
    for epoch in range(num_epochs):

        # Train
        train_loss = 0.

        model.train()
        for image, label, _ in tqdm.tqdm(iter(train_loader)):
            with torch.no_grad():
                image = image.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            output_logits = model(image)

            output_logprob = F.log_softmax(output_logits, dim=-1)

            loss = F.nll_loss(output_logprob, label)

            loss.backward()
            optimizer.step()

            train_loss += float(loss)

        train_loss /= len(train_loader)

        # Eval
        model.eval()
        test_acc = 0.
        for image, label, _ in tqdm.tqdm(iter(test_loader)):
            with torch.no_grad():
                image = image.cuda()
                label = label.cuda()
                output_logits = model(image)

                test_acc += int(torch.count_nonzero(output_logits.argmax(dim=-1) == label))
        test_acc = round(100 * test_acc / len(test_dataset), 3)

        logstr = f"| EPOCH {epoch} | TRAIN LOSS {train_loss} | TEST ACC {test_acc} |"
        print(logstr)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(logstr + "\n")
        writer.add_scalar("train_loss", train_loss, global_step=epoch)
        writer.add_scalar("test_acc", test_acc, global_step=epoch)

        if train_loss < best_train_loss:

            checkpoint_path = os.path.join(output_dir, f"checkpoint_{epoch}.pth")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            best_train_loss = train_loss

if __name__ == "__main__":

    pass