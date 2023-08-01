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
import torch.nn as nn
import torch.utils.data
import timm
from openclip_utils import (
    precompute_clip_image_embeddings,
    precompute_clip_text_embeddings,
    eval_dataset_clip_embeddings,
    eval_embeddings_model,
    eval_logits_model,
    train_probe_model,
    train_model_from_scratch,
    train_student_classification_model,
    train_student_embedding_model,
    EmbeddingDatasetWrapper,
    FilterTextEmbeddings
)
from torchvision.datasets import STL10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomResizedCrop, InterpolationMode, CenterCrop
from open_images_utils import (
    get_open_images_val_embedding_dataset,
    get_open_images_val_transform
)

STL10_LABELS = [
    'an airplane',
    'a bird',
    'a car',
    'a cat',
    'a deer',
    'a dog',
    'a horse',
    'a monkey',
    'a ship',
    'a truck'
]

def get_stl10_transform():
    transform = Compose([
        ToTensor(),
        Normalize(0.5, 0.25)
    ])
    return transform


def precompute_clip_stl10_image_embeddings(
        output_dir, 
        dataset_path,
        data_split,
        overwrite=False):

    dataset = STL10(
        root=dataset_path,
        download=True,
        split=data_split
    )

    precompute_clip_image_embeddings(
        output_dir,
        dataset,
        overwrite
    )


def precompute_clip_stl10_unlabeled_image_embeddings():
    precompute_clip_stl10_image_embeddings(
        output_dir="data/clip/stl10_unlabeled_image_embeddings",
        dataset_path="data/stl10",
        data_split="unlabeled"
    )


def precompute_clip_stl10_train_image_embeddings():
    precompute_clip_stl10_image_embeddings(
        output_dir="data/clip/stl10_train_image_embeddings",
        dataset_path="data/stl10",
        data_split="train"
    )


def precompute_clip_stl10_test_image_embeddings():
    precompute_clip_stl10_image_embeddings(
        output_dir="data/clip/stl10_test_image_embeddings",
        dataset_path="data/stl10",
        data_split="test"
    )


def precompute_clip_stl10_text_embeddings():
    precompute_clip_text_embeddings(
        output_path="data/clip/stl10_text_embeddings.pt",
        labels=STL10_LABELS,

    )


def get_clip_stl10_text_embeddings():
    return torch.load("data/clip/stl10_text_embeddings.pt")


def get_stl10_unlabeled_embedding_dataset(transform=None):
    if transform is None:
        transform = get_stl10_transform()
    return EmbeddingDatasetWrapper(
        dataset=STL10(
            root="data/stl10",
            download=True,
            split="unlabeled",
            transform=transform
        ),
        embeddings_dir="data/clip/stl10_unlabeled_image_embeddings"
    )


def get_stl10_train_embedding_dataset(transform=None):
    if transform is None:
        transform = get_stl10_transform()
    return EmbeddingDatasetWrapper(
        dataset=STL10(
            root="data/stl10",
            download=True,
            split="train",
            transform=transform
        ),
        embeddings_dir="data/clip/stl10_train_image_embeddings"
    )


def get_stl10_train_unlabeled_embedding_dataset(transform=None):
    return torch.utils.data.ConcatDataset([
        get_stl10_train_embedding_dataset(transform),
        get_stl10_unlabeled_embedding_dataset(transform)
    ])


def get_stl10_test_embedding_dataset(transform=None):
    if transform is None:
        transform = get_stl10_transform()
    return EmbeddingDatasetWrapper(
        dataset=STL10(
            root="data/stl10",
            download=True,
            split="test",
            transform=transform
        ),
        embeddings_dir="data/clip/stl10_test_image_embeddings"
    )


def eval_stl10_train_clip_embeddings():
    text_embeddings = get_clip_stl10_text_embeddings()
    dataset = get_stl10_train_embedding_dataset()
    accuracy = eval_dataset_clip_embeddings(dataset, text_embeddings)
    with open("data/clip/stl10_train_clip_acc.txt", 'w') as f:
        f.write(f"ACCURACY: {accuracy}")
    return accuracy


def eval_stl10_test_clip_embeddings():
    text_embeddings = get_clip_stl10_text_embeddings()
    dataset = get_stl10_test_embedding_dataset()
    accuracy = eval_dataset_clip_embeddings(dataset, text_embeddings)
    with open("data/clip/stl10_test_clip_acc.txt", 'w') as f:
        f.write(f"ACCURACY: {accuracy}")
    return accuracy

# Probe ablation


def train_probe_model_linear():
    train_probe_model(
        output_dir="data/experiments/train_probe_model_linear",
        probe_model=nn.Linear(512, len(STL10_LABELS)),
        train_dataset=get_stl10_train_embedding_dataset(),
        test_dataset=get_stl10_test_embedding_dataset(),
        learning_rate=3e-4,
        batch_size=64,
        num_workers=8,
        num_epochs=15,
        temperature=1.,
        seed=0
    )


def train_probe_model_mlp():

    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    train_probe_model(
        output_dir="data/experiments/train_probe_model_mlp",
        probe_model=model,
        train_dataset=get_stl10_train_embedding_dataset(),
        test_dataset=get_stl10_test_embedding_dataset(),
        learning_rate=3e-4,
        batch_size=64,
        num_workers=8,
        num_epochs=15,
        temperature=1.,
        seed=0
    )


def train_student_linear_probe(
        output_dir: str,
        arch: str,
        temperature: float=1.,
        train_dataset = None,
        test_dataset=None
    ):
    if train_dataset is None:
        train_dataset = get_stl10_train_unlabeled_embedding_dataset()
    if test_dataset is None:
        test_dataset = get_stl10_test_embedding_dataset()
    # call train_probe_model_linear first
    probe_model = nn.Linear(512, len(STL10_LABELS))
    probe_weights = "data/experiments/train_probe_model_linear/checkpoint_14.pth"
    if not os.path.exists(probe_weights):
        train_probe_model_linear()
    probe_model.load_state_dict(torch.load("data/experiments/train_probe_model_linear/checkpoint_14.pth"))
    train_student_classification_model(
        output_dir=output_dir,
        model=timm.create_model(arch, num_classes=len(STL10_LABELS)),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        learning_rate=3e-4,
        batch_size=64,
        num_workers=8,
        num_epochs=50,
        temperature=temperature,
        probe_model=probe_model,
        seed=0
    )

def train_student_zero_shot(
        output_dir: str,
        arch: str,
        temperature: float=1.,
        train_dataset = None,
        test_dataset = None
    ):
    if train_dataset is None:
        train_dataset = get_stl10_train_unlabeled_embedding_dataset()
    if test_dataset is None:
        test_dataset = get_stl10_test_embedding_dataset()
    train_student_classification_model(
        output_dir=output_dir,
        model=timm.create_model(arch, num_classes=len(STL10_LABELS)),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        learning_rate=3e-4,
        batch_size=64,
        num_workers=8,
        num_epochs=50,
        temperature=temperature,
        text_embeddings=get_clip_stl10_text_embeddings(),
        seed=0
    )

# Temperature ablation

def train_resnet18_from_scratch():
    train_model_from_scratch(
        output_dir="data/experiments/train_resnet18_from_scratch",
        model=timm.create_model("resnet18", num_classes=len(STL10_LABELS)),
        train_dataset=get_stl10_train_embedding_dataset(),
        test_dataset=get_stl10_test_embedding_dataset(),
        learning_rate=3e-4,
        batch_size=64,
        num_workers=8,
        num_epochs=50,
        seed=0
    )

def train_resnet18_zero_shot_train_only():
    train_student_zero_shot(
        output_dir=f"data/experiments/train_resnet18_zero_shot_train_only",
        arch="resnet18",
        temperature=1.,
        train_dataset=get_stl10_train_embedding_dataset()
    )

def train_resnet18_zero_shot():
    train_student_zero_shot(
        output_dir=f"data/experiments/train_resnet18_zero_shot",
        arch="resnet18",
        temperature=1.
    )

def train_resnet18_zero_shot_t100():
    train_student_zero_shot(
        output_dir=f"data/experiments/train_resnet18_zero_shot",
        arch="resnet18",
        temperature=100.
    )
def train_resnet18_zero_shot_tp5():
    train_student_zero_shot(
        output_dir=f"data/experiments/train_resnet18_zero_shot_tp5",
        arch="resnet18",
        temperature=0.5
    )

def train_resnet18_zero_shot_t2():
    train_student_zero_shot(
        output_dir=f"data/experiments/train_resnet18_zero_shot_t2",
        arch="resnet18",
        temperature=2.0
    )

def train_resnet18_linear_probe():
    train_student_linear_probe(
        output_dir=f"data/experiments/train_resnet18_linear_probe",
        arch="resnet18", 
        temperature=1.
    )

def train_resnet18_linear_probe_train_only():
    train_student_linear_probe(
        output_dir=f"data/experiments/train_resnet18_linear_probe_train_only",
        arch="resnet18", 
        temperature=1.,
        train_dataset=get_stl10_train_embedding_dataset()
    )

def train_resnet18_linear_probe_tp5():
    train_student_linear_probe(
        output_dir=f"data/experiments/train_resnet18_linear_probe_tp5",
        arch="resnet18", 
        temperature=0.5
    )

def train_resnet18_linear_probe_t2():
    train_student_linear_probe(
        output_dir=f"data/experiments/train_resnet18_linear_probe_t2",
        arch="resnet18", 
        temperature=2.0
    )

def train_resnet34_linear_probe():
    train_student_linear_probe(
        output_dir=f"data/experiments/train_resnet34_linear_probe",
        arch="resnet34", 
        temperature=1.
    )

def train_resnet50_linear_probe():
    train_student_linear_probe(
        output_dir=f"data/experiments/train_resnet50_linear_probe",
        arch="resnet50", 
        temperature=1.
    )

def train_embedding_text(output_dir: str, arch: str, train_dataset=None,
        test_dataset=None, weight_by_nearest_embedding=False, nearest_embedding_weight_std=1.):
    if train_dataset is None:
        train_dataset = get_stl10_train_unlabeled_embedding_dataset()
    if test_dataset is None:
        test_dataset = get_stl10_test_embedding_dataset()
    train_student_embedding_model(
        output_dir=output_dir,
        model=timm.create_model(arch, num_classes=512),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        learning_rate=3e-4,
        batch_size=64,
        num_workers=8,
        num_epochs=50,
        text_embeddings=get_clip_stl10_text_embeddings(),
        seed=0,
        include_test_accuracy=True,
        weight_by_nearest_embedding=weight_by_nearest_embedding,
        nearest_embedding_weight_std=nearest_embedding_weight_std
    )

def train_resnet18_embedding_text():
    train_embedding_text(f"data/experiments/train_resnet18_embedding_text","resnet18")


def eval_resnet18_embedding_text():
    model = timm.create_model("resnet18", num_classes=512)
    model.load_state_dict(torch.load("data/experiments/train_resnet18_embedding_text/checkpoint_48.pth"))

    eval_embeddings_model(
        output_dir=f"data/experiments/eval_resnet18_embedding_text",
        model=model,
        dataset=get_stl10_test_embedding_dataset(),
        text_embeddings=get_clip_stl10_text_embeddings()
    )

def eval_resnet18_embedding_linear():
    model = timm.create_model("resnet18", num_classes=512)
    model.load_state_dict(torch.load("data/experiments/train_resnet18_embedding_text/checkpoint_48.pth"))

    probe_model = nn.Linear(512, len(STL10_LABELS))
    probe_weights = "data/experiments/train_probe_model_linear/checkpoint_14.pth"
    if not os.path.exists(probe_weights):
        train_probe_model_linear()
    probe_model.load_state_dict(torch.load(probe_weights))

    eval_embeddings_model(
        output_dir=f"data/experiments/eval_resnet18_embedding_linear",
        model=model,
        dataset=get_stl10_test_embedding_dataset(),
        probe_model=probe_model
    )


def eval_resnet18_embedding_mlp():
    model = timm.create_model("resnet18", num_classes=512)
    model.load_state_dict(torch.load("data/experiments/train_resnet18_embedding_text/checkpoint_48.pth"))

    probe_model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(STL10_LABELS))
    )
    probe_weights = "data/experiments/train_probe_model_mlp/checkpoint_14.pth"
    if not os.path.exists(probe_weights):
        train_probe_model_mlp()
    probe_model.load_state_dict(torch.load(probe_weights))
    eval_embeddings_model(
        output_dir=f"data/experiments/eval_resnet18_embedding_mlp",
        model=model,
        dataset=get_stl10_test_embedding_dataset(),
        probe_model=probe_model
    )

def get_stl10_open_images_filtered_dataset_90(transform=None):
    return FilterTextEmbeddings(
        get_open_images_val_embedding_dataset(transform),
        text_embeddings=get_clip_stl10_text_embeddings(),
        thresh=0.9
    )

def train_resnet18_text_open_images_224_filter90():
    train_student_zero_shot(
        "data/experiments/train_resnet18_text_open_images_224_filter90",
        arch="resnet18",
        temperature=1.,
        train_dataset=get_stl10_open_images_filtered_dataset_90(),
        test_dataset=get_stl10_test_embedding_dataset(transform=get_open_images_val_transform())
    )
def train_resnet18_text_open_images_224_filter90_temp100():
    train_student_zero_shot(
        "data/experiments/train_resnet18_text_open_images_224_filter90_temp100",
        arch="resnet18",
        temperature=100.,
        train_dataset=get_stl10_open_images_filtered_dataset_90(),
        test_dataset=get_stl10_test_embedding_dataset(transform=get_open_images_val_transform())
    )
def train_resnet18_text_open_images_96_filter90_temp100():
    train_student_zero_shot(
        "data/experiments/train_resnet18_text_open_images_96_filter90_temp100",
        arch="resnet18",
        temperature=100.,
        train_dataset=get_stl10_open_images_filtered_dataset_90(get_open_images_val_transform(size=96)),
        test_dataset=get_stl10_test_embedding_dataset(transform=get_open_images_val_transform(size=96))
    )

if __name__ == "__main__":
    train_resnet18_zero_shot_t100()