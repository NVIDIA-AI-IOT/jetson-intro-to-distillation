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

import subprocess
import os
import boto3
import botocore
import tqdm
import PIL.Image
import concurrent.futures
from torchvision.transforms import (
    Compose, 
    ToTensor, 
    Normalize, 
    Resize,
    CenterCrop,
    InterpolationMode,
    ColorJitter
)
from openclip_utils import (
    precompute_clip_image_embeddings,
    EmbeddingDatasetWrapper
)

# see https://storage.googleapis.com/openimages/web/download_v7.html#df-point-labels
def get_open_images_data_dir():
    return "data/open_images"

def get_file_url_output_path(url: str):
    return os.path.join(get_open_images_data_dir(), os.path.basename(url))

def _download(url: str, output_path: str):
    subprocess.call([
        "wget",
        url,
        "-O",
        output_path
    ])

def get_val_image_ids_url():
    return "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"

def get_val_human_verified_labels_url():
    return "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv"

def get_val_image_ids_output_path():
    return get_file_url_output_path(get_val_image_ids_url())

def download_val_image_ids():
    _download(get_val_image_ids_url(), get_val_image_ids_output_path())

def read_image_ids(csv_path: str):
    image_ids = []
    with open(csv_path, 'r') as f:
        for line in f.readlines()[1:]:
            linesplit = line.split(',')
            image_id = linesplit[0].strip()
            subset = linesplit[1].strip()
            image_ids.append((image_id, subset))
    return image_ids

def get_val_image_ids(download=False):
    if download:
        download_val_image_ids()
    return read_image_ids(get_val_image_ids_output_path())


def get_boto3_bucket():
    bucket = boto3.resource(
        's3', 
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED
        )
    ).Bucket("open-images-dataset")
    return bucket

def get_images_output_dir():
    return os.path.join(get_open_images_data_dir(), "images")

def get_image_output_path(image_id: str):
    return os.path.join(get_images_output_dir(), f"{image_id}.jpg")

def download_image(bucket, image_id, subset):
    if not os.path.exists(get_image_output_path(image_id)):
        bucket.download_file(f"{subset}/{image_id}.jpg", get_image_output_path(image_id))

def download_images(image_ids):
    bucket = get_boto3_bucket()

    if not os.path.exists(get_images_output_dir()):
        os.makedirs(get_images_output_dir())


    progress_bar = tqdm.tqdm(
        total=len(image_ids), desc='Downloading images', leave=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [   
            executor.submit(
                download_image, bucket, image_id, subset
            )
            for image_id, subset in image_ids
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()
            progress_bar.update(1)

    progress_bar.close()

def download_val_images():
    download_images(get_val_image_ids())



class OpenImagesBaseDataset(object):
    def __init__(self, split: str, download=False, transform=None):
        self.folder = get_images_output_dir()
        if split == 'val':
            if download:
                download_val_images()
            self.image_ids = get_val_image_ids(download=True)
        else:
            raise RuntimeError(f"data split {split} not currently supported")
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id, subset = self.image_ids[index]
        image_path = get_image_output_path(image_id)

        label = -1
        image = PIL.Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_open_images_val_embeddings_dir():
    return "data/open_images/val_clip_embeddings"


def precompute_open_images_val_clip_embeddings():
    dataset = OpenImagesBaseDataset(
        split="val",
        download=False
    )

    precompute_clip_image_embeddings(
        get_open_images_val_embeddings_dir(),
        dataset,
        overwrite=False
    )

def get_open_images_val_transform(size=224):
    return Compose([
        ToTensor(),
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


def get_open_images_val_embedding_dataset(transform=None):
    if transform is None:
        transform = get_open_images_val_transform()
    return EmbeddingDatasetWrapper(
        dataset=OpenImagesBaseDataset(
            split="val",
            transform=transform
        ),
        embeddings_dir=get_open_images_val_embeddings_dir()
    )




if __name__ == "__main__":
    precompute_open_images_val_clip_embeddings()