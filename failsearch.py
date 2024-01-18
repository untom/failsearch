#!/bin/env python3
"""Uses AI models to search for images."""

import abc
import argparse
import gc
import logging
import numpy as np
import pathlib
import queue
import sqlite3
import time
import threading
import tqdm.auto as tqdm
import torch
import transformers

from concurrent.futures import ThreadPoolExecutor
from PIL import Image


IMAGE_FILE_EXTENSIONS = ["jpg", "jpeg", "png"]


logging.basicConfig(
    format='%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ],
    level=logging.INFO)


class SimilarityModelBase(abc.ABC):
    """Baseclass for CLIP-like models to calculate similarities."""

    def preprocess_images(self, images):
        with torch.no_grad():
            image_preproc = self.processor(images=images, return_tensors="pt")
            pixel_values = image_preproc['pixel_values']
        return pixel_values

    def preprocess_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return inputs

    def get_logits(self, image_embeds, text_embeds):
        with torch.no_grad():
            scale = self._get_logit_scale()
            logits = torch.matmul(text_embeds, image_embeds.t()) * scale
        return logits.detach().numpy()

    def create_image_embeddings(self, pixel_values):
        pass

    def create_text_embeddings(self, text_inputs):
        pass

    def _get_logit_scale(self):
        pass


class AlignModel(SimilarityModelBase):
    """An ALIGN model to calculate representations of text/images and score their similarities."""
    def __init__(self, use_gpu=True):
        model_name = "kakaobrain/align-base"
        self.processor = transformers.AlignProcessor.from_pretrained(model_name)
        self.model = transformers.AlignModel.from_pretrained(model_name)
        if use_gpu:
            self.model = self.model.to("cuda")

    def create_image_embeddings(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[1]
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds

    def create_text_embeddings(self, text_inputs):
        with torch.no_grad():
            attention_mask = text_inputs['attention_mask']
            input_ids = text_inputs['input_ids']
            text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs[0][:, 0, :]
            text_embeds = self.model.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def _get_logit_scale(self):
        return 1.0 / self.model.temperature


class ClipModel(SimilarityModelBase):
    """A CLIP model to calculate representations of text/images and score their similarities."""
    def __init__(self, use_gpu=True):
        model_name = "openai/clip-vit-base-patch32"
        self.processor = transformers.CLIPProcessor.from_pretrained(model_name)
        self.model = transformers.CLIPModel.from_pretrained(model_name)
        if use_gpu:
            self.model = self.model.to("cuda")

    def create_image_embeddings(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[1]
            image_embeds = self.model.visual_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds

    def create_text_embeddings(self, text_inputs):
        with torch.no_grad():
            attention_mask = text_inputs['attention_mask']
            input_ids = text_inputs['input_ids']
            text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs[1]
            text_embeds = self.model.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def _get_logit_scale(self):
        return self.model.logit_scale.exp()


def get_model(model_name, use_gpu):
    if model_name == "openai/clip-vit-base-patch32":
        return ClipModel(use_gpu) 
    elif model_name == "kakaobrain/align-base":
        return AlignModel(use_gpu)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class Database:
    """A Database to store image representations of a given model."""

    def __init__(self, fname=None):
        if fname is None:
            fname = ":memory:"
        self.con = sqlite3.connect(fname, check_same_thread=False, isolation_level="Deferred")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS clip_image_embeddings(path TEXT, image_embeds BLOB)")
        self.con.execute("CREATE TABLE IF NOT EXISTS config(key TEXT, value TEXT)")
        self.con.execute("PRAGMA synchronous = NORMAL")
        self.con.execute("PRAGMA journal_mode = WAL")  # Moar speed.

    def _set_config(self, key, value):
        self.con.execute("INSERT INTO config(key, value) VALUES(?, ?)", (key, value))
        self.con.commit()

    def _get_config(self, key):
        e = self.con.execute("SELECT value FROM config WHERE key = ?", (key,))
        return e.fetchone()[0]

    def set_model(self, model_name):
        self._set_config("model", model_name)

    def get_model(self):
        return self._get_config("model")


    def insert(self, paths, image_embeds):
        #assert image_embeds.shape[1] == self.embed_size, f"{image_embeds.shape[1]=} != {self.embed_size}"
        self._set_config("embedding_size", str(image_embeds.shape[1]))

        if len(paths) != image_embeds.shape[0]:
            raise RuntimeError(f"{len(paths)=} != {image_embeds.shape[0]}=")

        data = [(str(p), i.tobytes()) for p, i in zip(paths, image_embeds)]    
        self.con.executemany("INSERT INTO clip_image_embeddings(path, image_embeds) VALUES(?, ?)", data)
        self.con.commit()

    def get_num_entries(self):
        e = self.con.execute("SELECT COUNT(*) FROM clip_image_embeddings")
        return e.fetchone()[0]

    def get_all_data(self):
        n = self.get_num_entries()
        paths = []
        embed_size = int(self._get_config("embedding_size"))
        image_embeds = np.empty((n, embed_size), dtype=np.float32)
        for i, (pp, ii) in enumerate(self.con.execute("SELECT path, image_embeds FROM clip_image_embeddings")):
            image_embeds[i] = np.frombuffer(ii, np.float32)
            paths.append(pp)
        return paths, image_embeds

    def close(self):
        self.con.close()
        self.con = None


def get_all_files_with_extension(path, file_extensions):
    """Returns all files of a given extension under a given path (including subdirs)."""
    files = []
    for ext in file_extensions:
        files += list(path.rglob(f"*.{ext}"))
        files += list(path.rglob(f"*.{ext.upper()}"))
    files = [f for f in files if not '.stversions' in str(f)]  # filter out backups
    return files


def batch(sequence, batch_size):
    """Batches up a sequence into chunks."""
    n = len(sequence)
    i = 0
    while i < n:
        end_idx = i + batch_size if (i + batch_size < n) else n
        yield sequence[i:end_idx]
        i = end_idx


def index_directory(db, path, model_name, batch_size=32, num_threads=4, file_extensions=IMAGE_FILE_EXTENSIONS):
    """Stores embeddings for all image files below a given path in the db."""
    path = pathlib.Path(path)
    all_files = get_all_files_with_extension(path, file_extensions)

    logging.info(f"Total relevant files: {len(all_files)}")
    logging.info(f"Total GB in pixel_values: {(4 * 3*224*224 * len(all_files)) / 1024**3:3.2f}")
    logging.info(f"Total GB in image_embeds: {(4 * 512 * len(all_files)) / 1024**3:3.2f}")
    logging.info(f"Model: {model_name}")
    logging.info(f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}")

    db.set_model(model_name)
    model = get_model(model_name, use_gpu=True)

    # In the following, we start a 2-stage pipeline: Stage 1 loads a batch of
    # images from disk and preprocesses them (within a threadpool), and Stage 2
    # runs the actual vision model to create embeddings and stores them in a
    # database. I tried separating Stage 2 into two stages by making the DB 
    # its own Stage, but that was not faster at all.
    def _preprocess_images(q, model, files):
        while q.full():
            time.sleep(1)
        pixel_values_list = []
        processed_files = []

        # Process images w/o batching so we're able to catch exceptions.
        for f in files:
            try:
                with Image.open(f) as img:
                    img = img.convert("RGB")  # Convert grayscale images.
                    pixel_values_list.append(model.preprocess_images(img))
                processed_files.append(f)
            except Exception as err:
                logging.warning(f"{err}: {f}")

        if len(pixel_values_list):
            pixel_values = torch.cat(pixel_values_list, axis=0)
            pixel_values = pixel_values.to(device="cuda")
            q.put((processed_files, pixel_values))
            del pixel_values, pixel_values_list



    def _embed(q, model, db, num_total):
        n = 0
        with tqdm.tqdm(total=num_total) as progressbar:
            while True:
                tmp = q.get()
                if tmp is None:
                    return
                files, pixel_values = tmp
                image_embeds = model.create_image_embeddings(pixel_values).detach().cpu().numpy()
                del pixel_values
                db.insert(files, image_embeds)
                del files, image_embeds
                progressbar.update(1)
                n += 1
                if n % 100 == 0:
                    gc.collect()

    q = queue.Queue(maxsize=4)
    num_batches = len(all_files) // batch_size
    embed_thread = threading.Thread(target=_embed, args=(q, model, db, num_batches))
    embed_thread.start()

    with ThreadPoolExecutor(num_threads) as pool:
        tmp = pool.map(lambda f: _preprocess_images(q, model, f), batch(all_files, batch_size))
        for _ in tmp: pass  # waiting for jobs to be done.
        q.put(None)
    embed_thread.join() 
    db.close()


def search(db, text, n=5):
    """returns the closest N matches for a given text in the DB."""

    model_name = db.get_model()
    cpu_model = get_model(model_name, use_gpu = False)

    files, image_embeds = db.get_all_data()
    image_embeds = torch.from_numpy(image_embeds)
    text_values = cpu_model.preprocess_texts([text])
    text_embeds = cpu_model.create_text_embeddings(text_values)
    logits = cpu_model.get_logits(image_embeds, text_embeds)
    print(logits.shape)

    idx = np.argsort(logits[0])[-n:]
    return [files[i] for i in idx[::-1]]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--db', type=str, help='Path to database',
                        default="./failsearch.sqlite")
    parser.add_argument('-i', '--index', type=str, help='Index everything in the given directory', default=None)
    parser.add_argument('-s', '--search', type=str, help='Search DB for best matches', default=None)
    parser.add_argument('-n', '--num_results', type=int, help='Number of returned search results.', default=10)
    parser.add_argument('-t', '--num_threads', type=int, help='Number of threads used for indexing.', default=4)
    parser.add_argument('-m', '--model', type=str, help='Model name.', default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    db_path = pathlib.Path(args.db)
    if not db_path.exists():
        logging.warning("DB path doesn't exist, creating new DB.")
    db = Database(args.db)
    logging.info(f"Entries in DB: {db.get_num_entries()}")
    if args.index is not None and args.search is None:
        index_directory(db, path=args.index, model_name=args.model, num_threads=args.num_threads)
    elif args.search is not None and args.index is None:
        results = search(db, args.search, n=args.num_results)
        print("\n".join(results))
    else:
        print("Please use either the index or search method.")
