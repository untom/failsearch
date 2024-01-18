# Failsearch

`failsearch` uses AI to search through the media files on your computer.
It is able to match your search queries to the content of your files. If
you ask it to find the pictures of your three year old walking through
the snow, it will gladly do so.

This project uses multimodal AI to match search strings to image content.
More specifically, it uses uses Google's [SigLIP](https://huggingface.co/docs/transformers/main/en/model_doc/siglip)
or older models such as [CLIP](https://openai.com/research/clip).
These a neural network model that calculates the similarity between
an image and text.

![Screenshot of failsearch](./screenshot.jpg?raw=true)

# Installation

The program runs best when a GPU supported by PyTorch is available.
The program ran run without this, but indexing will probably be much slower.

To install dependencies, use the requirements.txt file:

```
pip install -r requirements.txt
```

# Usage

Use `failsearch -h` to see a list of all available options.
There are three steps to running failsearch:

## 1. Index your files

First you need to create an index of all your image files. Use
the `-i` flag to point the program towards the directory you'd
like to index.


```
python3 failsearch.py -i /media/images
```

## 2. Start the browser interface to search through your files

You can start the web interface by running

```
flask --app web_ui run
```

This will start the interface on port 5000. Browse to
http://127.0.0.1:5000 to see the web interface.

See `flask -h` for information on how to set a different port
or other options.



# License

Distributed Parameter Search is copyrighted (c) 2021 by Thomas Unterthiner and licensed under the
[General Public License (GPL) Version 3 or higher](http://www.gnu.org/licenses/gpl-3.0.html>).
See ``LICENSE.txt`` for the full details.
