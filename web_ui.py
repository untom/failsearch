#!/bin/env python3
"""A Web UI for failsearch."""

import failsearch
import flask
import logging
import os
import urllib.parse


logging.basicConfig(
    format='%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ],
    level=logging.INFO)


class Searcher:
    def __init__(self, db_name):
        db = failsearch.Database(_db_name)
        logging.info(f"Using DB: {_db_name}")
        model_name = db.get_model()
        logging.info(f"Model: {model_name}")
        self.model = failsearch.get_model(model_name, use_gpu=False)
        self.files, image_embeds = db.get_all_data()
        self.image_embeds = failsearch.torch.from_numpy(image_embeds)
    
    def search(self, text, n):
        return failsearch.search(text, self.image_embeds, self.files, self.model, n=n)

    def is_result_file(self, filepath):
        return filepath in self.files



app = flask.Flask(__name__)
_db_name = os.environ.get('DB_NAME', 'failsearch.sqlite')
searcher = Searcher(_db_name)


@app.route("/")
def main():
    #return f"<p>Hello from Failsearch! We have {db.get_num_entries()} entries in the DB</p>"

    text = flask.request.args.get('search')
    n = int(flask.request.args.get('n', 100))
    if text is not None:
        results = searcher.search(text, n=n)
        results_enc = [urllib.parse.quote(r, safe='') for r in results] 
    else:
        results_enc = None
    return flask.render_template('results.html', search_string=text, results=results_enc)


@app.route('/show/<path:imgpath>')
def show_image(imgpath):
    """Shows an image."""
    absolute_path = "/" + imgpath  # Hacky, but the leading '/' gets lost!

    # check whether the requested path is actually an indexed file as a 
    # precaution, otherwise this function could be used to access arbitrary
    # files on the server.
    if searcher.is_result_file(absolute_path):
        return flask.send_file(absolute_path, mimetype='image/png')
    else:
        return None
