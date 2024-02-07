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


app = flask.Flask(__name__)
_db_name = os.environ.get('DB_NAME', 'failsearch.sqlite')
logging.info(f"Using DB: {_db_name}")
db = failsearch.Database(_db_name)


@app.route("/")
def main():
    #return f"<p>Hello from Failsearch! We have {db.get_num_entries()} entries in the DB</p>"

    search = flask.request.args.get('search')
    if search is not None:
        results = failsearch.search(db, search, n=100)
        results_enc = [urllib.parse.quote(r, safe='') for r in results] 
    else:
        results_enc = None
    return flask.render_template('results.html', search_string=search, results=results_enc)


@app.route('/show/<path:imgpath>')
def show_image(imgpath):
    absolute_path = "/" + imgpath  # Hacky, but the leading '/' gets lost!
    return flask.send_file(absolute_path, mimetype='image/png')
