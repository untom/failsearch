#!/bin/env python3
"""A Web UI for failsearch."""

import failsearch
import flask
import urllib.parse

app = flask.Flask(__name__)
db = failsearch.Database("./failsearch.sqlite")

@app.route("/")
def hello_world():
    return f"<p>Hello from Failsearch! We have {db.get_num_entries()} entries in the DB</p>"


@app.route('/search/<text>')
def show_search_results(text):
    results = failsearch.search(db, text, n=20)
    results_enc = [urllib.parse.quote(r, safe='') for r in results]
    #esults_enc = results
    return flask.render_template('results.html', searchstring=text, results=results_enc)


@app.route('/show/<path:imgpath>')
def show_image(imgpath):
    absolute_path = "/" + imgpath  # Hacky, but the leading '/' gets lost!
    return flask.send_file(absolute_path, mimetype='image/png')
