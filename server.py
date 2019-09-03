#!/usr/bin/env python3

from flask import request, jsonify, json
from flask_api import FlaskAPI
import os

import com.nus.speech.server.config as cfg
from com.nus.speech.server.project_manager import ProjectManager as pm

__author__ = 'Cai Qing, jibin'
__copyright__ = 'Copyright 2019, The speech location '
__version__ = '2019a'
__maintainer__ = 'Cai Qing,  jibin'
__status__ = 'Prototype'

app = FlaskAPI(__name__)


@app.route('/sound/<file_key>', methods=['GET', 'POST'])
def sound_request(file_key):
    print(file_key)

    file_name = str(file_key) + '.mat'
    file_path = os.path.join('Data/all', file_name)
    abs_path = os.path.abspath(__file__).split('/')
    path = ""
    for i in range(len(abs_path) - 1):
        path = path + abs_path[i]
        path = path + "/"
    path += file_path
    # print(path)

    project = pm.get_project(cfg.start_project_id)
    print("project finished")
    args = []
    result = project.get_calculate_results(path, args)
    return json.dumps(result)


@app.route('/location/<file_key>', methods=['GET', 'POST'])
def location_request(file_key):
    file_name = str(file_key) + '.mat'
    file_path = os.path.join('Data/all', file_name)
    abs_path = os.path.abspath(__file__).split('/')
    path = ""
    for i in range(len(abs_path) - 1):
        path = path + abs_path[i]
        path = path + "/"
    path += file_path
    # print(path)
    project = pm.get_project(cfg.start_project_id)
    result = project.get_target_location(path)
    return json.dumps(result)


if __name__ == '__main__':
    print("engine server start: ")
    app.run(port=cfg.PORT, debug=True)
