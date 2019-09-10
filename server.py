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


@app.route('/sound', methods=['GET', 'POST'])
def sound_request():
    print("get request")
    # print(path)

    project = pm.get_project(cfg.start_project_id)
    print("project finished")
    # args = []
    result = project.get_calculate_results()
    return json.dumps(result)


if __name__ == '__main__':
    print("engine server start: ")
    app.run(port=cfg.PORT, debug=True)
