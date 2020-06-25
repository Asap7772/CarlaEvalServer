from flask import Flask
from flask import render_template, url_for, request, redirect
import copy
import os

import gym
import numpy as np
import torch.nn as nn
import d4rl
import d4rl.carla
import argparse
import threading
import time
import torch
import hashlib
import pickle

from railrl.torch.core import eval_np
import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector.path_collector import \
    ObsDictPathCollector, CustomObsDictPathCollector, MdpPathCollector, CustomMdpPathCollector
from collections import defaultdict
from convolution import ConvNet, TanhGaussianConvPolicy
from railrl.torch.sac.policies import MakeDeterministic

def enable_gpus(gpu_str):
    if gpu_str is not "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

enable_gpus('2')
ptu.set_gpu_mode(True)

app = Flask(__name__)
procs = []
env = gym.make('carla-lane-render-v0')
hash_val = set()
out=defaultdict(list)
stop_thread = False

def load_path(path):
    #check if file exists
    param_path = path + '/params.pkl'
    if param_path is None or not os.path.exists(param_path) or not os.path.isfile(param_path):
        return
    
    #check if file changed
    hasher = hashlib.md5()
    with open(param_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
        h = hasher.hexdigest()
        if h in hash_val:
            return
        hash_val.add(h)
    
    #load policy
    data = pickle.load(open(param_path, 'rb'))
    policy = data['evaluation/policy'].stochastic_policy
    policy.cuda()
    policy.eval()
    
    #path collector
    eval_path_collector = MdpPathCollector(
        env,
        MakeDeterministic(policy),
        sparse_reward=False,
    )
    paths = eval_path_collector.collect_new_paths(
        max_path_length=250,
        num_steps=1000,
        discard_incomplete_paths=True,
    )

    #calculate average return
    avg_return = 0
    for i in range(len(paths)):
        rewards = paths[i]['rewards']
        cum_rewards = np.cumsum(rewards)
        discounted_rewards = 0.9 ** np.arange(cum_rewards.shape[0])
        discounted_rewards = discounted_rewards * cum_rewards
        avg_return += np.sum(discounted_rewards)
    out[path].append(avg_return/len(paths))

@app.before_first_request
def activate_job():
    def run_job():
        while not stop_thread:
            print('Running carla eval script')
            for path in procs:
                load_path(path)
            time.sleep(3) 
    
    thread = threading.Thread(target=run_job)
    thread.start()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/addpath', methods=['GET', 'POST'])
def addpath():
    path_pkl=request.form['text']
    procs.append(path_pkl)
    print(path_pkl)
    return render_template('home.html', path_added=True, listp=False, clear=False, stop=False, path=path_pkl)     

@app.route('/list_proc', methods=['GET', 'POST'])
def list_proc():
    print('List Processes', procs)
    return render_template('home.html', path_added=False,listp=True, clear=False, stop=False, procs=procs)     

@app.route('/clear_proc', methods=['GET', 'POST'])
def clear_proc():
    procs.clear()
    print('Clear Processes')
    return render_template('home.html',path_added=False, listp=False, clear=True, stop=False)     

def pickle_return():
    for x in out.keys():
        print(x, out[x])
    import datetime; d = datetime.datetime.today() 
    with open(str(d)+'.pickle','wb') as f:
        pickle.dump(out, f,protocol=pickle.HIGHEST_PROTOCOL)


@app.route('/stop_proc', methods=['GET', 'POST'])
def stop_proc():
    print('Stopped Process')
    print(out)
    global stop_thread
    stop_thread = True

    pickle_return()

    return render_template('home.html',path_added=False, listp=False, clear=False, stop=True)     

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

