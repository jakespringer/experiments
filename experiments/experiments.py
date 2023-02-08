import numpy as np
import re
import os
import sys
import time
import itertools
from contextlib import contextmanager

global_experiment = None
re_brackets = re.compile(r'(.+)\[((?:-?)[0-9]+)\]')
year_in_ms = 31_540_000_000_000

def _reorder_list(lst, ord):
    return [lst[i] for i in ord]

class Experiment(list):
    def __init__(self, id=None, default_params={}, base_dir=None, init_step=False, new_id=True, **kwargs):
        super(Experiment, self).__init__(**kwargs)
        self.id = id
        if base_dir is not None:
            self.base_dir = os.path.abspath(base_dir)
        if new_id:
            new_id = self._find_new_id()
            if new_id is not None:
                self.id = new_id
                print(f'[experiments] Requested id already exists: updating to be {self.id}', file=sys.stderr)
        if base_dir is not None:
            print(f'[experiments] Recording experiment in {os.path.join(self.base_dir, self.id)}', file=sys.stderr)
        self.global_params = default_params
        self.global_params['id'] = self.id
        if init_step: self.step()

    def log(self, **values):
        self[-1].update(values)

    def step(self):
        self.append({'index': len(self), **self.global_params})

    def update_global(self, params):
        self.global_params.update(params)

    def query(self, *vargs, constraints=None):
        if constraints is None:
            constraints = {}
        for v in vargs:
            if v not in constraints:
                constraints[v] = Allow()
        if len(vargs) == 0:
            return [elem for elem in self if Experiment._filter(elem, constraints)]
        else:
            elems = [elem for elem in self if Experiment._filter(elem, constraints)]
            return [[Experiment._get_with_str(v, elem) for elem in elems if Experiment._in_with_str(v, elem)] for v in vargs]

    def _get_with_str(idx, d):
        check_idx = re_brackets.match(idx)
        if check_idx:
            return d[check_idx.group(1)][int(check_idx.group(2))]
        else:
            return d[idx]

    def _in_with_str(idx, d):
        check_idx = re_brackets.match(idx)
        if check_idx:
            return check_idx.group(1) in d
        else:
            return idx in d

    def _filter(elem, constraints):
        if constraints is None:
            return True
        for k, v in constraints.items():
            check_k = re_brackets.match(k)
            if check_k:
                k = check_k.group(1)
                ki = int(check_k.group(2))
                if k not in elem:
                    return False
                elif callable(v):
                    if not v(elem.get(k)[ki]):
                        return False
                elif elem.get(k)[ki] != v:
                    return False
            else:
                if k not in elem:
                    return False
                elif callable(v):
                    if not v(elem.get(k)):
                        return False
                elif elem.get(k) != v:
                    return False
        return True

    def get_param(self, k):
        return self.global_params[k]

    def _create_dir(self):
        path = os.path.join(self.base_dir, self.id)
        os.makedirs(path, exist_ok=True)
        return path

    def _find_new_id(self):
        path = os.path.join(self.base_dir, self.id)
        if not os.path.exists(path):
            return None
        for i in itertools.count():
            if not os.path.exists(f'{path}_{i}'):
                return f'{self.id}_{i}'

    def write_log(self):
        path = self._create_dir()
        np.savez(os.path.join(path, 'log.npz'), np.asarray(self))
        np.savez(os.path.join(path, 'defaults.npz'), **self.global_params)

    def recordz(self, filename=None, prefix=None, **kwargs):
        path = self._create_dir()
        if filename is None:
            filename = 'record' if prefix is None else prefix
            filename = f'{filename}_{(time.time_ns() // 1000) % year_in_ms}.npz'
        np.savez(os.path.join(path, filename), **kwargs)
        return filename

    def file_abspath(self, file):
        return os.path.join(self.base_dir, self.id, file)

class ExperimentGroup(Experiment):
    def __init__(self, *experiments, id=None, default_params={}, **kwargs):
        super(ExperimentGroup, self).__init__(id=id, default_params=default_params, init_step=False, new_id=False, **kwargs)
        for e in experiments:
            self.extend(e)

    def log(self, **values):
        raise Exception('Cannot call log on ExperimentGroup')

    def step(self):
        raise Exception('Cannot call step on ExperimentGroup')

    def append_experiment(self, id):
        path = os.path.join(self.base_dir, ex_id)
        log = np.load(os.path.join(path, 'log.npz'), allow_pickle=True)['arr_0']
        self.extend(log)

    def all_project_experiment_ids(self):
        possible_dirs = os.listdir(self.base_dir)
        dirs = [d for d in possible_dirs if 'defaults.npz' in os.listdir(os.path.join(self.base_dir, d))]
        return dirs

    def read_file(self, id, file, allow_pickle=True):
        return np.load(os.path.join(self.base_dir, id, file))

def Leq(x):
    return lambda y: y <= x

def Geq(x):
    return lambda y: y >= x

def Less(x):
    return lambda y: y < x

def Greater(x):
    return lambda y: y > x

def And(f1, f2):
    return lambda y: f1(y) and f2(y)

def Or(f1, f2):
    return lambda y: f1(y) or f2(y)

def Not(f):
    return lambda y: not f(y)

def Between(x1, x2):
    return And(Geq(x1), Leq(x2))

def Outside(x1, x2):
    return Not(Between(x1, x2))

def In(xs):
    return lambda y: y in xs

def Allow():
    return lambda y: True

def Prevent():
    return lambda y: False

def set_experiment(experiment):
    global global_experiment
    global_experiment = experiment

def get_experiment():
    global global_experiment
    return global_experiment

def get_id():
    return global_experiment.id

@contextmanager
def new_experiment(id, default_params={}, base_dir={}, **kwargs):
    global global_experiment
    old_experiment = global_experiment
    experiment = Experiment(id=id, default_params=default_params, base_dir=base_dir, **kwargs)
    global_experiment = experiment
    try:
        yield experiment
    finally:
        if experiment.base_dir is not None: experiment.write_log()
        global_experiment = old_experiment

def log(**values):
    return global_experiment.log(**values)

def step():
    return global_experiment.step()

def update_default(params):
    global_experiment.global_params.update(params)

def query(*vargs, constraints):
    return global_experiment.query(*vargs, constraints=constraints)

def get_param(k):
    return global_experiment.get_param(k)

def write_log():
    return global_experiment.write_log()

def recordz(**kwargs):
    return global_experiment.recordz(**kwargs)

def file_abspath(file):
    return global_experiment.file_abspath(file)

def load_project(id, base_dir, **kwargs):
    prj = ExperimentGroup(id=id, base_dir=base_dir, **kwargs)
    for ex_id in prj.all_project_experiment_ids():
        prj.append_experiment(ex_id)
    return prj