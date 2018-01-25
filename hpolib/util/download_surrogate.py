import os
import logging
import pickle
import urllib.request as ur
import zipfile
import ast
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class DownloadSurrogate:

    def __init__(self,url, folder="/"):
        self.url = url
        self.X = None
        self.y = None
        self.c = None
        self.N = 200
        self.n_epochs = 100
        self.path = "./"
        self.folder = folder
        self.data = self.download_data()
        self.load_data()



    def download_data(self, file_name="data.zip"):
        ur.urlretrieve(self.url, file_name)
        zf = zipfile.ZipFile(file_name)
        return zf


    def load_configs(self):
        configs = []
        learning_curves = []
        i = 0
        while len(configs) < self.N:
            try:
                res = self.data.read(self.folder + "config_%d.json" % i)
                res = ast.literal_eval(res.decode("utf-8"))
            except KeyError:
                logging.error("Config %d not found" % i)
                i += 1
                continue

            learning_curves.append(res["learning_curve"])
            configs.append(list(res["config"].values()))
            i += 1

        configs = np.array(configs)
        learning_curves = np.array(learning_curves)
        return configs, learning_curves


    def load_timesteps(self):
        learning_curve_time_step = []
        i = 0
        while len(learning_curve_time_step) < self.N:
            try:
                res = self.data.read(self.folder + "config_%d.json" % i)
                res = ast.literal_eval(res.decode("utf-8"))
            except KeyError:
                logging.error("Config %d not found" % i)
                i += 1
                continue
            learning_curve_time_step.append(res["time_steps"])
            i += 1

        learning_curve_time_step = np.array(learning_curve_time_step)
        return learning_curve_time_step


    def load_data(self):
        configs, learning_curves = self.load_configs()
        time_steps = self.load_timesteps()
        t_idx = np.arange(1, self.n_epochs + 1)
        for i in range(self.N):
            x = np.repeat(configs[i, None, :], t_idx.shape[0], axis=0)
            x = np.concatenate((x, t_idx[:, None]), axis=1)

            # lc = learning_curves[i][:]
            lc = np.repeat(learning_curves[i, None, :], t_idx.shape[0], axis=0)

            # time_step_lc = time_steps[i][:]
            time_step_lc = np.repeat(time_steps[i, None, :], t_idx.shape[0], axis=0)
            time_step_lc[1:] -= time_step_lc[:-1].copy()

            if self.X is None:
                self.X = x
                self.y = lc
                self.c = time_step_lc

            else:
                self.X = np.concatenate((self.X, x), 0)
                self.y = np.concatenate((self.y, lc), 0)
                self.c = np.concatenate((self.c, time_step_lc), 0)



    def get_learning_curve_rf(self):
        rf = RandomForestRegressor(n_estimators=10)
        rf.fit(self.X, self.y)
        return rf


    def get_time_cost_rf(self):
        time_step_rf = RandomForestRegressor(n_estimators=10)
        time_step_rf.fit(self.X, self.c)
        return time_step_rf

