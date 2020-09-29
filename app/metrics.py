import os
import csv
from pathlib import Path

class Metrics():
    def __init__(self, save_path):
        self.save_path = save_path
        self.create_paths()

    def create_paths(self):
        path = Path(self.save_path)
        if not path.exists():
            path.mkdir(parents=True)
        path = self.save_path + '/metrics.csv'
        if not os.path.exists(path):
            with open(path, "w"):
                pass

    def save(self, metrics):
        path = self.save_path + '/metrics.csv'
        metrics_file = open(path, "a")
        with metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(metrics)