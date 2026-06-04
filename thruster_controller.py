from abc import ABC

import os
import numpy as np
import json
import time
import logging
from watchdog.observers.polling import PollingObserver
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor

from hall_diffusion.utils.thruster_data import ThrusterDataset, invert_fft_vector

from forward import ForwardModel

class DataFileHandler(FileSystemEventHandler):

    def __init__(self, observer):
        self.observer = observer
        self.triggered = False

    def on_modified(self, event):
        print("File modification detected! stopping observer")
        self.observer.stop()
        self.triggered = True

class ThrusterController(ABC):
    def __init__(self):
        pass

    def control_to(self, c):
        # Control flow rate, discharge voltage, magnet currents to target c
        pass

    def take_data(self):
        # Take o-scope, plasma probe, etc data
        # TODO: figure out interface
        pass

class SimulationController(ThrusterController):
    def __init__(self, dir, data_file):
        self.dir = dir
        self.output_file = data_file
        self.model = None
        self.setpoint = None
    
    def control_to(self, c):
        control_keys = list(c.keys())
        self.setpoint = list(c.values())
        self.model = ForwardModel(
            "thrusters/h9.json",
            controls = control_keys,
            dataset_dir="inputs",
            verbose=True,
            num_workers=1,
            duration=2e-3,
        )

    def take_data(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        path = self.output_file
        observer = PollingObserver(timeout=0.1)
        data_handler = DataFileHandler(observer)
        observer.schedule(LoggingEventHandler(), path, recursive=True)
        observer.schedule(data_handler, path, recursive=True)
        observer.start()
        
        exec = ThreadPoolExecutor(max_workers=1)
        inputs = [(None, self.setpoint)]
        future = exec.submit(self.model, inputs, output_files=[path], dir=self.dir)

        try:
            while not data_handler.triggered:
                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

        _, data_fourier = future.result()[0]

        times = np.linspace(0, 1e-3, 1000)
        data_timedomain = invert_fft_vector(times, data_fourier)

        data_dict = {
            "discharge_current_fourier": data_fourier.tolist(),
            "discharge_current_time": times.tolist(),
            "discharge_current_signal": data_timedomain.tolist(),
        }

        return data_dict

if __name__ == "__main__":
    controller = SimulationController(
        dir="control_outputs",
        data_file="output.json"
    )

    controller.control_to({
        "magnetic_field_scale": 1.0,
    })

    data = controller.take_data()

    print(data)

