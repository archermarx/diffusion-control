from abc import ABC

import os
import time
import logging
from watchdog.observers.polling import PollingObserver
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor

from forward import ForwardModel

class DataFileHandler(FileSystemEventHandler):

    def __init__(self, observer):
        self.observer = observer
        self.triggered = False

    def on_modified(self, event):
        print("modification detected! stopping observer")
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
    def __init__(self, dir, output_file):
        self.dir = dir
        self.model = ForwardModel(
            "thrusters/h9.json",
            controls = [
                #"anode_mass_flow_rate_kg_s",
                #"discharge_voltage_v",
                "magnetic_field_scale",
            ],
            dataset_dir="inputs",
            verbose=True,
            num_workers=1,
            duration=2e-3,
        )
        self.output_file = output_file
    
    def control_to(self, c):
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
        inputs = [(None, c)]
        files = [path]
        future = exec.submit(self.model, inputs, output_files=files, dir=self.dir)

        try:
            while not data_handler.triggered:
                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

        self.result = future.result()


if __name__ == "__main__":

    controller = SimulationController(
        dir="control_outputs",
        output_file="output.json"
    )

    controller.control_to([1.0])