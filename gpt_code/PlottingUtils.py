from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os

class Plotter:
    def __init__(self, model_id) -> None:
        log_path = os.path.join("./logs", model_id)
        self.model_id = model_id
        self.data = EventAccumulator(log_path)
        self.data.Reload()

    def print_potential_scalars(self):
        scalar_names = self.data.Tags()['scalars']
        scalars = []
        for name in scalar_names:
            scalars.append(name)
        if len(scalars) == 0:
            print("No data. Please check if you uploaded the right file.")
        print(scalars)
    
    def plot_scalars(self, scalars):
        scalar_data = []
        for scalar in scalars:
            try:
                scalar_events = self.data.Scalars(scalar)
                steps = [event.step for event in scalar_events]
                vals = [event.value for event in scalar_events]
                print(vals)
                scalar_data.append((steps, vals))
            except:
                print(f"Error retrieving scalar {scalar}.")

        plt.plot(steps, vals)
        plt.title(f"{scalar} vs steps")
        plt.ylabel(f"{scalar}")
        plt.xlabel("steps")
        plt.show()
        plt.close()
