from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
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
                scalar_data.append((steps, vals))
            except:
                print(f"Error retrieving scalar {scalar}.")

        plt.plot(steps, vals)
        plt.title(f"{scalar} vs steps")
        plt.ylabel(f"{scalar}")
        plt.xlabel("steps")
        plt.show()
        plt.close()
    
    def get_vals(self, scalar):
        try:
            scalar_events = self.data.Scalars(scalar)
            vals = [event.value for event in scalar_events]
        except:
            return (f"Error retrieving scalar {scalar}.")
        
        return vals
    
    def plot_exps(experiments, scalar):
        vals = []
        for experiment in experiments:
            curr_plotter = Plotter(experiment)
            curr_vals = curr_plotter.get_vals(scalar)
            vals.append(curr_vals)
        
        steps = np.arange(len(vals[0]))

        for i in range(len(vals)):
            plt.plot(steps, vals[i], label=str(experiments[i]))
        

        plt.legend()
        plt.title(f"{scalar} vs steps")
        plt.ylabel(f"{scalar}")
        plt.xlabel("steps")
        plt.show()
    
    def plot_perps(perps):
        labels = [entry[0] for entry in perps]
        data = [entry[1] for entry in perps]
        steps = []
        vals = []
        
        for entry in data:
            steps.append([sub_entry[0] for sub_entry in entry])
            vals.append([sub_entry[1] for sub_entry in entry])

        for i in range(len(steps)):
            plt.plot(steps[i], vals[i], label=labels[i])

        plt.legend()
        plt.title("perplexities vs steps")
        plt.ylabel("perplexities")
        plt.xlabel("steps")
        plt.show()
    