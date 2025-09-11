# live_loss_plot.py
import matplotlib.pyplot as plt
import time
import random

def show_live_loss(num_epochs=20):
    losses = []
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], "b-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    for epoch in range(1, num_epochs+1):
        loss = random.uniform(0.5, 1.5) / epoch  # fake decreasing loss
        losses.append(loss)

        line.set_xdata(range(1, len(losses)+1))
        line.set_ydata(losses)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.5)

    plt.ioff()
    plt.show()
    return losses

if __name__ == "__main__":
    show_live_loss()