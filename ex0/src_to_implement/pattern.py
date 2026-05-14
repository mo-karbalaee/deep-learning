import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        tile = np.zeros((self.tile_size, self.tile_size), dtype=np.float64)
        block = np.block([
            [tile, tile + 1],
            [tile + 1, tile]
        ])
        
        n_tiles = self.resolution // (2 * self.tile_size)
        self.output = np.tile(block, (n_tiles, n_tiles))
        return self.output.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x_vals = np.arange(self.resolution)
        y_vals = np.arange(self.resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        cx, cy = self.position
        self.output = ((X - cx) ** 2 + (Y - cy) ** 2) <= (self.radius ** 2)
        return self.output.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        n = self.resolution
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)
        
        self.output = np.stack([X, Y, 1 - X], axis=2)
        return self.output.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output)
        plt.show()
