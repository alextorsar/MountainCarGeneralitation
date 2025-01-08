import numpy as np

class Tile_functions:
    def __init__(self, num_tilesets, num_tiles_per_dim, low, high):
        self.num_tilesets = num_tilesets
        self.num_tiles_per_dim = num_tiles_per_dim
        self.low = np.array(low)
        self.high = np.array(high)
        self.tile_width = (self.high - self.low) / self.num_tiles_per_dim
        self.offsets = (np.arange(self.num_tilesets) / self.num_tilesets)[:, None] * self.tile_width
        self.iht_size = 4096

    def get_tiles(self, state):
        state = np.array(state)
        tiles = []
        for offset in self.offsets:
            # Ajustar el estado con el offset y escalar por el tamaño de la tesela
            indices = np.floor((state - self.low + offset) / self.tile_width).astype(int)
            # Convertir los índices a un hash único
            tiles.append(self.hash_tile(tuple(indices)))
        return tiles
    
    def hash_tile(self, indices):
        # Convierte una tupla de índices a un identificador único
        return hash(indices) % self.iht_size
