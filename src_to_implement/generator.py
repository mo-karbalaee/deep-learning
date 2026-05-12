import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                           5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        self.indices = sorted([int(k) for k in self.labels.keys()])
        self.num_samples = len(self.indices)

        self.images = {}
        for idx in self.indices:
            img_path = os.path.join(file_path, '{}.npy'.format(idx))
            self.images[idx] = np.load(img_path)

        self._epoch = 0
        self._current_pos = 0

        self._order = np.array(self.indices, dtype=int)
        if self.shuffle:
            np.random.shuffle(self._order)

    def next(self):
        images_batch = []
        labels_batch = []
        samples_remaining = self.batch_size

        while samples_remaining > 0:
            available = self.num_samples - self._current_pos

            if samples_remaining <= available:
                chunk = self._order[self._current_pos: self._current_pos + samples_remaining]
                self._current_pos += samples_remaining
                samples_remaining = 0
            else:
                chunk = self._order[self._current_pos:]
                samples_remaining -= available
                
                self._epoch += 1
                self._order = np.array(self.indices, dtype=int)
                if self.shuffle:
                    np.random.shuffle(self._order)
                self._current_pos = 0

            for idx in chunk:
                img = self.images[idx].copy()
                target_h, target_w, target_c = self.image_size
                
                if img.shape != (target_h, target_w, target_c):
                    img = resize(img, (target_h, target_w, target_c), anti_aliasing=True, preserve_range=True)
                
                if self.rotation or self.mirroring:
                    img = self.augment(img)
                    
                images_batch.append(img)
                labels_batch.append(int(self.labels[str(idx)]))

        images_out = np.array(images_batch)
        labels_out = np.array(labels_batch, dtype=int)
        return images_out, labels_out

    def augment(self, img):
        if self.mirroring:
            choice = np.random.randint(0, 4)
            if choice == 1:
                img = np.fliplr(img)
            elif choice == 2:
                img = np.flipud(img)
            elif choice == 3:
                img = np.fliplr(np.flipud(img))

        if self.rotation:
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k)

        return img

    def current_epoch(self):
        return self._epoch

    def class_name(self, x):
        return self.class_dict[x]

    def show(self):
        images, labels = self.next()
        n = len(images)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).flatten()
        
        for i, (img, label) in enumerate(zip(images, labels)):
            ax = axes[i]
            display = img.astype(np.float64)
            if display.max() > 1.0:
                display = display / 255.0
            ax.imshow(display)
            ax.set_title(self.class_name(label))
            ax.axis('off')
            
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()
