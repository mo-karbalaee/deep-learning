import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:

    def __init__(self,
                 model,
                 crit,
                 optim=None,
                 train_dl=None,
                 val_test_dl=None,
                 cuda=True,
                 early_stopping_patience=-1,
                 scheduler=None):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._scheduler = scheduler

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,
              x,
              fn,
              export_params=True,
              opset_version=10,
              do_constant_folding=True,
              input_names = ['input'],
              output_names = ['output'],
              dynamic_axes={'input' : {0 : 'batch_size'},
                            'output' : {0 : 'batch_size'}},
              dynamo=False)

    def train_step(self, x, y):
        self._optim.zero_grad()
        output = self._model(x)
        loss = self._crit(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):
        output = self._model(x)
        loss = self._crit(output, y)
        return loss.item(), output

    def train_epoch(self):
        self._model.train()
        total_loss = 0.0
        batches = 0
        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            total_loss += self.train_step(x, y)
            batches += 1
        return total_loss / max(batches, 1)

    def val_test(self):
        self._model.eval()
        total_loss = 0.0
        batches = 0
        predictions = []
        labels = []
        with t.no_grad():
            for x, y in self._val_test_dl:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                loss, output = self.val_test_step(x, y)
                total_loss += loss
                batches += 1
                predictions.append(output.cpu())
                labels.append(y.cpu())

        predictions = t.cat(predictions, dim=0).numpy()
        labels = t.cat(labels, dim=0).numpy()
        predictions = (predictions > 0.5).astype(int)

        avg_loss = total_loss / max(batches, 1)
        f1 = f1_score(labels, predictions, average='macro')
        print('val loss: {:.4f} | mean F1: {:.4f}'.format(avg_loss, f1))
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        val_losses = []
        best_val_loss = None
        epochs_without_improvement = 0
        epoch = 0

        while True:
            if epochs > 0 and epoch >= epochs:
                break

            train_loss = self.train_epoch()
            val_loss = self.val_test()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self._scheduler is not None:
                self._scheduler.step(val_loss)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(epoch)
            else:
                epochs_without_improvement += 1

            if 0 < self._early_stopping_patience <= epochs_without_improvement:
                break

            epoch += 1

        return train_losses, val_losses
