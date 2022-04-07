
from sklearn.model_selection import train_test_split
import numpy as np


class BootstrapCR():
    def __init__(self, n_bootstraps, make_ncr, make_trainer,
                C_train, X_train, Y_train,
                C_test, X_test, Y_test):
        self.ncrs = []
        self.trainers = []
        self.dataloaders = {"train": [], "val": [], "test": []}
        for i in range(n_bootstraps):
            ncr = make_ncr()
            self.ncrs.append(ncr)
            C_train_train, C_val, X_train_train, X_val, Y_train_train, Y_val = train_test_split(
                C_train, X_train, Y_train, test_size=0.3)
            train_dataloader = ncr.dataloader(C_train_train.values, X_train_train, Y_train_train.values, batch_size=1)
            val_dataloader   = ncr.dataloader(C_val.values, X_val, Y_val.values, batch_size=16)
            test_dataloader = ncr.dataloader(C_test.values, X_test, Y_test.values, batch_size=16)

            self.dataloaders["train"].append(train_dataloader)
            self.dataloaders["val"].append(val_dataloader)
            self.dataloaders["test"].append(test_dataloader)
            self.trainers.append(make_trainer())

    def fit(self):
        for i in range(len(self.ncrs)):
            print("Fitting Bootstrap {} of {}".format(i+1, len(self.ncrs)))
            self.trainers[i].fit(self.ncrs[i], self.dataloaders['train'][i])

    def predict_y(self, dataloader, individual_preds=False):
        preds = np.array([self.trainers[i].predict_y(self.ncrs[i], dataloader) for i in range(len(self.ncrs))])
        if individual_preds:
            return preds
        return np.mean(preds, axis=0)

    def predict_params(self, dataloader, individual_preds=False):
        models = np.array([self.trainers[i].predict_params(self.ncrs[i], dataloader)[0] for i in range(len(self.ncrs))])
        mus = np.array([self.trainers[i].predict_params(self.ncrs[i], dataloader)[1] for i in range(len(self.ncrs))])
        if individual_preds:
            return models, mus
        return np.mean(models, axis=0), np.mean(mus, axis=0)