"""
Definition of functions for training / eval of DL models
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(model, optimizer, loss_fn, train_loader, test_loader=None, lr_scheduler=None,
          lr_scheduler_on_metric=None, n_epochs=1,plot_loss=True, n_prints=10, n_evals=0):
    """
    Training a model for n_epochs, with n_evals validations (n_evals < n_epochs)
    """
    train_losses = []
    val_losses = []
    batch_print = len(train_loader) // (n_prints)
    period_eval = n_epochs // n_evals if n_evals > 0 else 0
    epochs_eval = []
    last_lr = 0
    lr_values, epochs_change_lr = [], []
    lr_sched = lr_scheduler_on_metric if lr_scheduler_on_metric else lr_scheduler

    for epoch in range(n_epochs):
        print(f"--- Epoch {epoch + 1} ---")

        model.train()
        train_loss = count = 0

        for batch, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.df(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1

            if batch % batch_print == 0:
                print(f"[{epoch + 1:5d}: {batch + 1:4d}] | Training Loss: {loss.item():0.4f} ||")

        train_losses.append(train_loss / count)

        if n_evals > 0 and epoch % period_eval == 0:

            epochs_eval.append(epoch)
            val_loss = eval(model, test_loader, loss_fn)
            val_losses.append(val_loss)
            print(f"<<<<<<<    Val loss: {val_losses[-1]:0.4f}      >>>>>>>")

            if lr_scheduler_on_metric:
                lr_sched.step(val_loss)

            elif lr_scheduler:
                lr_sched.step()

            if lr_sched:
                lr_value = lr_sched.get_last_lr()[0]
                if last_lr != lr_value:
                    last_lr = lr_value
                    lr_values.append(lr_value)
                    epochs_change_lr.append(epoch)
                    print(f"<><><>      New lr : {lr_value:.1e}      <><><>")
        
    if plot_loss:
        plot_losses(train_losses, val_losses, epochs_eval, lr_values, epochs_change_lr)

def eval(model, test_loader, loss_fn):
    """ Evaluation on test dataset, returns validation loss """

    model.eval()
    with torch.no_grad():

        val_loss = 0
        for batch, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs.to(device))
            val_loss += loss_fn(outputs, labels.to(device)).item()

    return val_loss / (batch + 1)


def plot_losses(train_losses, val_losses, epochs_eval, lr_values=None, epochs_change_lr=None):
    """ Plots train and validation losses"""

    plt.plot(np.arange(len(train_losses)), train_losses, c="blue", label="Train loss")
    plt.plot(epochs_eval, val_losses, c="orange", label="Val loss")

    for epoch in epochs_change_lr:
        plt.axvline(x=epoch, color='red', linestyle='--', alpha=0.7)

    # Annotate the learning rates
    for epoch, lr in zip(epochs_change_lr, lr_values):  # lrs[1:] because the first lr is before any change
        plt.text(epoch, plt.gca().get_ylim()[0] + 0.75 * (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]),
                 f'LR={lr:.1e}', color='red',fontsize=10,verticalalignment='bottom', rotation=90)

    plt.legend()
    plt.show()

def predict(model, X, batch_size=64):
    """ Computes classification prediction of model on input X """

    N_batches = len(X) // batch_size
    y_pred = []
    X = torch.Tensor(X).to(device)
    with torch.no_grad():
        for batch in range(N_batches):
            y_pred.append(model(X[batch * batch_size: (batch + 1) * batch_size]))
        if N_batches * batch_size < len(X):
            y_pred.append(model(X[N_batches * batch_size:]))
    return torch.vstack(y_pred).argmax(axis=1)

def compute_accuracy(model, eval_loader, n_epochs=1, s=0.5):

    model.eval()
    with torch.no_grad():

        y_preds = []
        for epoch in range(n_epochs):

            y_pred = []
            if epoch == 0:
                y_test = []
            for batch, (inputs, labels) in enumerate(eval_loader):
                outputs = model(inputs.to(device))
                outputs = outputs.cpu().squeeze(1) # binary classification
                y_pred.append(outputs)
                if epoch == 0:
                    y_test.append(labels.squeeze(1))
            
            y_preds.append(torch.cat(y_pred))
        
    y_preds = torch.tensor(np.array(y_preds))
    y_pred = y_preds.mean(axis=0)
    y_pred = (y_pred >= s).int()
    y_test = torch.cat(y_test)

    pos_proportion = y_pred.sum() / len(y_pred)
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Proportion of positive predictions : {pos_proportion:.2f}")

    return accuracy, y_pred

def CNN_kaggle_eval(model, eval_loader, n_epochs=1, s=0.5):

    model.eval()
    with torch.no_grad():

        y_preds = []
        for epoch in range(n_epochs):

            y_pred = []
            ids_list = []

            for inputs, ids in eval_loader:
                outputs = model(inputs.to(device))
                outputs = outputs.cpu().squeeze(1) # binary classification
                y_pred.append(outputs)
                ids_list += list(ids)
            
            y_preds.append(torch.cat(y_pred))
        
    y_preds = torch.tensor(np.array(y_preds))
    y_pred = y_preds.mean(axis=0)
    y_pred = (y_pred >= s).int()

    return y_pred, ids_list