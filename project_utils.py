import librosa
from librosa import feature
from librosa import display as ldisplay
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn


def visualize_row(row):
  fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15, 15))
  img1 = ldisplay.specshow(row["mfcc"], ax=ax[0], x_axis='time')
  ax[0].set(title='MFCC')
  ax[0].label_outer()
  img2 = ldisplay.specshow(row["delta"], ax=ax[1], x_axis='time')
  ax[1].set(title=r'MFCC-$\Delta$')
  ax[1].label_outer()
  img3 = ldisplay.specshow(row["delta2"], ax=ax[2], x_axis='time')
  ax[2].set(title=r'MFCC-$\Delta^2$')
  fig.colorbar(img1, ax=[ax[0]])
  fig.colorbar(img2, ax=[ax[1]])
  fig.colorbar(img3, ax=[ax[2]])

def oversample(df):
    classes = df.label.value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['label'] == key]) 
    classes_sample = []
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df

def pad_tensor(tensor, max_size):
  target = torch.zeros(13, max_size)
  _, y_shape = tensor.size()
  cut_value = min(y_shape, max_size)
  half_pad = (max_size - cut_value)//2
  target[:, half_pad:cut_value + half_pad] = tensor
  return target

def get_categories(train_df):
  c2i={}
  i2c={}
  categories = sorted(train_df["label"].unique())
  for i, category in enumerate(categories):
    c2i[category]=i
    i2c[i]=category
  return (categories, c2i, i2c)


class SCData(Dataset):
  def __init__(self, df, labels, c2i, i2c, max_length):
    self.df = df
    self.data = []
    self.labels = []
    self.caegories = labels
    self.c2i = c2i
    self.i2c = i2c
    for ind in tqdm(range(len(df))):
      row = df.iloc[ind]
      
      mfcc = pad_tensor(torch.Tensor(row['mfcc']), max_length)
      delta = pad_tensor(torch.Tensor(row['delta']), max_length)
      delta2 = pad_tensor(torch.Tensor(row['delta2']), max_length)

      # print(mfcc.size(), 
      #   delta.size(), 
      #   delta2.size(), 
      #   torch.cat((mfcc , delta, delta2), 0).size()
      #   )

      self.data.append(torch.cat((mfcc , delta, delta2), 0).transpose(dim0=-1, dim1=-2))
      # self.data.append(mfcc)
      self.labels.append(self.c2i[row['label']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

def plot_train_evo(train_loss, val_loss, train_acc, val_acc, name, desc):
  # Create count of the number of epochs
  epoch_count = range(1, len(train_loss) + 1)

  # Visualize loss history
  plt.figure(figsize=(16,12))
  plt.plot(epoch_count, train_loss, 'b-')
  plt.plot(epoch_count, val_loss, 'g-')
  plt.legend(['Training Loss', 'Validation Loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title(name + ": Loss")
  plt.savefig(f'{name}loss_{desc}.png')
  plt.show()

  plt.figure(figsize=(16,12))
  plt.plot(epoch_count, train_acc, 'r--')
  plt.plot(epoch_count, val_acc, 'o--')
  plt.legend(['Train Acc', 'Val acc'])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title(name + ": Accuracy")
  plt.savefig(f'{name}acc_{desc}.png')
  plt.show()

def plot_conf_matrix(model, valid_loader, categories, name, desc, device):
  model = model.to(device)
  yh = []
  ys = []
  for i, (audios, labels) in enumerate(valid_loader):
    audios = audios.to(device)
    labels = labels.to(device)
    y_hat = model(audios)
    y_hat = y_hat.argmax(1)
    y = labels
    yh = yh + y_hat.tolist()
    ys = ys + y.tolist()
    
  df_cm = pd.DataFrame(confusion_matrix(ys, yh), index = categories,
                    columns = categories)
  plt.figure(figsize = (20,18))
  plt.title("Matríz de confusión: "+ name)
  sns_plot = sn.heatmap(df_cm, annot=True, fmt='g')
  sns_plot.figure.savefig(f'{name}confmat_{desc}.png')

#Esta función permite inicializar todas las semillas de números pseudoaleatorios.
# Puedes usar esta función para resetear los generadores de números aleatorios
def iniciar_semillas():
  SEED = 1234

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True


#Calcula el tiempo transcurrido entre dos timestamps
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def test_model(model, criterion, test_loader, device):
  with torch.no_grad():
    epoch_loss = 0
    epoch_acc = 0
    for audios, labels in test_loader:
        audios = audios.to(device)
        labels = labels.to(device)
        outputs = model(audios)

        top_pred = outputs.argmax(1, keepdim=True)
        correct = top_pred.eq(labels.view_as(top_pred)).sum()
        acc = correct.float()/labels.shape[0]

        loss = criterion(outputs, labels)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(test_loader), epoch_acc / len(test_loader)

#Esta función realiza el entrenamiento completo de una red. Recibe como parámetros:
#     -network: la red neuronal
#     -optimizer: el optimizador para entrenamiento
#     -train_loader: el dataloader de datos de entrenamiento
#     -tes_loader: el dataloader de datos de prueba
#     -name: nombre a usar para guardar en disco la red con el mejor accuracy

def train_complete(network, criterion ,optimizer, epochs, train_loader, test_loader, name, device, silent=True):
  
  train_loss_evolution = []
  train_acc_evolution = []
  val_acc_evolution = []
  val_loss_evolution = []

  network = network.to(device)
  criterion = criterion.to(device)

  best_valid_acc = float('-inf')

  for epoch in range(epochs):
    
    start_time = time.time()

    for i, (audios, labels) in enumerate(train_loader):  
      audios = audios.to(device)
      labels = labels.to(device)
      
      outputs = network(audios)
      loss = criterion(outputs, labels)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    #Train + validation cycles  
    train_loss, train_acc = test_model(network, criterion, train_loader, device)
    valid_loss, valid_acc = test_model(network, criterion, test_loader, device)

    train_loss_evolution.append(train_loss)
    train_acc_evolution.append(train_acc)
    val_acc_evolution.append(valid_acc)
    val_loss_evolution.append(valid_loss)

    #Si encontramos un modelo con accuracy de validación mayor, lo guardamos
    if valid_acc > best_valid_acc:
     best_valid_acc = valid_acc
     torch.save(network.state_dict(), f'{name}.pt')
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if (not silent):
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
  
  #Cuando se termina el entrenamiento, cargamos el mejor modelo guardado y calculamos el accuracy de prueba
  network.load_state_dict(torch.load(f'{name}.pt'))

  test_loss , test_acc = test_model(network, criterion, test_loader, device)
  print(f'Test Loss: {test_loss:.3f} | Mejor test acc: {test_acc*100:.2f}%')
  return (train_loss_evolution, train_acc_evolution, val_acc_evolution, val_loss_evolution)

