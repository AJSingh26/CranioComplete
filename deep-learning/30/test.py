"""Tests the model"""

import argparse
import logging
import math
import os

import data_loader
import torch
from scipy.io import savemat
from torch import nn

import net1 as net
import utils

import scipy.ndimage as nd
import scipy.io as io
import matplotlib

import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import generateMaterials

import numpy
from stl import mesh

parser = argparse.ArgumentParser()
parser.add_argument('--i', default=25, help="Index of example instance for testing")
parser.add_argument('--test_data_dir', default='./data/test_data.mat', help="Path containing the testing dataset")
parser.add_argument('--model_dir', default='./logs/', help="Directory containing the model")


def test_all(model, test_data, test_labels, tesize):
  """
  Takes all the test samples and outputs the completion and the average denoising error
  """

  # set model to evaluation mode
  model.eval()

  test_data.cuda()

  inputs = torch.Tensor(tesize, 1, 30, 30, 30).cuda()
  outputs = torch.Tensor(tesize, 30*30*30).cuda()
  perfect_cubes = torch.Tensor(tesize, 1, 30, 30, 30).cuda()

  for k in range(tesize):
    input = test_data[k]
    input=input.reshape(1,1,30,30,30)
    input = input.float()
    input=input.cuda()
    
    perfect_input = test_labels[k]
    perfect_input = perfect_input.double()
    
    perfect_cubes[k] = perfect_input
    inputs[k] = input
    output=model.forward(input)
    outputs[k]=output

  #inputs=inputs.cuda()
  #outputs=model.forward(inputs)
  #outputs=outputs.double()
  outputs=outputs.reshape(tesize, 1,30,30,30)

  #After computing the outputs, estimate the error on the denoising task
  #error = reconstructed - original input
  err = 0

  for i in range(tesize):
    output = outputs[i]
    
    bin_output = torch.gt(output, 0.5)  # gt means >
    bin_output = bin_output.double()

    perfect_cube = perfect_cubes[i]
    perfect_cube = perfect_cube.double()
    perfect_cube = perfect_cube.cuda()
    noisey_voxels_tensor = torch.ne(bin_output, perfect_cube)  #This will give a binary tensor (1,30,30,30) indicating 1 where the cubes are equal in value and 0 otherwise; --ne means not equal (!=)
    noisey_voxels_idx = torch.nonzero(noisey_voxels_tensor) #matrix containing the voxels in which the reconstruction is different than the original

    dummy = torch.numel(noisey_voxels_idx)  #numel counts the number of elements in the matrix "noisey_voxels_idx" 

    if dummy > 0:
      err = err + noisey_voxels_idx.size()[0]

    else:
      print('no error in this example')

  aa = err/tesize

  te_err = aa * 100 / 13824  #13824 is the total number of voxels in the grid for this specific resolution (24x24x24=13824)
  return te_err


def test_instance(model, i, test_data, test_labels, tesize):
  """
  this function is meant to feed the corrupted 3D test data to the network and save the output in binary format
  -- this output can then be read and visualized in matlab.
  -- computes the reconstruction error and the BCE loss for this instance
  """

  # set model to evaluation mode
  model.eval()
    
  #error = reconstructed - original input
  err = 0
    
  test_labels = test_labels.reshape(tesize, 30*30*30)
  model.encoder.__delitem__(0) #remove the dropout layer

  inputs=torch.Tensor(1,1,30,30,30)
  inputs=inputs.cuda()

  input=test_data[i]
  #Printing size of what I think is the defected skull
  defected_skull = input.view((30, 30, 30)).unsqueeze(dim=0).cpu().detach().numpy()
  print("Saving Defected Skull...")
  #SavePloat_Voxels(defected_skull, "./", 'defected_skull')
  input=input.cuda()
  inputs[0]=input

  #Saving Defected Skull to .stl file
  #print(type(test_data[i]))
  defected_skull_save = test_data[i].cpu().detach().numpy()
  defected_skull_model = VoxelModel(defected_skull_save, generateMaterials(4))
  defected_skull_mesh = Mesh.fromVoxelModel(defected_skull_model)
  defected_skull_mesh.export('defected_skull.stl')

  perfect_cube=test_labels[i]
  #print(perfect_cube.view((30, 30, 30)).dtype)
  #Printing size of what I think is the healthy skull
  healthy_skull = perfect_cube.view((30,30,30)).unsqueeze(dim=0).cpu().detach().numpy()
  print("Saving Ground Truth Skull...")
  #SavePloat_Voxels(healthy_skull, "./", 'ground_truth_skull')
  outputs=model.forward(inputs)
  outputs=outputs.float()

  #Saving Healthy Skull to .stl file
  #print(test_labels[i].view((30, 30, 30)).dtype)
  test_skull = test_labels[i].view((30, 30, 30))
  #print(type(test_skull))
  healthy_skull_save = test_skull.cpu().detach().numpy()
  healthy_skull_model = VoxelModel(healthy_skull_save, generateMaterials(4))
  healthy_skull_mesh = Mesh.fromVoxelModel(healthy_skull_model)
  healthy_skull_mesh.export('healthy_skull.stl')

  outputs=outputs.cuda()
    
  bin_output = torch.gt(outputs, 0.5)  # gt means >
  bin_output = bin_output.double()
    
  perfect_cube = perfect_cube.double()
  perfect_cube = perfect_cube.cuda()
  noisey_voxels_tensor = torch.ne(bin_output, perfect_cube)
  noisey_voxels_idx = torch.nonzero(noisey_voxels_tensor) 

  dummy = torch.numel(noisey_voxels_idx)

  if dummy > 0:
    err = err + noisey_voxels_idx.size()[0]
    err = (err/13824)*100
  else:
    print('no error in this example')
    
  perfect_cube=perfect_cube.float()
  loss_fn = nn.BCELoss()
  bce=loss_fn(outputs, perfect_cube)
    
  return err, bce, outputs


def save_output(outputs, filename):

  """
  Saves the reconstruction output in a .mat file
  """

  outputs=outputs.reshape(30,30,30)
  outputs=torch.squeeze(outputs)

  dims=outputs.ndimension()

  if dims > 1:
    for i in range(math.floor(dims/2)-1):
      outputs=torch.transpose(outputs, i, dims-i-1)
    outputs=outputs.contiguous()
    
  outputs=outputs.data
  outputs=outputs.cpu()
  np_outputs=outputs.numpy()
  np_outputs.astype(float)
  savemat(filename, dict([('output', np_outputs)]))

def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

  args = parser.parse_args()
  i=args.i
  i=int(i)

  # Get the logger
  utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))


  #Load testing data
  #logging.info("Loading the test dataset...")
  test_data, tesize = data_loader.load_data(args.test_data_dir, 'dataset')
  #logging.info("Number of testing examples: {}".format(tesize))

  test_labels, tesize_labels = data_loader.load_data(args.test_data_dir, 'labels')

  #initialize the model
  autoencoder = net.VolAutoEncoder()
  autoencoder.cuda()

  #reload weights of the trained model from the saved file
  utils.load_checkpoint(os.path.join(args.model_dir, 'last.pth.tar'), autoencoder)


  #------test all-----------
  #te_err = test_all(autoencoder, test_data, test_labels, tesize)
  #logging.info("Test All: The test error is {} %".format(te_err))

        
  #------test instance-----------w/ Recons Error and BCE loss
  logging.info("Testing Instance: 58 ---- Loading data...")
  logging.info("Reconstruction Starting...")
  te_err, bce, outputs=test_instance(autoencoder,58,test_data,test_labels,tesize)
  logging.info("Reconstruction Completed")
  logging.info("Calculating Reconstruction results...")
  logging.info("Reconstruction error: {}".format(te_err))
  logging.info("BCE Loss: {}".format(bce))
  #logging.info("Test instance {}: The reconstruction error is {} % and BCE Loss is {}".format(58,te_err,bce))

  outputs = outputs.view((30, 30, 30))
  outputs = torch.round(outputs)
  outputs = outputs.type(torch.cuda.DoubleTensor)
  #print(outputs.dtype)
  print("Saving Reconstructed Skull...")

  reconstructed_skull = outputs.unsqueeze(dim=0).cpu().detach().numpy()
  
  #Saving Reconstructed Skull
  #print(outputs.shape)
  reconstructed_skull_save = outputs.cpu().detach().numpy()
  #print(reconstructed_skull_save.shape)
  reconstructed_skull_model = VoxelModel(reconstructed_skull_save, generateMaterials(4))
  reconstructed_skull_mesh = Mesh.fromVoxelModel(reconstructed_skull_model)
  reconstructed_skull_mesh.export('reconstructed_skull.stl')

  #SavePloat_Voxels(reconstructed_skull, "./", 'reconstructed_skull')

  filename= './recons'+str(i)
  save_output(outputs,filename)
