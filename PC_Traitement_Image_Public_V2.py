# coding=utf-8
import os
import time
import numpy as np
import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter.messagebox import askyesno
from tkinter.font import nametofont
import cv2
from threading import Thread
from datetime import datetime
import argparse
import sys
import math
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
from PIL import ImageEnhance


#  Logiciel de traitement de videos post acquisition Copyright Alain Paillou 2018-2019  #


# Select the image for treatment

flag_noir_blanc = 0 # Set if colour or monochrome video : 0 is colour video   1 is monochrome video
Video_Test ='D:\Alain\Astro\Soft\SkyNano\Images\Image.jpg4' # The path to your image


# Répertoires de sauvegarde sur PC
image_path = 'D:\Alain\Astro\Soft\SkyNano\Images' # Set the path to your Images directory
video_path= 'D:\Alain\Astro\Soft\SkyNano\Videos' # Set the path to your Video directory


# Titre programme
titre = "Traitement Images PC V1.0 Python OpenCVCUDA - Copyright Alain PAILLOU 2018-2020"

# Choix de la qualité des sauvegardes Imges et videos
flag_HQ = 0 # si 1 : sauvegarde HQ non compressée    si 0 : sauvegarde LQ compressée

# Initialisation des constantes d'exposition mode rapide
exp_min=100 #µs
exp_max=10000 #µs
exp_delta=100 #µs
exp_interval=2000

#  Initialisation des paramètres fixés par les boites scalebar
val_resolution = 1
mode_BIN=1
res_x_max = 3096
res_y_max = 2080
res_cam_x = 3096
res_cam_y = 2080
cam_displ_x = 1266
cam_displ_y = 950
val_exposition = exp_min #  temps exposition en µs
val_gain = 100
val_denoise = 0.7
val_histo_min = 0
val_histo_max = 255
val_contrast_CLAHE = 1.5
val_phi = 2.0
val_theta = 125
val_heq2 = 1.0
text_info1 = "Test information"
val_nb_captures = 1
nb_cap_video =0
val_nb_capt_video = 100
val_nb_darks = 5
dispo_dark = 'Dark NON dispo'
FlipV = 0
FlipH = 0
ImageNeg = 0
val_red = 100
val_blue = 100
val_green = 100
fw_position = 0
val_FS = 1
compteur_FS = 0
val_denoise_KNN = 0.35
val_USB = 75
val_SGR = 90
val_AGR = 20
val_NGB = 5
val_SAT = 1.0
nb_erreur = 0
nb_erreur = 0
text_TIP = ""
val_ampl = 1.0
val_deltat = 0
timer1 = 0.0

# Initialisation des filtres soft
flag_2DConv = 0
flag_gaussian = 0
flag_bilateral = 0
flag_full_res = 0
flag_sharpen_soft1 = 0
flag_unsharp_mask = 0
flag_denoise_soft = 0
flag_histogram_equalize1 = 0
flag_histogram_equalize2 = 0
flag_histogram_stretch = 0
flag_histogram_phitheta = 0
flag_contrast_CLAHE = 0
#flag_noir_blanc = 0
flag_AmpSoft = 0
flag_acquisition_en_cours = False
flag_autorise_acquisition = False
flag_image_disponible = False
flag_premier_demarrage = True
flag_BIN2 = False
flag_BIN3 = False
flag_cap_pic = False
flag_sub_dark = False
flag_cap_video = False
flag_acq_rapide = True
flag_colour_camera = True
flag_filter_wheel = False
flag_seuillage_PB = False
Im1OK = False
Im2OK = False
Im3OK = False
filter_on = False
flag_filtrage_ON = True
flag_filtre_work = 0
flag_denoise_KNN = False
flag_denoise_Paillou = False
flag_GR = False
flag_TIP = False
flag_cross = False
flag_SAT = False
flag_pause_video = False

mod = SourceModule("""
__global__ void Mono_ampsoft_GPU(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, float val_ampl)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  
  index = i * width + j;
  
  if (index < nb_pixels) {
      dest_r[index] = (int)(min(max(int(img_r[index] * val_ampl), 0), 255));
    } 
}
""")

Mono_ampsoft_GPU = mod.get_function("Mono_ampsoft_GPU")


mod = SourceModule("""
__global__ void Colour_ampsoft_GPU(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, float val_ampl)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  
  index = i * width + j;
  
  if (index < nb_pixels) {
      dest_r[index] = (int)(min(max(int(img_r[index] * val_ampl), 0), 255));
      dest_g[index] = (int)(min(max(int(img_g[index] * val_ampl), 0), 255));
      dest_b[index] = (int)(min(max(int(img_b[index] * val_ampl), 0), 255));
    } 
}
""")

Colour_ampsoft_GPU = mod.get_function("Colour_ampsoft_GPU")


mod = SourceModule("""
__global__ void Denoise_Paillou_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, int cell_size, int sqr_cell_size)
{    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
     
    long int index1;
    long int index2;
    int delta;
    //float3 corr = {0, 0, 0};
    float3 Grd = {0, 0, 0};
    float3 Mean = {0, 0, 0};
    float3 Delta =  {0, 0, 0};

    delta = (int)(abs(cell_size/2));
    index1 = ix + iy * imageW;
    
    if(ix<=(imageW-cell_size) && ix > delta && iy<=(imageH-cell_size) && iy > delta){
        // Dead pixels detection and correction
        for(float n = -delta; n <= delta; n++)
            for(float m = -delta; m <= delta; m++) {
                index2 = ix + m + (iy + n) * imageW;
                Grd.x += img_r[index1]-img_r[index2];
                Grd.y += img_g[index1]-img_g[index2];
                Grd.z += img_b[index1]-img_b[index2];
                Mean.x += img_r[index2];
                Mean.y += img_g[index2];
                Mean.z += img_b[index2];
                }
        Delta.x = (Grd.x / (sqr_cell_size * (1.0 + Grd.x/Mean.x))*(-0.00392157 * img_r[index1] +1.0));
        Delta.y = (Grd.y / (sqr_cell_size * (1.0 + Grd.y/Mean.y))*(-0.00392157 * img_g[index1] +1.0));
        Delta.z = (Grd.z / (sqr_cell_size * (1.0 + Grd.z/Mean.z))*(-0.00392157 * img_b[index1] +1.0));
        if (dest_r[index1] > abs(Delta.x) && dest_g[index1] > abs(Delta.y) && dest_b[index1] > abs(Delta.z)) {
            dest_r[index1] = (int)(min(max(int(img_r[index1] - Delta.x), 0), 255));
            dest_g[index1] = (int)(min(max(int(img_g[index1] - Delta.y), 0), 255));
            dest_b[index1] = (int)(min(max(int(img_b[index1] - Delta.z), 0), 255));
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
            dest_g[index1] = int((img_g[ix - 1 + iy * imageW] + img_g[ix + 1 + iy * imageW] + img_g[ix + (iy-1) * imageW] + img_g[ix + (iy+1) * imageW])/4.0);
            dest_b[index1] = int((img_b[ix - 1 + iy * imageW] + img_b[ix + 1 + iy * imageW] + img_b[ix + (iy-1) * imageW] + img_b[ix + (iy+1) * imageW])/4.0);
        }
    }
}
""")

Denoise_Paillou_Colour = mod.get_function("Denoise_Paillou_Colour")


mod = SourceModule("""
__global__ void Denoise_Paillou_Mono(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, int cell_size, int sqr_cell_size)
{ 
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
     
    long int index1;
    long int index2;
    int delta;
    //float3 corr = {0, 0, 0};
    float Grd = 0;
    float Mean = 0;
    float Delta =  0;

    delta = (int)(abs(cell_size/2));
    index1 = ix + iy * imageW;
    
    if(ix<=(imageW-cell_size) && ix > delta && iy<=(imageH-cell_size) && iy > delta){
        // Dead pixels detection and correction
        for(float n = -delta; n <= delta; n++)
            for(float m = -delta; m <= delta; m++) {
                index2 = ix + m + (iy + n) * imageW;
                Grd += img_r[index1]-img_r[index2];
                Mean += img_r[index2];
                }
        Delta = (Grd / (sqr_cell_size * (1.0 + Grd/Mean))*(-0.00392157 * img_r[index1] +1.0));
        if (dest_r[index1] > abs(Delta)) {
            dest_r[index1] = (int)(min(max(int(img_r[index1] - Delta), 0), 255));
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
        }
    }
}
""")

Denoise_Paillou_Mono = mod.get_function("Denoise_Paillou_Mono")



mod = SourceModule("""
__global__ void Histo_Mono(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
int flag_histogram_stretch, float val_histo_min, float val_histo_max, int flag_histogram_equalize2, float val_heq2, int flag_histogram_phitheta,
float val_phi, float val_theta)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  
  index = i * width + j;
  
  if (index < nb_pixels) {
  if (flag_histogram_phitheta == 1) {
      dest_r[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_r[index]-val_theta)/32.0))));
      img_r[index] = dest_r[index];
    }
  if (flag_histogram_equalize2 == 1 ) {
      dest_r[index] = (int)(255.0*__powf(((img_r[index]) / 255.0),val_heq2));
      img_r[index] = dest_r[index];
    }
  if (flag_histogram_stretch == 1 ) {
      dest_r[index] = (int)(min(max(int((img_r[index]-val_histo_min)*(255.0/(val_histo_max-val_histo_min))), 0), 255));
      
    }    
  }
}
""")

Histo_Mono = mod.get_function("Histo_Mono")


mod = SourceModule("""
__global__ void Sharp_Mono(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
 int flag_sharpen_soft1, int flag_unsharp_mask)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  float red;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;

  #define filterWidth 5
  #define filterHeight 5

  index = i * width + j;

  if(index < nb_pixels){
  if (flag_sharpen_soft1 == 1) {
    img_r[index] = dest_r[index];
    float filter[filterHeight][filterWidth] =
    {
      -1, -1, -1, -1, -1,
      -1,  2,  2,  2, -1,
      -1,  2,  8,  2, -1,
      -1,  2,  2,  2, -1,
      -1, -1, -1, -1, -1,
    };
    factor = 1.0 / 8.0;
  
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    }
  if (flag_unsharp_mask == 1) {
    img_r[index] = dest_r[index];
    float filter[filterHeight][filterWidth] =
    {
      1,  4,  6,  4,  1,
      4, 16, 24, 16,  4,
      6, 24, 36, 24,  6,
      4, 16, 24, 16,  4,
      1,  4,  6,  4,  1,
    };
    factor = 1.0 / 256.0;
  
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(img_r[index] * 1.5 - (red * factor * 0.5)), 0), 255));
    }
  }
}
""")

Sharp_Mono = mod.get_function("Sharp_Mono")


mod = SourceModule("""
__global__ void Smooth_Mono(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
int flag_2DConv, int flag_gaussian)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  float red;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;

  #define filterWidth 5
  #define filterHeight 5

  index = i * width + j;

  if(index < nb_pixels){
  if (flag_2DConv == 1) {
    float filter[filterHeight][filterWidth] =
    {
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
    };
    
    factor = 1.0 / 25.0;
      
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    }
  if (flag_gaussian == 1) {
    img_r[index] = dest_r[index];
    float filter[filterHeight][filterWidth] =
    {
      1,  4,  6,  4,  1,
      4, 16, 24, 16,  4,
      6, 24, 36, 24,  6,
      4, 16, 24, 16,  4,
      1,  4,  6,  4,  1,
    };
    factor = 1.0 / 256.0;
  
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    }
  }
}
""")

Smooth_Mono = mod.get_function("Smooth_Mono")



mod = SourceModule("""
__global__ void Histo_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, int flag_histogram_stretch, float val_histo_min, float val_histo_max, int flag_histogram_equalize2,
float val_heq2, int flag_histogram_phitheta, float val_phi, float val_theta)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  long int delta_histo = val_histo_max-val_histo_min;
  
  index = i * width + j;
  
  if (index < nb_pixels) {
  if (flag_histogram_phitheta == 1) {
      dest_r[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_r[index]-val_theta)/32.0))));
      dest_g[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_g[index]-val_theta)/32.0))));
      dest_b[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_b[index]-val_theta)/32.0))));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    }
  if (flag_histogram_equalize2 == 1 ) {
      dest_r[index] = (int)(255.0*__powf(((img_r[index]) / 255.0),val_heq2));
      dest_g[index] = (int)(255.0*__powf(((img_g[index]) / 255.0),val_heq2));
      dest_b[index] = (int)(255.0*__powf(((img_b[index]) / 255.0),val_heq2));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    } 
  if (flag_histogram_stretch == 1 ) {
      dest_r[index] = (int)(min(max(int((img_r[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      dest_g[index] = (int)(min(max(int((img_g[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      dest_b[index] = (int)(min(max(int((img_b[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
    }
  }
}
""")


Histo_Color = mod.get_function("Histo_Color")

mod = SourceModule("""
__global__ void Set_RGB(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, float val_red, float val_green, float val_blue)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  
  index = i * width + j;
  
  if (index < nb_pixels) {
      if (val_blue != 1.0) {
          dest_r[index] = (int)(min(max(int(img_r[index] * val_blue), 0), 255));
          }
      if (val_green != 1.0) {        
          dest_g[index] = (int)(min(max(int(img_g[index] * val_green), 0), 255));
          }
      if (val_red != 1.0) {  
          dest_b[index] = (int)(min(max(int(img_b[index] * val_red), 0), 255));
          }
    } 
}
""")


Set_RGB = mod.get_function("Set_RGB")

mod = SourceModule("""
__global__ void Smooth_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, int flag_2DConv,int flag_gaussian)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  float red;
  float green;
  float blue;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;
  
  index = i * width + j;

  #define filterWidth 5
  #define filterHeight 5

  if(index < nb_pixels){
  if (flag_2DConv == 1) {
    float filter[filterHeight][filterWidth] =
    {
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
    };
    
    factor = 1.0 / 25.0;
      
    red = 0.0;
    green = 0.0;
    blue = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    dest_g[index] = (int)(min(max(int(factor * green), 0), 255));
    dest_b[index] = (int)(min(max(int(factor * blue), 0), 255));
    }

  if (flag_gaussian == 1) {
    img_r[index] = dest_r[index];
    img_g[index] = dest_g[index];
    img_b[index] = dest_b[index];
    float filter[filterHeight][filterWidth] =
    {
      1,  4,  6,  4,  1,
      4, 16, 24, 16,  4,
      6, 24, 36, 24,  6,
      4, 16, 24, 16,  4,
      1,  4,  6,  4,  1,
    };
    factor = 1.0 / 256.0;
  
    red = 0.0;
    green = 0.0;
    blue = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    dest_g[index] = (int)(min(max(int(factor * green), 0), 255));
    dest_b[index] = (int)(min(max(int(factor * blue), 0), 255));
    }
  }
}
""")

Smooth_Color = mod.get_function("Smooth_Color")


mod = SourceModule("""
__global__ void Sharp_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, int flag_sharpen_soft1, int flag_unsharp_mask)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int nb_pixels = width*height;
  float red;
  float green;
  float blue;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;
  
  index = i * width + j;

  #define filterWidth 5
  #define filterHeight 5

  if(index < nb_pixels){
  if (flag_sharpen_soft1 == 1) {
    img_r[index] = dest_r[index];
    img_g[index] = dest_g[index];
    img_b[index] = dest_b[index];
    float filter[filterHeight][filterWidth] =
    {
      -1, -1, -1, -1, -1,
      -1,  2,  2,  2, -1,
      -1,  2,  8,  2, -1,
      -1,  2,  2,  2, -1,
      -1, -1, -1, -1, -1,
    };
    factor = 1.0 / 8.0;
  
    red = 0.0;
    green = 0.0;
    blue = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    dest_g[index] = (int)(min(max(int(factor * green), 0), 255));
    dest_b[index] = (int)(min(max(int(factor * blue), 0), 255));
    }
  if (flag_unsharp_mask == 1) {
    img_r[index] = dest_r[index];
    img_g[index] = dest_g[index];
    img_b[index] = dest_b[index];
    float filter[filterHeight][filterWidth] =
    {
      1,  4,  6,  4,  1,
      4, 16, 24, 16,  4,
      6, 24, 36, 24,  6,
      4, 16, 24, 16,  4,
      1,  4,  6,  4,  1,
    };
    factor = 1.0 / 256.0;
  
    red = 0.0;
    green = 0.0;
    blue = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(img_r[index] * 1.5 - (red * factor * 0.5)), 0), 255));
    dest_g[index] = (int)(min(max(int(img_g[index] * 1.5 - (green * factor * 0.5)), 0), 255));
    dest_b[index] = (int)(min(max(int(img_b[index] * 1.5 - (blue * factor * 0.5)), 0), 255));
    }
  }
}
""")

Sharp_Color = mod.get_function("Sharp_Color")



mod = SourceModule("""
__global__ void NLM2_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 3;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 3;
    const float limymin = NLM_BLOCK_RADIUS + 3;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])
                + (img_g[index2] - img_g[index1]) * (img_g[index2] - img_g[index1])
                + (img_b[index2] - img_b[index1]) * (img_b[index2] - img_b[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0.0, 0.0, 0.0};

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float3 clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index3];
                clrIJ.y = img_g[index3];
                clrIJ.z = img_b[index3];
                
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float3 clr00 = {0.0, 0.0, 0.0};
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4] / 256.0;
        clr00.y = img_g[index4] / 256.0;
        clr00.z = img_b[index4] / 256.0;
        
        clr.x = clr.x + (clr00.x - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z - clr.z) * lerpQ;
        
       
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
""")

NLM2_Colour_GPU = mod.get_function("NLM2_Colour")


mod = SourceModule("""
__global__ void NLM2_Mono(unsigned char *dest_r, unsigned char *img_r,
int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 3;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 3;
    const float limymin = NLM_BLOCK_RADIUS + 3;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ = img_r[index3];
                
                clr += clrIJ * weightIJ;
 
                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float clr00 = 0.0;
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4] / 256.0;
        
        clr = clr + (clr00 - clr) * lerpQ;
       
        dest_r[index5] = (int)(clr * 256.0);
    }
}
""")

NLM2_Mono_GPU = mod.get_function("NLM2_Mono")



mod = SourceModule("""
__global__ void KNN_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define KNN_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = KNN_BLOCK_RADIUS + 3;
    const float limxmax = imageW - KNN_BLOCK_RADIUS - 3;
    const float limymin = KNN_BLOCK_RADIUS + 3;
    const float limymax = imageH - KNN_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};
        float3 clr00 = {0, 0, 0};
        float3 clrIJ = {0, 0, 0};
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4];
        clr00.y = img_g[index4];
        clr00.z = img_b[index4];
    
        for(float i = -KNN_BLOCK_RADIUS; i <= KNN_BLOCK_RADIUS; i++)
            for(float j = -KNN_BLOCK_RADIUS; j <= KNN_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index2];
                clrIJ.y = img_g[index2];
                clrIJ.z = img_b[index2];
                float distanceIJ = ((clrIJ.x - clr00.x) * (clrIJ.x - clr00.x)
                + (clrIJ.y - clr00.y) * (clrIJ.y - clr00.y)
                + (clrIJ.z - clr00.z) * (clrIJ.z - clr00.z)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr.x = clr.x + (clr00.x / 256.0 - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y / 256.0 - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z / 256.0 - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
""")

KNN_Colour_GPU = mod.get_function("KNN_Colour")


mod = SourceModule("""
__global__ void KNN_Mono(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define KNN_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = KNN_BLOCK_RADIUS + 3;
    const float limxmax = imageW - KNN_BLOCK_RADIUS - 3;
    const float limymin = KNN_BLOCK_RADIUS + 3;
    const float limymax = imageH - KNN_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;
        float clr00 = 0.0;
        float clrIJ = 0.0;
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4];

        for(float i = -KNN_BLOCK_RADIUS; i <= KNN_BLOCK_RADIUS; i++)
            for(float j = -KNN_BLOCK_RADIUS; j <= KNN_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ = img_r[index2];
                float distanceIJ = ((clrIJ - clr00) * (clrIJ - clr00)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr += clrIJ * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr = clr + (clr00 / 256.0 - clr) * lerpQ;
        
        dest_r[index5] = (int)(clr * 256.0);
    }
}
""")

KNN_Mono_GPU = mod.get_function("KNN_Mono")



def quitter() :
    global camera,flag_autorise_acquisition,thread1
    flag_autorise_acquisition = False
    time.sleep(1)
    fenetre_principale.quit()
    
        

def refresh() :
    global val_nb_capt_video,flag_cap_video,image_brut_Base, image_brut_read,camera,video,traitement, img_cam,cadre_image,rawCapture,image_affichee,image_brut_CV,flag_image_disponible,\
           thread_1,flag_acquisition_en_cours,flag_autorise_acquisition,flag_premier_demarrage,flag_BIN2,image_traitee,val_SAT
    if flag_premier_demarrage == True :
        flag_premier_demarrage = False
        image_brut_read = cv2.imread(Image_Test,cv2.IMREAD_COLOR)
        image_brut_read = cv2.cvtColor(image_brut_read,cv2.COLOR_BGR2RGB)
        image_brut_Base = image_brut_read
        flag_image_disponible = True
    image_brut_Base = image_brut_read
    application_filtrage()
    if flag_filtrage_ON == True and flag_noir_blanc == 0 and flag_colour_camera == True and flag_SAT == True :
        img_base=PIL.Image.fromarray(image_traitee)
        converter = ImageEnhance.Color(img_base)
        img_sat = converter.enhance(val_SAT)
        img_sat_np = np.array(img_sat)
        img_sat_np = cv2.GaussianBlur(img_sat_np,(17,17),cv2.BORDER_DEFAULT,)
        image_traitee = cv2.addWeighted(image_traitee,0.6,img_sat_np,0.4,0)
        kern_sharp_soft1 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.int8) # application filtre logiciel sharpen 1
        image_traitee=cv2.filter2D(image_traitee,-1,kern_sharp_soft1)
    img_cam=PIL.Image.fromarray(image_traitee)
    #image_traitee = np.array(img_cam)
    if flag_cap_pic == True:
        pic_capture()
    if flag_cap_video == True :
        video_capture()
    #print("OK")
    if res_cam_x > cam_displ_x and flag_full_res == 0 :
        cadre_image.im=img_cam.resize((cam_displ_x,cam_displ_y), PIL.Image.NEAREST)
    else :
        cadre_image.im = img_cam
    cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
    cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)
    fenetre_principale.after(100, refresh)


def application_filtrage() :
    global image_brut_Base,image_traitee1,image_traitee2,image_traitee3,compteur_FS,Im1OK,Im2OK,Im3OK,image_brut_CV,img_brut_tmp,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max, \
           flag_cap_pic,flag_traitement,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,val_heq2,val_SGR,val_NGB,val_AGR,flag_filtre_work
    if flag_filtrage_ON == True :
#        start_time = time.time()        
        nb_ThreadsX = 32
        nb_ThreadsY = 16
        flag_filtre_work = True
        if flag_noir_blanc == 1 or flag_colour_camera == False :
            # traitement image monochrome
            
            height,width,layers = image_brut_Base.shape
            image_brut_CV,img,imb = cv2.split(image_brut_Base)
            nb_pixels = height * width

            # Image Negatif
            if ImageNeg == 1 :
                image_brut_CV = cv2.bitwise_not(image_brut_CV) # Test fonction cv2.bitwise_not

            # bilateral openCV
            if flag_bilateral == 1 :
                image_brut = cv2.UMat(image_brut_CV)
                image_brut=cv2.bilateralFilter(image_brut,5,125,125) # Application filtre bilateral
                image_brut_CV = cv2.UMat.get(image_brut)

            # 2DConv = Blur CUDA
            # Gaussian = Gaussian Blur CUDA
            if flag_2DConv == 1 or flag_gaussian == 1 :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
            
                r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_brut_CV)
                drv.memcpy_htod(img_r_gpu, image_brut_CV)
                res_r = np.empty_like(image_brut_CV)
                Smooth_Mono(r_gpu, img_r_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_2DConv), np.intc(flag_gaussian), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                image_brut_CV = res_r
                r_gpu.free()
                img_r_gpu.free()


            # Histo equalize 2 CUDA
            # Histo stretch CUDA
            # Histo Phi Theta CUDA           
            if flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1 :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_brut_CV)
                drv.memcpy_htod(img_r_gpu, image_brut_CV)
                res_r = np.empty_like(image_brut_CV)
                Histo_Mono(r_gpu, img_r_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \
                   np.intc(flag_histogram_equalize2), np.float32(val_heq2), np.intc(flag_histogram_phitheta), np.float32(val_phi), np.float32(val_theta), \
                   block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                image_brut_CV = res_r
                r_gpu.free()
                img_r_gpu.free()


            # Amplification soft image
            if flag_AmpSoft == 1 :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_brut_CV)
                img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, image_brut_CV)
                res_r = np.empty_like(image_brut_CV)
                Mono_ampsoft_GPU(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.float32(val_ampl), \
                     block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                r_gpu.free()
                img_r_gpu.free()
                image_brut_CV=res_r                


            # Denoise PAILLOU CUDA
            if flag_denoise_Paillou == 1 :
                cell_size = 3
                sqr_cell_size = cell_size * cell_size
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1

                r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_brut_CV)
                img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, image_brut_CV)
                res_r = np.empty_like(image_brut_CV)

                Denoise_Paillou_Mono(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.intc(cell_size), \
                            np.intc(sqr_cell_size),block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(res_r, r_gpu)
                r_gpu.free()
                img_r_gpu.free()
                image_brut_CV=res_r                

 
            # Denoise NLM2 CUDA
            if flag_denoise_soft == 1 :
                nb_ThreadsXs = 8
                nb_ThreadsYs = 8
                nb_blocksX = (width // nb_ThreadsXs) + 1
                nb_blocksY = (height // nb_ThreadsYs) + 1
                param=float(val_denoise)
                Noise = 1.0/(param*param)
                lerpC = 0.8
                r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_brut_CV)
                img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, image_brut_CV)
                res_r = np.empty_like(image_brut_CV)
                NLM2_Mono_GPU(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsXs,nb_ThreadsYs,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                r_gpu.free()
                img_r_gpu.free()
                image_brut_CV=res_r                
                     
                
            # Denoise KNN CUDA
            if flag_denoise_KNN == 1 :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                param=float(val_denoise_KNN)
                Noise = 1.0/(param*param)
                lerpC = 0.4
                r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_brut_CV)
                img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, image_brut_CV)
                res_r = np.empty_like(image_brut_CV)
                KNN_Mono_GPU(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                r_gpu.free()
                img_r_gpu.free()
                image_brut_CV=res_r                
                
               
            # SharpenSoft1 = Sharpen 1 CUDA
            # UnsharpMask CUDA
            if flag_sharpen_soft1 == 1 or flag_unsharp_mask == 1 :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
            
                r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_brut_CV)
                drv.memcpy_htod(img_r_gpu, image_brut_CV)
                res_r = np.empty_like(image_brut_CV)
                Sharp_Mono(r_gpu, img_r_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_sharpen_soft1), np.intc(flag_unsharp_mask), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                image_brut_CV = res_r
                r_gpu.free()
                img_r_gpu.free()
                       

            # Histo equalize 1 openCV
            if flag_histogram_equalize1 ==1 :               # Egalisation histogramme methode 1
                image_brut_CV=cv2.equalizeHist(image_brut_CV)


         
            # Contrast CLAHE openCV
            if flag_contrast_CLAHE ==1 :
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                image_brut_CV = clahe.apply(image_brut_CV)

            # Gradient Removal openCV
            if flag_GR == True :
                seuil = np.percentile(image_brut_CV, val_SGR)
                gradient = image_brut_CV
                th,gradient = cv2.threshold(gradient,seuil,255,cv2.THRESH_TRUNC) 
                niveau_blur = val_NGB*2 + 3
                gradient=cv2.GaussianBlur(gradient,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,)
                attenuation = gradient * ((100.0-val_AGR) / 100.0) 
                gradient = np.uint8(attenuation)
                image_brut_CV = cv2.subtract(image_brut_CV,gradient)
            

            
        if flag_noir_blanc == 0 and flag_colour_camera == True:
            # traitement image couleur

            height,width,layers = image_brut_Base.shape
            nb_pixels = height * width
            image_brut_CV = image_brut_Base


            # Image negative
            if ImageNeg == 1 :
                img_b,img_g,img_r = cv2.split(image_brut_CV)
                img_b = cv2.bitwise_not(img_b)
                img_g = cv2.bitwise_not(img_g)
                img_r = cv2.bitwise_not(img_r)
                image_brut_CV=cv2.merge((img_b,img_g,img_r))
                #image_brut_CV = cv2.applyColorMap(img_g, cv2.COLORMAP_JET)

            # bilateral openCV
            if flag_bilateral == 1 :
                image_brut = cv2.UMat(image_brut_CV)
                image_brut=cv2.bilateralFilter(image_brut,5,125,125) # Application filtre bilateral
                image_brut_CV = cv2.UMat.get(image_brut)


            if flag_2DConv == 1 or flag_gaussian == 1 :
                # 2DConv = Blur CUDA
                # Gaussian = Gaussian Blur CUDA
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
            
                b,g,r = cv2.split(image_brut_CV)
            
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)
            
                Smooth_Color(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_2DConv), np.intc(flag_gaussian), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()

                image_brut_CV=cv2.merge((res_b,res_g,res_r))


            if flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1 :
                # Histo equalize 2 CUDA
                # Histo stretch CUDA
                # Histo Phi Theta CUDA
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                b,g,r = cv2.split(image_brut_CV)
            
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)
            
                Histo_Color(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \
                   np.intc(flag_histogram_equalize2), np.float32(val_heq2), np.intc(flag_histogram_phitheta), np.float32(val_phi), np.float32(val_theta), \
                   block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()
            
                image_brut_CV=cv2.merge((res_b,res_g,res_r))
                #image_brut_CV=cv2.cvtColor(image_brut_CV, cv2.COLOR_BGR2RGB)

            # Amplification soft image
            if flag_AmpSoft == 1 :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1

                b,g,r = cv2.split(image_brut_CV)
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)

                Colour_ampsoft_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.float32(val_ampl), \
                     block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()
                image_brut_CV=cv2.merge((res_b,res_g,res_r))            

            # Denoise PAILLOU CUDA
            if flag_denoise_Paillou == 1 :
                cell_size = 3
                sqr_cell_size = cell_size * cell_size
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1

                b,g,r = cv2.split(image_brut_CV)
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)

                Denoise_Paillou_Colour(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.intc(cell_size), \
                            np.intc(sqr_cell_size),block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()
                image_brut_CV=cv2.merge((res_b,res_g,res_r))


            # Denoise NLM2 CUDA
            if flag_denoise_soft == 1 :
                nb_ThreadsXs = 8
                nb_ThreadsYs = 8
                nb_blocksX = (width // nb_ThreadsXs) + 1
                nb_blocksY = (height // nb_ThreadsYs) + 1
                param=float(val_denoise)
                Noise = 1.0/(param*param)
                lerpC = 0.8
                
                b,g,r = cv2.split(image_brut_CV)
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)

                NLM2_Colour_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsXs,nb_ThreadsYs,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()
                image_brut_CV=cv2.merge((res_b,res_g,res_r))            
            
            
            # Denoise KNN CUDA
            if flag_denoise_KNN == 1 :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                param=float(val_denoise_KNN)
                Noise = 1.0/(param*param)
                lerpC = 0.4
                
                b,g,r = cv2.split(image_brut_CV)
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)

                KNN_Colour_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                
                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()
                image_brut_CV=cv2.merge((res_b,res_g,res_r))            

            # Adjust RGB
            if (val_red != 100 or val_green != 100 or val_blue != 100) :
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                mod_red = val_red / 100.0
                mod_green = val_green / 100.0
                mod_blue = val_blue / 100.0
            
                b,g,r = cv2.split(image_brut_CV)
            
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)
            
                Set_RGB(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                   np.float32(mod_red), np.float32(mod_green), np.float32(mod_blue), \
                                  block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()

                image_brut_CV=cv2.merge((res_b,res_g,res_r))


            if flag_sharpen_soft1 == 1 or flag_unsharp_mask == 1 :
                # SharpenSoft1 = Sharpen 1 CUDA
                # UnsharpMask CUDA
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
            
                b,g,r = cv2.split(image_brut_CV)
            
                b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
                drv.memcpy_htod(b_gpu, b)
                drv.memcpy_htod(img_b_gpu, b)
                res_b = np.empty_like(b)
            
                g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(g_gpu, g)
                img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
                drv.memcpy_htod(img_g_gpu, g)
                res_g = np.empty_like(g)
            
                r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(r_gpu, r)
                img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
                drv.memcpy_htod(img_r_gpu, r)
                res_r = np.empty_like(r)
            
                Sharp_Color(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_sharpen_soft1), np.intc(flag_unsharp_mask), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r, r_gpu)
                drv.memcpy_dtoh(res_g, g_gpu)
                drv.memcpy_dtoh(res_b, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()

                image_brut_CV=cv2.merge((res_b,res_g,res_r))
                  
            # Histo equalize 1 openCV
            if flag_histogram_equalize1 ==1 :               # Egalisation histogramme methode 1
                img_b,img_g,img_r = cv2.split(image_brut_CV)
                img_b=cv2.equalizeHist(img_b)
                img_g=cv2.equalizeHist(img_g)
                img_r=cv2.equalizeHist(img_r)
                image_brut_CV=cv2.merge((img_b,img_g,img_r))

                             
            # Contrast CLAHE openCV
            if flag_contrast_CLAHE ==1 :
                img_b,img_g,img_r = cv2.split(image_brut_CV)
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                img_b=clahe.apply(img_b)
                img_g=clahe.apply(img_g)
                img_r=clahe.apply(img_r)
                image_brut_CV=cv2.merge((img_b,img_g,img_r))

            # Gradient Removal openCV
            if flag_GR == True :
                seuil = np.percentile(image_brut_CV[1], val_SGR)
                img_b,img_g,img_r = cv2.split(image_brut_CV)
                th,img_b = cv2.threshold(img_b,seuil,255,cv2.THRESH_TRUNC) 
                th,img_g = cv2.threshold(img_g,seuil,255,cv2.THRESH_TRUNC) 
                th,img_r = cv2.threshold(img_r,seuil,255,cv2.THRESH_TRUNC) 
                niveau_blur = val_NGB*2 + 3
                img_b=cv2.GaussianBlur(img_b,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT) 
                img_g=cv2.GaussianBlur(img_g,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT) 
                img_r=cv2.GaussianBlur(img_r,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT) 
                att_b = img_b * ((100.0-val_AGR) / 100.0) 
                att_g = img_g * ((100.0-val_AGR) / 100.0) 
                att_r = img_r * ((100.0-val_AGR) / 100.0)
                img_b = np.uint8(att_b)
                img_g = np.uint8(att_g)
                img_r = np.uint8(att_r)
                gradient=cv2.merge((img_b,img_g,img_r))
                image_brut_CV = cv2.subtract(image_brut_CV,gradient)
                #image_brut_CV=cv2.cvtColor(image_brut_CV, cv2.COLOR_BGR2RGB)



        compteur_FS = compteur_FS+1
        if compteur_FS > val_FS :
            compteur_FS = 1
        if compteur_FS == 1 :
            image_traitee1 = image_brut_CV
            Im1OK = True
        if compteur_FS == 2 :
            image_traitee2 = image_brut_CV
            Im2OK = True
        if compteur_FS == 3 :
            image_traitee3 = image_brut_CV
            Im3OK = True
        if val_FS == 1 :
            image_traitee = image_brut_CV
        if val_FS == 2 :
            if Im2OK == True :
                image_traitee = cv2.add(image_traitee1,image_traitee2)
            else :
                image_traitee = image_brut_CV
        if val_FS == 3 :
            if Im3OK == True :
                image_traitee = cv2.add(image_traitee1,image_traitee2)
                image_traitee = cv2.add(image_traitee,image_traitee3)
            else :
                image_traitee = image_brut_CV
        flag_filtre_work = False
    else :
        image_traitee = image_brut_Base
              
                        
def pic_capture() :
    global start,nb_pic_cap,nb_acq_pic,labelInfo1,flag_cap_pic,nb_cap_pic,image_path,image_traitee
    if nb_cap_pic <= val_nb_captures :
        if flag_HQ == 0 :
            if flag_filter_wheel == True :
                nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '_F' + "%01d" % fw_position + '.jpg' # JPEG File format
            else :
                nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '.jpg'  # JPEG File format
        else :
            if flag_filter_wheel == True:
                nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '_F' + "%01d" % fw_position + '.tif' # TIF file format loseless format
            else :
                nom_fichier = start.strftime(
                    'PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '.tif'  # TIF file format loseless format
        if image_traitee.ndim == 3 :
            if flag_HQ == 0 :
                cv2.imwrite(os.path.join(image_path,nom_fichier), cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format
            else :
                cv2.imwrite(os.path.join(image_path, nom_fichier), cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB)) # TIFF file format
        else :
            if flag_HQ == 0 :
                cv2.imwrite(os.path.join(image_path,nom_fichier), image_traitee, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format
            else :
                cv2.imwrite(os.path.join(image_path, nom_fichier), image_traitee) # TIFF file format
        labelInfo1 = Label (cadre, text = "capture n° "+ nom_fichier)
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        nb_cap_pic += 1
    else :
        flag_cap_pic = False
        labelInfo1 = Label (cadre, text = "                                                                                                        ") 
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        labelInfo1 = Label (cadre, text = " Capture pictures terminee")
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
    
def start_pic_capture() :
    global nb_acq_pic,flag_cap_pic,nb_cap_pic,start
    flag_cap_pic = True
    nb_cap_pic =1
    start = datetime.now()
 
def stop_pic_capture() :
    global nb_cap_pic
    nb_cap_pic = val_nb_captures +1

def video_capture() :
    global image_traitee,start_video,nb_cap_video,nb_acq_video,labelInfo1,flag_cap_video,video_path,val_nb_capt_video,videoOut,echelle11
    if nb_cap_video == 1 :
        if flag_HQ == 0:
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # video compressée
        else :
            fourcc = 0 # video RAW
            #fourcc = cv2.VideoWriter_fourcc(*'DIB ') # video non compressee
        if flag_filter_wheel == True:
            nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.avi'
        else :
            nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.avi'
        if image_traitee.ndim == 3 :
            height,width,layers = image_traitee.shape
            if flag_HQ == 0 :
                videoOut = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 30, (width, height), isColor = True) # video compressée
            else :
                videoOut = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 30, (width, height), isColor=True) # video RAW
        else :
            height,width = image_traitee.shape
            if flag_HQ == 0 :
                videoOut = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 30, (width, height), isColor = False) # video compressée
            else :
                videoOut = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 30, (width, height), isColor=False) # video RAW
        labelInfo1 = Label (cadre, text = "                                                                                                        ") 
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        labelInfo1 = Label (cadre, text = " Acquisition vidéo en cours")
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
    if nb_cap_video <= val_nb_capt_video -2 :
        if image_traitee.ndim == 3 :
            videoOut.write(cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB))
        else :
            videoOut.write(image_traitee)
        if nb_cap_video % 10 == 0 :
            labelInfo1 = Label (cadre, text = " frame : " + str (nb_cap_video) + "                                       ")
            labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        nb_cap_video += 1
    else :
        videoOut.release()
        flag_cap_video = False
        labelInfo1 = Label (cadre, text = " Acquisition vidéo terminee     ")
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        
def start_video_capture() :
    global nb_cap_video,flag_cap_video,start_video,val_nb_capt_video
    flag_cap_video = True
    nb_cap_video =1
    if val_nb_capt_video == 0 :
        val_nb_capt_video = 10000
    start_video = datetime.now()
    
 
def stop_video_capture() :
    global nb_cap_video,val_nb_capt_video
    nb_cap_video = val_nb_capt_video +1
    

# Fonctions récupération des paramètres grace aux scalebars
def mode_acq_rapide() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_acq_rapide = True
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    time.sleep(0.001)
 
def mode_acq_medium() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_acq_rapide = False
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    time.sleep(0.001)
 
def mode_acq_lente() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_acq_rapide = False
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    time.sleep(0.001)

def valeur_exposition (event=None) :
    global timeout_val,flag_acq_rapide,camera,val_exposition,echelle1,val_resolution,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    val_exposition = echelle1.get()
    time.sleep(0.001)


def valeur_gain (event=None) :
    global camera,val_gain,echelle2,flag_sub_dark,dispo_dark,labelInfoDark,flag_stop_acquisition
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    val_gain = echelle2.get()
    time.sleep(0.001)
   

def choix_BIN1(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark,mode_BIN
    time.sleep(0.001)

def choix_BIN2(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark,mode_BIN
    time.sleep(0.001)
    
def choix_BIN3(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark,mode_BIN
    time.sleep(0.001)

def choix_resolution_camera(event=None) :
    global val_FS,camera,traitement,val_resolution,res_cam_x,res_cam_y, img_cam,rawCapture,res_x_max,res_y_max,echelle3,flag_image_disponible,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    time.sleep(0.001)

def choix_valeur_denoise(event=None) :
    global val_denoise
    val_denoise=echelle4.get()
    if val_denoise == 0 :
        val_denoise += 1       

def commande_img_Neg() :
    global ImageNeg
    if choix_img_Neg.get() == 0 :
        ImageNeg = 0
    else :
        ImageNeg = 1
        
def commande_2DConvol() :
    global flag_2DConv
    if choix_2DConv.get() == 0 :
        flag_2DConv = 0
    else :
        flag_2DConv = 1
        
def commande_gaussian() :
    global flag_gaussian
    if choix_gaussian.get() == 0 :
        flag_gaussian = 0
    else :
        flag_gaussian = 1
        
def commande_bilateral() :
    global flag_bilateral
    if choix_bilateral.get() == 0 :
        flag_bilateral = 0
    else :
        flag_bilateral = 1

def commande_TIP() :
    global flag_TIP
    if choix_TIP.get() == 0 :
        flag_TIP = False
    else :
        flag_TIP = True

def commande_SAT() :
    global flag_SAT
    if choix_SAT.get() == 0 :
        flag_SAT = False
    else :
        flag_SAT = True

def commande_cross() :
    global flag_cross
    if choix_cross.get() == 0 :
        flag_cross = False
    else :
        flag_cross = True
        
def commande_mode_full_res() :
    global flag_full_res
    if choix_mode_full_res.get() == 0 :
        flag_full_res = 0
    else :
        flag_full_res = 1

def choix_valeur_CLAHE(event=None) :
    global val_contrast_CLAHE,echelle9
    val_contrast_CLAHE=echelle9.get()

def commande_sharpen_soft1() :
    global flag_sharpen_soft1
    if choix_sharpen_soft1.get() == 0 :
        flag_sharpen_soft1 = 0
    else :
        flag_sharpen_soft1 = 1
        
def commande_unsharp_mask() :
    global flag_unsharp_mask
    if choix_unsharp_mask.get() == 0 :
        flag_unsharp_mask = 0
    else :
        flag_unsharp_mask = 1

def commande_denoise_soft() :
    global flag_denoise_soft
    if choix_denoise_soft.get() == 0 :
        flag_denoise_soft = 0
    else :
        flag_denoise_soft = 1

def commande_denoise_Paillou() :
    global flag_denoise_Paillou
    if choix_denoise_Paillou.get() == 0 :
        flag_denoise_Paillou = 0
    else :
        flag_denoise_Paillou = 1

def choix_KNN() :
    global flag_denoise_KNN
    if choix_denoise_KNN.get() == 0 :
        flag_denoise_KNN = 0
    else :
        flag_denoise_KNN = 1

def commande_DEF() :
    global flag_DEF
    if choix_DEF.get() == 0 :
        flag_DEF = 0
    else :
        flag_DEF = 1

def commande_GR() :
    global flag_GR
    if choix_GR.get() == 0 :
        flag_GR = 0
    else :
        flag_GR = 1

def commande_histogram_equalize1() :
    global flag_histogram_equalize1
    if choix_histogram_equalize1.get() == 0 :
        flag_histogram_equalize1 = 0
    else :
        flag_histogram_equalize1 = 1

def commande_histogram_equalize2() :
    global flag_histogram_equalize2
    if choix_histogram_equalize2.get() == 0 :
        flag_histogram_equalize2 = 0
    else :
        flag_histogram_equalize2 = 1

def choix_histo_min(event=None) :
    global camera,val_histo_min,echelle5
    val_histo_min=echelle5.get()
    
def choix_phi(event=None) :
    global val_phi,echelle12
    val_phi=echelle12.get()

def choix_theta(event=None) :
    global val_theta,echelle13
    val_theta=echelle13.get()
    
def choix_histo_max(event=None) :
    global camera,val_histo_max,echelle6
    val_histo_max=echelle6.get()
      
def commande_histogram_stretch() :
    global flag_histogram_stretch
    if choix_histogram_stretch.get() == 0 :
        flag_histogram_stretch = 0
    else :
        flag_histogram_stretch = 1
    
def commande_histogram_phitheta() :
    global flag_histogram_phitheta
    if choix_histogram_phitheta.get() == 0 :
        flag_histogram_phitheta = 0
    else :
        flag_histogram_phitheta = 1
      
def commande_contrast_CLAHE() :
    global flag_contrast_CLAHE
    if choix_contrast_CLAHE.get() == 0 :
        flag_contrast_CLAHE= 0
    else :
        flag_contrast_CLAHE = 1

def commande_filtrage_ON() :
    global flag_filtrage_ON
    if choix_filtrage_ON.get() == 0 :
        flag_filtrage_ON= 0
    else :
        flag_filtrage_ON = 1

def commande_HQ_capt() :
    global flag_HQ
    if choix_HQ_capt.get() == 0 :
        flag_HQ = 0
    else :
        flag_HQ = 1

def commande_hard_bin() :
    global camera
    time.sleep(0.001)

def commande_noir_blanc() :
    global val_FS,flag_noir_blanc,flag_sub_dark,dispo_dark,labelInfoDark,format_capture,flag_stop_acquisition, flag_sub_dark, dispo_dark, labelInfoDark
    reset_FS()
    flag_stop_acquisition=True
    flag_sub_dark = False
    time.sleep(0.001)

def choix_nb_captures(event=None) :
    global val_nb_captures
    val_nb_captures=echelle8.get()

def choix_nb_video(event=None) :
    global val_nb_capt_video
    val_nb_capt_video=echelle11.get()

def commande_sub_dark() :
    global flag_sub_dark,dispo_dark
    if choix_sub_dark.get() == 0 :
        flag_sub_dark = False
    else :
        flag_sub_dark = True

def start_cap_dark() :
    global val_FS,camera,flag_sub_dark,dispo_dark,labelInfoDark,flag_stop_acquisition,flag_acquisition_en_cours,labelInfo1,val_nb_darks,text_info1,xLI1,yLI1,Master_Dark
    time.sleep(0.001)
    
def choix_nb_darks(event=None) :
    global val_nb_darks, echelle10
    val_nb_darks=echelle10.get()
    
def choix_w_red(event=None) :
    global val_red, echelle14
    val_red=echelle14.get()

def choix_w_green(event=None) :
    global val_green, echelle65
    val_green=echelle65.get()
    
def choix_w_blue(event=None) :
    global val_blue, echelle15
    val_blue=echelle15.get()

def choix_heq2(event=None) :
    global val_heq2, echelle16
    val_heq2=echelle16.get()

def choix_val_KNN(event=None) :
    global val_denoise_KNN, echelle30
    val_denoise_KNN=echelle30.get()

def choix_SGR(event=None) :
    global val_SGR, echelle60
    val_SGR=echelle60.get()

def choix_AGR(event=None) :
    global val_AGR, echelle61
    val_AGR=echelle61.get()

def choix_NGB(event=None) :
    global val_NGB, echelle62
    val_NGB=echelle62.get()

def choix_USB(event=None) :
    global val_USB, echelle50,camera
    val_USB=echelle50.get()

def choix_SAT(event=None) :
    global val_SAT, echelle55
    val_SAT=echelle55.get()

def choix_position_EFW0(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 0
        print('FW positionselect :',fw_position)

def choix_position_EFW1(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 1
        print('FW positionselect :',fw_position)

def choix_position_EFW2(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 2
        print('FW positionselect :',fw_position)

def choix_position_EFW3(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 3
        print('FW positionselect :',fw_position)

def choix_position_EFW4(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 4
        print('FW positionselect :',fw_position)

def choix_FS(event=None) :
    global val_FS
    val_FS=echelle20.get()

def reset_FS(event=None) :
    global val_FS,compteur_FS,Im1OK,Im2OK,Im3OK
    val_FS = 1
    compteur_FS = 0
    Im1OK = False
    Im2OK = False
    Im3OK = False
    echelle20.set(val_FS)    

def choix_val_SAT(event=None) :
    global val_SAT, echelle70
    val_SAT=echelle70.get()

def commande_AmpSoft() :
    global flag_AmpSoft
    if choix_AmpSoft.get() == 0 :
        flag_AmpSoft = 0
    else :
        flag_AmpSoft = 1

def choix_amplif(event=None) :
    global val_ampl, echelle80
    val_ampl = echelle80.get()

def commande_FW() :
    time.sleep(0.01)

# définition fenetre principale
fenetre_principale = Tk ()
#w, h = fenetre_principale.winfo_screenwidth(), fenetre_principale.winfo_screenheight()-20
w,h=1900,1060
fenetre_principale.geometry("%dx%d+0+0" % (w, h))
fenetre_principale.protocol("WM_DELETE_WINDOW", quitter)
default_font = nametofont("TkDefaultFont")
default_font.configure(size=7)
fenetre_principale.title(titre)

# Création cadre général
cadre = Frame (fenetre_principale, width = 1800 , heigh = 950)
cadre.pack ()

mode_acq = IntVar()
mode_acq.set(1) # Initialisation du mode d'acquisition a Rapide

choix_bin = IntVar ()
choix_bin.set(1) # Initialisation BIN 1 sur choix 1, 2 ou 3

choix_TIP = IntVar ()
choix_TIP.set(0) # Initialisation TIP Inactif

choix_cross = IntVar ()
choix_cross.set(0) # Initialisation Croix centre image

choix_SAT = IntVar () # réglage saturation couleurs
choix_SAT.set(0)

choix_img_Neg = IntVar ()
choix_img_Neg.set(0) # Initialisation image en négatif inactif

choix_2DConv = IntVar()
choix_2DConv.set(0) # intialisation filtre 2D convolution inactif

choix_gaussian = IntVar()
choix_gaussian.set(0) # initialisation filtre gaussien inactif

choix_bilateral = IntVar()
choix_bilateral.set(0) # Initialisation filtre Median inactif

choix_mode_full_res = IntVar()
choix_mode_full_res.set(0) # Initialisation mode full resolution inactif

choix_sharpen_soft1 = IntVar()
choix_sharpen_soft1.set(0) # initialisation mode sharpen software 1 inactif

choix_unsharp_mask = IntVar()
choix_unsharp_mask.set(0) # initialisation mode unsharp mask inactif

choix_denoise_soft = IntVar()
choix_denoise_soft.set(0) # initialisation mode denoise software inactif

choix_histogram_equalize1 = IntVar()
choix_histogram_equalize1.set(0) # initialisation mode histogram equalize 1 inactif

choix_histogram_equalize2 = IntVar()
choix_histogram_equalize2.set(0) # initialisation mode histogram equalize 2 inactif

choix_histogram_stretch = IntVar()
choix_histogram_stretch.set(0) # initialisation mode histogram stretch inactif

choix_histogram_phitheta = IntVar()
choix_histogram_phitheta.set(0) # initialisation mode histogram Phi Theta inactif

choix_contrast_CLAHE = IntVar()
choix_contrast_CLAHE.set(0) # initialisation mode contraste CLAHE inactif

choix_noir_blanc = IntVar()
choix_noir_blanc.set(0) # initialisation mode noir et blanc inactif

choix_hard_bin = IntVar()
choix_hard_bin.set(0) # initialisation mode hardware bin disable

choix_HQ_capt = IntVar()
choix_HQ_capt.set(0) # initialisation mode capture Low Quality

choix_sub_dark = IntVar()
choix_sub_dark.set(0) # Initialisation sub dark inactif

choix_filtrage_ON = IntVar()
choix_filtrage_ON.set(1) # Initialisation Filtrage ON actif

choix_denoise_KNN = IntVar()
choix_denoise_KNN.set(0) # Initialisation Filtrage Denoise KNN

choix_denoise_Paillou = IntVar()
choix_denoise_Paillou.set(0) # Initialisation Filtrage Denoise Paillou

choix_GR = IntVar()
choix_GR.set(0) # Initialisation Filtre Gradient Removal

presence_FW = IntVar()
presence_FW.set(0) # Initialisation absence FW

choix_AmpSoft = IntVar()
choix_AmpSoft.set(0) # Initialisation amplification software OFF

# initialisation des boites scrolbar, buttonradio et checkbutton

xCBEPF=1290 # Sélection filtre Edge preserving
yCBEPF=365

xCBDEF=1470 # Sélection filtre Detail Enhancing
yCBDEF=415

xBRB=1440 # Sélection mode BIN 1, 2 ou 3 bouton radio
yBRB=45

xCBFNB=1350 # Check box force N&B
yCBFNB=65

xCBSA=1350 # Sélection mode Full Resolution
yCBSA=45

xS3=1550 # Choix résolution
yS3=105

xBRMA=1470 # Mode acquisition bouton radio
yBRMA=195

xS1=1400 # Durée exposition en ms
yS1=230

xS2=1400 # Selection du Gain
yS2=152

xCBFV=1290 # Sélection Flip V
yCBFV=280

xCBFH=1350 # Sélection Flip H
yCBFH=280

xCBIN=1290 # Sélection Image Negative
yCBIN=315

#xCBF = 1200 # Selection Filtrage ON
#yCBF = 320

xCB2DC = 1450 # Selection 2D convolution filter
yCB2DC=315

xCBGA= 1200 # Sélection filter Gaussian
yCBGA=315

xCBBL=1200 # Selection filtre bilateral
yCBBL=315

xCBSS1=1290 # Selection filtre sharpen software 1
yCBSS1=415

xCBDS=1630 # Selection filtre Denoise software
yCBDS=315

xS4=1710 # Sélection parametrage Image denoise soft
yS4=315

xCBHPT = 1310 # selection histogramme Phi Theta
yCBHPT = 640

xS12=1460 # Sélection parametrages histo phi theta
yS12=640

xCBHS = 1310 # selection histogramme stretch
yCBHS = 590

xCBCC = 1310 # selection histogramme Contrat CLAHE
yCBCC = 690

xS5=1460 # Sélection parametrages histo mini et maxi
yS5=590

xS8=1430 # Sélection parametrages nb acquisition pictures
yS8=800

xS9=1460 # Sélection parametrages contrast CLAHE
yS9=690

xS10=1430 # Sélection nombre de darks
yS10=740

xS11=1430 # Sélection nombre de video
yS11=870

xS13=1460 # Réglage balance rouge et bleu
yS13=280

xdark = 1600 # Sélection sub dark
ydark = 750

xLI1 = 1350 # label info 1
yLI1 = 940

xCBHE12=1260 # Selection filtre histo equalize 1 et 2
yCBHE12=540

xS14=1360 # Sélection parametrages histo eq 2 paramètre
yS14=540

xCBFW = 1400 # CheckButton présence FW ... ou pas
yCBFW = 20

xCBHB = 1680 # CheckButton Hardware Bin
yCBHB = 45

xCBHQC = 1680 # CheckButton HQ Capture
yCBHQC = 65

xUSB = 15 # Choix bandwidth USB
yUSB = 200

xTIP = 15 # Text in Picture
yTIP = 20

xSAT = 1
ySAT = 230

xGR = 1315 # Filtre Grdient removal
yGR = 470

# Choix position EFW

CBHE1 = Checkbutton(cadre,text="Histo Eq 1", variable=choix_histogram_equalize1,command=commande_histogram_equalize1,onvalue = 1, offvalue = 0)
CBHE1.place(anchor="w",x=xCBHE12+70, y=yCBHE12+15)

CBTIP = Checkbutton(cadre,text="TIP", variable=choix_TIP,command=commande_TIP,onvalue = 1, offvalue = 0)
CBTIP.place(anchor="w",x=xTIP-10, y=yTIP)

CBCR = Checkbutton(cadre,text="Cr", variable=choix_cross,command=commande_cross,onvalue = 1, offvalue = 0)
CBCR.place(anchor="w",x=xTIP-10, y=yTIP+30)

CBSAT = Checkbutton(cadre,text="SAT", variable=choix_SAT,command=commande_SAT,onvalue = 1, offvalue = 0)
CBSAT.place(anchor="w",x=xSAT, y=ySAT)

# Choix parametre saturation
echelle70 = Scale (cadre, from_ = 0, to = 100, command= choix_val_SAT, orient=VERTICAL, length = 150, width = 10, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle70.set(val_SAT)
echelle70.place(anchor="c", x=xSAT+10, y=ySAT+100)

CBHE2 = Checkbutton(cadre,text="Histo Eq 2", variable=choix_histogram_equalize2,command=commande_histogram_equalize2,onvalue = 1, offvalue = 0)
CBHE2.place(anchor="w",x=xCBHE12+170, y=yCBHE12+15)

# Choix HQ Capture
CBHQC = Checkbutton(cadre,text="HQ Capture", variable=choix_HQ_capt,command=commande_HQ_capt,onvalue = 1, offvalue = 0)
CBHQC.place(anchor="w",x=xCBHQC, y=yCBHQC)

# Choix filtre Gradient Removal
CBGR = Checkbutton(cadre,text="Grad Rem", variable=choix_GR,command=commande_GR,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=xGR, y=yGR-5)

# Choix filtre Denoise KNN
CBEPF = Checkbutton(cadre,text="Dn KNN", variable=choix_denoise_KNN,command=choix_KNN,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=xCBDS-20, y=yCBEPF-5+30)

# Choix filtre Denoise Paillou
CBEPF = Checkbutton(cadre,text="Dn Paillou", variable=choix_denoise_Paillou,command=commande_denoise_Paillou,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=xCBDS-100, y=yCBEPF-5+30)

# Choix Hardware Bin
CBHB = Checkbutton(cadre,text="Hard BIN", variable=choix_hard_bin,command=commande_hard_bin,onvalue = 1, offvalue = 0)
CBHB.place(anchor="w",x=xCBHB, y=yCBHB)

# Choix forcage N&B
CBFNB = Checkbutton(cadre,text="Force N&B", variable=choix_noir_blanc,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNB.place(anchor="w",x=xCBFNB, y=yCBFNB)

#Choix histogramme Sigmoide
CBHPT = Checkbutton(cadre,text="Histo Sigmoide", variable=choix_histogram_phitheta,command=commande_histogram_phitheta,onvalue = 1, offvalue = 0)
CBHPT.place(anchor="w",x=xCBHPT+20, y=yCBHPT+10)

# Choix histogramme stretch
CBHS = Checkbutton(cadre,text="Histo Stretch", variable=choix_histogram_stretch,command=commande_histogram_stretch,onvalue = 1, offvalue = 0)
CBHS.place(anchor="w",x=xCBHS+20, y=yCBHS+10)

# Choix contrast CLAHE
CBCC = Checkbutton(cadre,text="Contrast CLAHE", variable=choix_contrast_CLAHE,command=commande_contrast_CLAHE,onvalue = 1, offvalue = 0)
CBCC.place(anchor="w",x=xCBCC+20, y=yCBCC+10)

# Choix filtre 2D convolution
CB2DC = Checkbutton(cadre,text="2D convol", variable=choix_2DConv,command=commande_2DConvol,onvalue = 1, offvalue = 0)
CB2DC.place(anchor="w",x=xCB2DC+80, y=yCB2DC+30)

# Choix filtrage ON
CBF = Checkbutton(cadre,text="Filtrage ON", variable=choix_filtrage_ON,command=commande_filtrage_ON,onvalue = 1, offvalue = 0)
CBF.place(anchor="w",x=xCBFNB+90, y=yCBFNB)

# Choix filtre gaussien
CBGA = Checkbutton(cadre,text="Gaussian", variable=choix_gaussian,command=commande_gaussian,onvalue = 1, offvalue = 0)
CBGA.place(anchor="w",x=xCBGA+185, y=yCBGA+30)

# Choix filtre Bilateral
CBBL = Checkbutton(cadre,text="Bilateral", variable=choix_bilateral,command=commande_bilateral,onvalue = 1, offvalue = 0)
CBBL.place(anchor="w",x=xCBBL+260, y=yCBBL+30)

# Choix du mode amlification soft
CBAS = Checkbutton(cadre,text="Amplif Soft", variable=choix_AmpSoft,command=commande_AmpSoft,onvalue = 1, offvalue = 0)
CBAS.place(anchor="w",x=xCBFV+25, y=yCBFV-10-5)

echelle80 = Scale (cadre, from_ = 0, to = 4.0, command= choix_amplif, orient=HORIZONTAL, length = 300, width = 10, resolution = 0.02, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle80.set(val_ampl)
echelle80.place(anchor="w", x=xS13-30,y=yS13-10-5)

# Choix du mode image en négatif
CBIN = Checkbutton(cadre,text="Img Neg", variable=choix_img_Neg,command=commande_img_Neg,onvalue = 1, offvalue = 0)
CBIN.place(anchor="w",x=xCBIN+25, y=yCBIN+30)

CBMFR = Checkbutton(cadre,text="Full Res", variable=choix_mode_full_res,command=commande_mode_full_res,onvalue = 1, offvalue = 0)
CBMFR.place(anchor="w",x=xCBSA, y=yCBSA)

CBSS1 = Checkbutton(cadre,text="Sharpen", variable=choix_sharpen_soft1,command=commande_sharpen_soft1,onvalue = 1, offvalue = 0)
CBSS1.place(anchor="w",x=xCBSS1+25, y=yCBEPF-5+30)

CBUSM = Checkbutton(cadre,text="Unsharp Mask", variable=choix_unsharp_mask,command=commande_unsharp_mask,onvalue = 1, offvalue = 0)
CBUSM.place(anchor="w",x=xCBSS1+95, y=yCBEPF-5+30)

CBDS = Checkbutton(cadre,text="Dn NLM2", variable=choix_denoise_soft,command=commande_denoise_soft,onvalue = 1, offvalue = 0)
CBDS.place(anchor="w",x=xCBDS-20, y=yCBDS+30)

# Choix du mode d'aquisition
labelMode_Acq = Label (cadre, text = "Mode acquisition")
labelMode_Acq.place (anchor="w",x=xBRMA, y=yBRMA-10)
RBMA1 = Radiobutton(cadre,text="Fast", variable=mode_acq,command=mode_acq_rapide,value=1)
RBMA1.place(anchor="w",x=xBRMA+120, y=yBRMA-10)
RBMA1 = Radiobutton(cadre,text="Medium", variable=mode_acq,command=mode_acq_medium,value=2)
RBMA1.place(anchor="w",x=xBRMA+170, y=yBRMA-10)
RBMA2 = Radiobutton(cadre,text="Slow", variable=mode_acq,command=mode_acq_lente,value=3)
RBMA2.place(anchor="w",x=xBRMA+250, y=yBRMA-10)

# Choix du mode BINNING - 1, 2 ou 3
labelBIN = Label (cadre, text = "BINNING : ")
labelBIN.place (anchor="w",x=xBRB, y=yBRB)
RBB1 = Radiobutton(cadre,text="BIN1", variable=choix_bin,command=choix_BIN1,value=1)
RBB1.place(anchor="w",x=xBRB+70, y=yBRB)
RBB2 = Radiobutton(cadre,text="BIN2", variable=choix_bin,command=choix_BIN2,value=2)
RBB2.place(anchor="w",x=xBRB+120, y=yBRB)
RBB2 = Radiobutton(cadre,text="BIN3", variable=choix_bin,command=choix_BIN3,value=3)
RBB2.place(anchor="w",x=xBRB+180, y=yBRB)

labelParam1 = Label (cadre, text = "Exposition ms")
labelParam1.place(anchor="e", x=xS1,y=yS1-10)
echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 400, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
echelle1.set(val_exposition)
echelle1.place(anchor="w", x=xS1,y=yS1-10)

labelParam2 = Label (cadre, text = "Gain")
labelParam2.place(anchor="e", x=xS2-50,y=yS2-10)
echelle2 = Scale (cadre, from_ = 0, to = 510 , command= valeur_gain, orient=HORIZONTAL, length = 450, width = 10, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle2.set(val_gain)
echelle2.place(anchor="w", x=xS2-50,y=yS2-10)

labelParam3 = Label (cadre, text = "Resolution")
labelParam3.place(anchor="e", x=xS3,y=yS3-10)
echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 250, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle3.set (val_resolution)
echelle3.place(anchor="w", x=xS3,y=yS3-10)

echelle4 = Scale (cadre, from_ = 0.1, to = 2, command= choix_valeur_denoise, orient=HORIZONTAL, length = 120, width = 10, resolution = 0.05, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle4.set(val_denoise)
echelle4.place(anchor="w", x=xS4-30,y=yS4+30)

labelParam5 = Label (cadre, text = "Min") # choix valeur histogramme strech minimum
labelParam5.place(anchor="w", x=xS5,y=yS5+10)
echelle5 = Scale (cadre, from_ = 0, to = 150, command= choix_histo_min, orient=HORIZONTAL, length = 130, width = 10, resolution = 2, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle5.set(val_histo_min)
echelle5.place(anchor="w", x=xS5+30,y=yS5+10)

labelParam6 = Label (cadre, text = "Max") # choix valeur histogramme strech maximum
labelParam6.place(anchor="w", x=xS5+180,y=yS5+10)
echelle6 = Scale (cadre, from_ = 150, to = 255, command= choix_histo_max, orient=HORIZONTAL, length = 130, width = 10, resolution = 5, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle6.set(val_histo_max)
echelle6.place(anchor="w", x=xS5+210,y=yS5+10)

echelle8 = Scale (cadre, from_ = 1, to = 501, command= choix_nb_captures, orient=HORIZONTAL, length = 350, width = 10, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle8.set(val_nb_captures)
echelle8.place(anchor="w", x=xS8,y=yS8)

labelParam9 = Label (cadre, text = "Clip") # choix valeur contrate CLAHE
labelParam9.place(anchor="w", x=xS9,y=yS9+10)
echelle9 = Scale (cadre, from_ = 0.5, to = 4, command= choix_valeur_CLAHE, orient=HORIZONTAL, length = 200, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle9.set(val_contrast_CLAHE)
echelle9.place(anchor="w", x=xS9+30,y=yS9+10)

echelle10 = Scale (cadre, from_ = 5, to = 30, command= choix_nb_darks, orient=HORIZONTAL, length = 150, width = 10, resolution =1, label="",showvalue=1,tickinterval=5,sliderlength=20)
echelle10.set(val_nb_darks)
echelle10.place(anchor="w", x=xS10,y=yS10+10)

echelle11 = Scale (cadre, from_ = 0, to = 4000, command= choix_nb_video, orient=HORIZONTAL, length = 350, width = 10, resolution = 20, label="",showvalue=1,tickinterval=500,sliderlength=20)
echelle11.set(val_nb_capt_video)
echelle11.place(anchor="w", x=xS11,y=yS11-15)

labelParam12 = Label (cadre, text = "Pnt") # choix valeur histogramme Signoide param 1
labelParam12.place(anchor="w", x=xS12,y=yS12+10)
echelle12 = Scale (cadre, from_ = 0.5, to = 3, command= choix_phi, orient=HORIZONTAL, length = 130, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle12.set(val_phi)
echelle12.place(anchor="w", x=xS12+30,y=yS12+10)

labelParam13 = Label (cadre, text = "Dec") # choix valeur histogramme Signoide param 1
labelParam13.place(anchor="w", x=xS12+170,y=yS12+10)
echelle13 = Scale (cadre, from_ = 30, to = 200, command= choix_theta, orient=HORIZONTAL, length = 130, width = 10, resolution = 5, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle13.set(val_theta)
echelle13.place(anchor="w", x=xS12+210,y=yS12+10)

labelParam14 = Label (cadre, text = "R") # choix balance rouge
labelParam14.place(anchor="w", x=xS13-140,y=yS13+30)
echelle14 = Scale (cadre, from_ = 40, to = 160, command= choix_w_red, orient=HORIZONTAL, length = 140, width = 10, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle14.set(val_red)
echelle14.place(anchor="w", x=xS13-130,y=yS13+30)

labelParam65 = Label (cadre, text = "G") # choix balance rouge
labelParam65.place(anchor="w", x=xS13+20,y=yS13+30)
echelle65 = Scale (cadre, from_ = 40, to = 160, command= choix_w_green, orient=HORIZONTAL, length = 140, width = 10, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle65.set(val_green)
echelle65.place(anchor="w", x=xS13+30,y=yS13+30)

labelParam15 = Label (cadre, text = "B") # choix balance bleue
labelParam15.place(anchor="w", x=xS13+180,y=yS13+30)
echelle15 = Scale (cadre, from_ = 40, to = 160, command= choix_w_blue, orient=HORIZONTAL, length = 140, width = 10, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle15.set(val_blue)
echelle15.place(anchor="w", x=xS13+190,y=yS13+30)

labelParam16 = Label (cadre, text = "Peq2") # choix valeur histogramme eq 2
labelParam16.place(anchor="w", x=xS14+150,y=yS14+15)
echelle16 = Scale (cadre, from_ = 0.3, to = 4, command= choix_heq2, orient=HORIZONTAL, length = 250, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle16.set(val_heq2)
echelle16.place(anchor="w", x=xS14+190,y=yS14+15)

# Choix nombre de frames stackees
labelParam20 = Label (cadre, text = "# FS")
labelParam20.place(anchor="e", x=xS3-200,y=yS3-10)
echelle20 = Scale (cadre, from_ = 1, to = 3, command= choix_FS, orient=HORIZONTAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle20.set(val_FS)
echelle20.place(anchor="w", x=xS3-200,y=yS3-10)

# Choix Parametre Denoise KNN
labelParam30 = Label (cadre, text = "")
labelParam30.place(anchor="e", x=xCBEPF+90,y=yCBEPF-5)
echelle30 = Scale (cadre, from_ = 0.05, to = 1.5, command= choix_val_KNN, orient=HORIZONTAL, length = 120, width = 10, resolution = 0.05, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle30.set(val_denoise_KNN)
echelle30.place(anchor="w", x=xS4-30,y=yCBEPF-5+30)

# Choix parametre bandwidth USB
labelParam50 = Label (cadre, text = "USB")
labelParam50.place(anchor="w", x=xUSB-3,y=yUSB-130)
echelle50 = Scale (cadre, from_ = 0, to = 100, command= choix_USB, orient=VERTICAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle50.set(val_USB)
echelle50.place(anchor="c", x=xUSB,y=yUSB-60)

# Choix Parametre Seuil Gradient Removal
labelParam60 = Label (cadre, text = "")
labelParam60.place(anchor="e", x=xGR+100,y=yGR-5)
echelle60 = Scale (cadre, from_ = 0, to = 100, command= choix_SGR, orient=HORIZONTAL, length = 170, width = 10, resolution = 5, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle60.set(val_SGR)
echelle60.place(anchor="w", x=xGR+105,y=yGR-5)

# Choix Parametre Atenuation Gradient Removal
labelParam61 = Label (cadre, text = "A")
labelParam61.place(anchor="e", x=xGR+305,y=yGR-5)
echelle61 = Scale (cadre, from_ = 0, to = 100, command= choix_AGR, orient=HORIZONTAL, length = 170, width = 10, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle61.set(val_AGR)
echelle61.place(anchor="w", x=xGR+310,y=yGR-5)

# Choix Parametre Niveau Gaussian Blur Gradient Removal
labelParam62 = Label (cadre, text = "Niv Blur")
labelParam62.place(anchor="e", x=xGR+100,y=yGR+40)
echelle62 = Scale (cadre, from_ = 0, to = 50, command= choix_NGB, orient=HORIZONTAL, length = 170, width = 10, resolution = 1, label="",showvalue=1,tickinterval=10,sliderlength=20)
echelle62.set(val_NGB)
echelle62.place(anchor="w", x=xGR+105,y=yGR+40)

# Choix appliquer dark
CBAD = Checkbutton(cadre,text="Sub Dark", variable=choix_sub_dark,command=commande_sub_dark,onvalue = 1, offvalue = 0)
CBAD.place(anchor="w",x=xdark, y=ydark)

labelInfoDark = Label (cadre, text = dispo_dark) # label info Dark
labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)

labelInfo1 = Label (cadre, text = text_info1) # label info n°1
labelInfo1.place(anchor="w", x=xLI1,y=yLI1)

Button (fenetre_principale, text = "Cap Dark", command = start_cap_dark).place(x=1470,y=750, anchor="e")

Button (fenetre_principale, text = "Start Pic Cap", command = start_pic_capture).place(x=1470,y=785, anchor="e")
Button (fenetre_principale, text = "Stop Pic Cap", command = stop_pic_capture).place(x=1470,y=815, anchor="e")

Button (fenetre_principale, text = "REC Video", command = start_video_capture).place(x=1470,y=855, anchor="e")
Button (fenetre_principale, text = "Stop Video", command = stop_video_capture).place(x=1470,y=885, anchor="e")
#Button (fenetre_principale, text = "Pause Video", command = pause_video_capture).place(x=1470,y=915, anchor="e")

Button (fenetre_principale, text = "Quitter", command = quitter).place(x=1750,y=955, anchor="e")

cadre_image = Canvas (cadre, width = cam_displ_x, height = cam_displ_y, bg = "dark grey")
cadre_image.place(anchor="w", x=40,y=cam_displ_y/2+5)

cv2.setUseOptimized(True)

fenetre_principale.after(150, refresh)
fenetre_principale.mainloop()
fenetre_principale.destroy()






