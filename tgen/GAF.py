#!/home/adriano/Escritorio/TFG/venv/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt
import tgen.activity_data as act
import tgen.recurrence_plots as rec
#import activity_data as act
#import recurrence_plots as rec
from PIL import Image
import cv2
import time
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
import os
def normalize_to_zero_one(series):
    """
    Normaliza una serie temporal al rango [0, 1].
    
    Args:
    series (np.array): Serie temporal a normalizar.
    
    Returns:
    np.array: Serie temporal normalizada al rango [0, 1].
    """
    min_val = np.min(series)
    max_val = np.max(series)
    normalized_series = (series - min_val) / (max_val - min_val)
    normalized_series = np.where(normalized_series >= 1., 1., normalized_series)
    normalized_series = np.where(normalized_series <= 0., 0., normalized_series)
    return normalized_series
def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))
def sin_diff(a, b):
    """To work with tabulate."""
    return(math.sin(a-b))
def normalize_series(series):
    min_val = np.min(series)
    max_val = np.max(series)
    # Floating point inaccuracy!
    scaled_serie =2 * (series - min_val) / (max_val - min_val) - 1
    scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)
    return scaled_serie

def gramian_angular_field(series, method='summation'):
    # Normalización de la serie temporal
    normalized_series = normalize_to_zero_one(series)
    #print("SERIE_normalizada_R",normalized_series)
    # Conversión a ángulos
    # Polar encoding
    #print("Serie norm",series,normalized_series)
    phi = np.arccos(normalized_series) 
    
    #print("Serie norm",phi,normalized_series)
    #print("Angulos",phi)
    #print("valores de PHI de R",phi)
    """
    Para la reconstruccion nos servirá
    r = np.linspace(0, 1, len(normalized_series))
    print("Radius",r)
    """
    # Construcción del GAF
    if method == 'summation':
        gaf = tabulate(phi, phi, cos_sum)
        
    elif method == 'difference':
        gaf = tabulate(phi, phi, sin_diff)
    else:
        raise ValueError("Method must be 'summation' or 'difference'")
    
    return gaf
    

def varGAF(data, dim,TIME_STEPS):
    x = []
    k=0
    if dim == 'x':
        k=0
    elif dim == 'y':
        k=1
    elif dim == 'z': 
        k=2
    
    x=np.array(data[:,k]).reshape(1,-1)
    
    X_gaf = gramian_angular_field(x)
    #print(X_gaf)
    
    return X_gaf

def RGBfromGAFMatrix_of_XYZ(X,Y,Z):
    if X.shape != Y.shape or X.shape != Z.shape or Y.shape != Z.shape:
        print('XYZ should be in same shape!')
        return 0
    #print(X.shape)
    dimImage = X.shape[0]
    newImage = np.zeros((dimImage,dimImage,3))
    for i in range(dimImage):
        for j in range(dimImage):
            _pixel = []
            _pixel.append(X[i][j])
            _pixel.append(Y[i][j])
            _pixel.append(Z[i][j])
            newImage[i][j] = _pixel
    return newImage


def SavevarGAF_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    if not all([(x==0).all()]):
     _r = varGAF(x,'x', TIME_STEPS)
     _g = varGAF(x,'y', TIME_STEPS)
     _b = varGAF(x,'z', TIME_STEPS)

     
     #print("Y", _g[1][4])
     #print("Z", _b[1][4])
     #print("Y", _g)
     #print("Z", _b)
     
     # plt.close('all')
     # plt.figure(figsize=(1,1))
     # plt.axis('off')
     # plt.margins(0,0)
     # plt.gca().xaxis.set_major_locator(plt.NullLocator())
     # plt.gca().yaxis.set_major_locator(plt.NullLocator())

     #print("fig size: width=", plt.figure().get_figwidth(), "height=", plt.figure().get_figheight())
    
     if normalized:
          #print(_r)
          newImage = RGBfromGAFMatrix_of_XYZ(normalize_to_zero_one(_r), normalize_to_zero_one(_g),normalize_to_zero_one(_b))
          #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          # print(newImage.shape)
          #print(newImage[1][4][0]* 255)
          
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
          # plt.imshow(newImage)
          
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
               newImage.save(f"{path}{sj}{action}{item_idx}gasf.png")
          # plt.close('all')
     else:
          newImage = RGBfromGAFMatrix_of_XYZ((_r+1)/2, (_g+1)/2,(_b+1)/2)
          newImage = Image.fromarray((newImage * 255).astype(np.uint8))
          # plt.imshow(newImage)
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure') #dpi='figure' for preserve the correct pixel size (TIMESTEPS x TIMESTEPS)
               newImage.save(f"{path}{sj}{action}{item_idx}gasf.png")
          # plt.close('all')
    
     return newImage
    
    else:
     return None
    
   
def calcular_errores(valores_verdaderos, valores_aproximados):
    # Convertir las listas a arrays de numpy para facilitar los cálculos
    valores_verdaderos = np.array(valores_verdaderos)
    valores_aproximados = np.array(valores_aproximados)
    
    # Calcular el error absoluto
    errores_absolutos = np.abs(valores_verdaderos - valores_aproximados)
    
    # Calcular el error relativo (evitando la división por cero)
    errores_relativos = np.abs(errores_absolutos / valores_verdaderos)
    
    # Calcular el error absoluto promedio
    error_absoluto_promedio = np.mean(errores_absolutos)
    
    # Calcular el error relativo promedio
    error_relativo_promedio = np.mean(errores_relativos)
    
    return error_absoluto_promedio, error_relativo_promedio
def Reconstruct_GAF(img,dictionary,a):
    _r= img[:,:,0].astype('float')
    _g= img[:,:,1].astype('float')
    _b= img[:,:,2].astype('float')
    
    _r=np.interp(_r,(0,255),(-1,1))
    _g=np.interp(_g,(0,255),(-1,1))
    _b=np.interp(_b,(0,255),(-1,1))
    
    r=GAF_to_TS(_r,0,dictionary[a])
    g=GAF_to_TS(_g,1,dictionary[a])
    b=GAF_to_TS(_b,2,dictionary[a])
    N=[]
    N.append(r)
    N.append(g)
    N.append(b)
    return N
def GAF_to_TS(gaf,i,dictionary,method='summation'):
    #We recontruct first phi vector based on gasf matrix diagonal property
    
    phi=np.arccos(np.diag(gaf))/2
    
    X_normalized=np.cos(phi)
    
    
    #RECONSTRUCCIÓN DISCRETA DE LOS ESTADOS:
    #Sabemos que el valor medio de los datos maximos es 
    MAX=dictionary[0] 
    MIN=dictionary[1]
    

    #x=np.interp(X_normalized,(np.min(X_normalized),np.max(X_normalized)),(MIN[i],MAX[i]))
    x=X_normalized
       
     

    # Transformar los puntos
     #posi =min_b + n * (max_b - min_b)
    x=np.interp(x,(np.min(x),np.max(x)),(MIN[i], MAX[i])) 

    return x
