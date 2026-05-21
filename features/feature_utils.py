import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    asm = graycoprops(glcm, 'ASM').mean()

    return [[
        contrast,
        homogeneity,
        energy,
        correlation,
        dissimilarity,
        asm
    ]]