import pandas as pd
import numpy as np
from PIL import Image
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt
from utils.perimetric_complexity import perimetricComplexity
from utils.complexity_metrics import calculate_stroke_count, pixel_count_complexity, apply_distance_transform, num_components
from utils.unicode_utils import *
from utils.perimetric_complexity import perimetricComplexity
from utils.calculate_perimeter_lengths import calculate_perimeter_lengths
from utils.complexity_metrics import *

smiles_file_path = '/Users/tiananoll-walker/Documents/bmsis/alphabets_code/alphabet_complexity-main/alphabet_complexity/Coded_AAs.txt'
with open(smiles_file_path, 'r') as f:
    lines = f.readlines()

smiles_aas_path = '/Users/tiananoll-walker/Documents/bmsis/alphabets_code/alphabet_complexity-main/alphabet_complexity/data/SMILESforAAs.txt'
with open(smiles_aas_path, 'r') as f:
    smiles_aas = [line.strip() for line in f.readlines()]

amino_acid_prevalence = {
    'Alanine': 0.074,
    'Arginine': 0.042,
    'Asparagine': 0.044,
    'Aspartic_acid': 0.059,
    'Cysteine': 0.033,
    'Glutamic_acid': 0.058,
    'Glutamine': 0.037,
    'Glycine': 0.074,
    'Histidine': 0.029,
    'Isoleucine': 0.038,
    'Leucine': 0.076,
    'Lysine': 0.072,
    'Methionine': 0.018,
    'Phenylalanine': 0.040,
    'Proline': 0.050,
    'Serine': 0.081,
    'Threonine': 0.062,
    'Tryptophan': 0.013,
    'Tyrosine': 0.033,
    'Valine': 0.068
}    

def calculate_perimeter_lengths(picture, threshold=0.001):
    picture_pil = Image.fromarray(picture)
    gray_image = picture_pil.convert('L')
    picture_np = np.array(gray_image)
    binarized_image = (picture_np > threshold).astype(np.uint8) * 255
    contours = measure.find_contours(binarized_image, 0.5)
    stroke_lengths = [len(contour) for contour in contours]
    return stroke_lengths

# Generate images for specific indices
def generate_molecule_image(molecule):
    mol_block = Chem.MolToMolBlock(molecule)
    image = Draw.MolToImage(molecule, size=(300, 300))
    return image


smiles = []  
is_biological = []
names = []
aa_prevalence = []

for line in lines:
    parts = line.strip().split(',')
    if len(parts) >= 2:
        name = parts[0].strip() 
        smiles_str = parts[1].strip()  
    
        if name in amino_acid_prevalence:  
            is_bio = True
            prevalence = amino_acid_prevalence[name]
        else:
            is_bio = False
            prevalence = None
    
        names.append(name)
        smiles.append(smiles_str)
        aa_prevalence.append(prevalence)
        is_biological.append(is_bio)

for smiles_str in smiles_aas:
    smiles.append(smiles_str)
    is_biological.append(False)
    names.append(None) 
    aa_prevalence.append(None)


data = {'Name':names,'SMILES': smiles, 'Is_Biological': is_biological, 'Prevalence': aa_prevalence}
df_molecules = pd.DataFrame(data)


perimetric_complexities = []
compression_sizes = []
pixel_counts = []
num_components_list = []
distance_transform_means = []
distance_transform_stds = []
perimeter_lengths = []
num_perimeters = []
symmetry_horizontal_values = []
symmetry_vertical_values = []

def rowtobin(row):
    white = [255,255,255]
    return [1 - int(np.array_equiv(x, white)) for x in row]

def complexity(imarr):
    perimetric_complexity = perimetricComplexity(imarr)
    pixel_count = pixel_count_complexity(imarr)
    num_components_value = num_components(imarr)
    distance_transform = apply_distance_transform(imarr)
    distance_transform_mean = np.mean(distance_transform)
    distance_transform_std = np.std(distance_transform)
    stroke_lengths = calculate_perimeter_lengths(imarr)
    check_symm_h = check_symmetry_nw(imarr, type = "h")
    check_symm_v = check_symmetry_nw(imarr, type = "v")
    
    return np.array([perimetric_complexity, pixel_count, num_components_value, distance_transform_mean, distance_transform_std, check_symm_h, check_symm_v])

for smiles_str in df_molecules['SMILES']:
    molecule = Chem.MolFromSmiles(smiles_str)
    image = generate_molecule_image(molecule)
    img_data = np.array(image)

    bw_image = np.array([rowtobin(row) for row in img_data])
    bw_image = bw_image.astype(np.uint8)


    # Calculate complexity metrics using smitopic functions
    complexity_metrics = complexity(bw_image)

    perimetric_complexity = complexity_metrics[0]
    pixel_count = complexity_metrics[1]
    num_components_value = complexity_metrics[2]
    distance_transform_mean = complexity_metrics[3]
    distance_transform_std = complexity_metrics[4]
    check_symm_h = complexity_metrics[5]
    check_symm_v = complexity_metrics[6]

    stroke_lengths = calculate_perimeter_lengths(img_data)

    
    perimetric_complexities.append(perimetric_complexity)
    pixel_counts.append(pixel_count)
    num_components_list.append(num_components_value)
    distance_transform_means.append(distance_transform_mean)
    distance_transform_stds.append(distance_transform_std)
    perimeter_lengths.append(stroke_lengths)
    num_perimeters.append(len(stroke_lengths))
    symmetry_horizontal_values.append(check_symm_h)
    symmetry_vertical_values.append(check_symm_h)

df_molecules['Perimetric_Complexity'] = perimetric_complexities
df_molecules['Pixel_Count'] = pixel_counts
df_molecules['Num_Components'] = num_components_list
df_molecules['Distance_Transform_Mean'] = distance_transform_means
df_molecules['Distance_Transform_Std'] = distance_transform_stds
df_molecules['Perimeter_Length'] = perimeter_lengths
df_molecules['Num_Perimeters'] = num_perimeters
df_molecules['Symmetry_Horizontal'] = symmetry_horizontal_values
df_molecules['Symmetry_Vertical'] = symmetry_vertical_values

print(df_molecules)

excel_file_path = 'molecules_data_rdkit.xlsx'
df_molecules.to_excel(excel_file_path, index=False, float_format='%.3f')
