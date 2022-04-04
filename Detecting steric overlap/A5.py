# Assignment 5: Detecting steric overlap
# Author: Alejandro Corrochano

import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from statistics import mean

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

# calculate euclidean distance
def calculate_distance(pt1, pt2):
    return np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 +(pt2[2]-pt1[2])**2)

# Gets coordinates of atom
def get_coordinates(line):
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])

    return (x, y, z)

# Retrieves ATOM and HETATM data from PDB file
def read_data(protein_data):
    atom_array = [line for line in protein_data if line.split(' ')[0] == 'ATOM' or line.split(' ')[0] == 'HETATM']
    return atom_array

# Reads data from test files provided
def read_test_data(test_file):
    file = open(test_file)
    data = [line for line in file]
    collided_atoms = [int(line.split(' ')[0]) for line in data[:-2]]
    num_collisions = len(collided_atoms)
    num_comparisons = int(data[::-1][0].split(' ')[-1])
    file.close()
    return collided_atoms, num_collisions, num_comparisons

# Opens up the PDB file and returns relevant data
def read_proteins(prots):
    ps = []
    for p in prots:
        curated_protein = str(p + '.pdb')
        file = open(curated_protein)
        data = [line.rstrip('\n') for line in file]
        file.close()
        atom_data = read_data(data)
        ps.append(atom_data)

    return ps

# Get labels for randomly selected atoms for the training set
def get_labels(atoms, moving_chain, fixed_chain):
    labels = []
    comparisons = 0
    for indi, atomi in enumerate(atoms):
        collision = False
        for indj, atomj in enumerate(fixed_chain):
            coordsi, coordsj = get_coordinates(moving_chain[atomi]), get_coordinates(atomj)
            distance = calculate_distance(coordsi, coordsj)
            comparisons += 1
            if (distance-4) <= 0:
                labels.append(1)
                collision = True
                break

        if collision == False:
            labels.append(0)
        collision = False

    print('Percentage of maximum total number of comparisons employed: %.2f %% (%d)'
                        %(((comparisons/(len(moving_chain)*len(fixed_chain)))*100), comparisons))
    return labels, comparisons

#
def get_training_set(num_atoms, moving_chain, fixed_chain):
    random_atoms = np.random.randint(0, len(moving_chain)-1, size = num_atoms)

    # Get the coordinates from each atom
    coordinates = [get_coordinates(moving_chain[atom]) for atom in random_atoms]

    # Create dataframe with coordinates
    df = pd.DataFrame(np.array(coordinates), columns=['x', 'y', 'z'])
    labels, comparisons = get_labels(random_atoms, moving_chain, fixed_chain)
    df['label'] = labels

    return random_atoms, df, comparisons

# Train the model. Returns selected atoms and their labels, trained model, and number of comparisons.
def train_model(num_samples, moving, fixed):

    # Generate dataframe
    train_atoms, df, comparisons = get_training_set(num_samples, moving, fixed)

    # Build predictive model
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    #X_train, X_val, y_train, y_val = train_test_split(df[['x','y','z']].values, df['label'], test_size=0.2)
    cv_results = cross_validate(clf, df[['x','y','z']].values, df['label'], cv=5)
    print('Mean accuracy in cross validation:  %.2f' %mean(cv_results['test_score']))

    #Train the model
    clf.fit(df[['x','y','z']].values, df['label'])

    return train_atoms, df['label'], clf, comparisons

# Predict collisions for the remaining atoms in the moving chain.
def predict_collisions(moving_chain, model, train_atoms, train_labels):
    preds = []
    for ind, atom in enumerate(moving_chain):
        # If atom was used for the training, its label is appended to prediction list
        if ind in train_atoms:
            preds.append(train_labels[np.where(train_atoms == ind)[0][0]])
        else:
            coords = get_coordinates(atom)
            prediction = model.predict(np.array(coords).reshape(1, -1))
            preds.append(prediction[0])

    print('Approximated number of collisions:', sum(preds))
    return preds

def output_atoms(predictions, moving_chain, comparisons, args):
    output = [moving_chain[ind] for ind, pred in enumerate(predictions) if pred == 1]
    output.append('Number of collisions: ' + str(sum(predictions)))
    output.append('Number of comparisons needed: ' + str(comparisons))

    output_file_name = str('Output_'+ args[1] + '_' + args[2] +'.txt')
    f = open(output_file_name, "a")
    for row in output:
        f.write(row+'\n')
    f.close()

    return output_file_name

if __name__ == '__main__':
    args = sys.argv
    # Set fixed and moving chain
    fixed_chain, moving_chain = read_proteins([args[1], args[2]])
    # Train predictive model
    train_atoms, train_labels, model, comparisons = train_model(250, moving_chain, fixed_chain)
    #Predictions in remaning
    predictions = predict_collisions(moving_chain, model, train_atoms, train_labels)
    # Compare against test data provided
    if '1CDH' in args[1]:
        collided_atoms, num_collisions, test_comparisons = read_test_data('1CDH_2CSN.txt')
    else:
        collided_atoms, num_collisions, test_comparisons = read_test_data('2CSN_1CDH.txt')

    true_labels = [1 if atom in collided_atoms else 0 for atom in range(len(moving_chain))]

    print('Accuracy of predictions %.2f%%' %(accuracy_score(true_labels, predictions)*100))
    print('Original number of collisions: %d' %num_collisions)
    print('Difference in number of comparisons required: %.2f%%' %(((comparisons - test_comparisons)/test_comparisons)*100))

    # Create output file
    output_file = output_atoms(predictions, moving_chain, comparisons, args)
    print('File created with name:', output_file)
