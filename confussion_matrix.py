import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    confusion_matrix = np.array([
        [12, 0, 0, 0, 2],  # Obligación
        [0, 7, 0, 0, 1],   # Indicación
        [0, 0, 4, 0, 2],   # Peligro
        [0, 0, 0, 9, 0],   # Prohibición
        [1, 0, 0, 1, 0],   # Falsos positivos (Entorno detectado como señales)
    ])


    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', pad=20)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['Obligation', 'Indication', 'Danger', 'Prohibition', 'Environment'], rotation=45, ha='left')
    ax.set_yticklabels([''] + ['Obligation', 'Indication', 'Danger', 'Prohibition', 'Environment'])
    plt.xlabel('Predicted', labelpad=20)
    plt.ylabel('True Value', labelpad=20)

    # Adjust the space between labels
    plt.subplots_adjust(left=0.2, right=0.8, top=0.85, bottom=0.2)

    # Annotate each cell with the numeric value
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, str(confusion_matrix[i, j]), va='center', ha='center', color='black')

    plt.show()
