
<h1>-Hybrid-Deep-Learning-Approaches-for-Bird-Species-Classification</h1>

<body>
<p align="center">
  <a href="mailto:arifmiahcse952@gmail.com"><img src="https://img.shields.io/badge/Email-arifmiah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/Arif-miad"><img src="https://img.shields.io/badge/GitHub-%40ArifMiah-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://www.linkedin.com/in/arif-miah-8751bb217/"><img src="https://img.shields.io/badge/LinkedIn-Arif%20Miah-blue?style=flat-square&logo=linkedin"></a>

 
  
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801998246254-green?style=flat-square&logo=whatsapp">
  
</p>



# Bird Species Classification Project

## Project Overview
This project utilizes deep learning models, specifically Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN), to classify six distinct bird species from the Bird Species Dataset. The aim is to explore the effectiveness of CNN and ANN architectures in identifying and categorizing bird species based on their unique visual characteristics. This project has applications in image classification, wildlife research, and ornithology studies.

## Dataset
The **Bird Species Dataset** contains images and metadata for six unique bird species, each with distinguishing visual and biological features. This diverse dataset allows for the training and testing of image classification models in the field of avian identification. Below is a brief overview of the species included:

- **American Goldfinch**: Recognizable by its vibrant yellow feathers, this small bird is native to North America and commonly seen in fields and meadows.
- **Barn Owl**: Known for its heart-shaped face, this medium-sized owl is found worldwide, often inhabiting farmlands and woodlands.
- **Carmine Bee-eater**: A strikingly colorful bird from Africa, noted for its reddish-pink feathers and diet of flying insects.
- **Downy Woodpecker**: The smallest woodpecker in North America, identifiable by its black-and-white feather pattern and characteristic wood-pecking behavior.
- **Emperor Penguin**: Native to Antarctica, this is the largest species of penguin, known for its resilience to extreme cold and distinct black, white, and yellow coloring.
- **Flamingo**: Famous for its pink or reddish plumage, this wading bird is often found in lagoons and shallow lakes.

## Table of Contents
- **Import Libraries**: Loading the necessary Python libraries.
- **Data Preprocessing**: Cleaning and preparing the dataset for model training.
- **Data Loading**: Loading and organizing the dataset for analysis.
- **Visualize Bird Species**: Displaying sample images from each bird species for data exploration.
```python
# Set the number of images to display per species
num_images = 6

# Get unique bird species labels
bird_species = df['labels'].unique()

# Set up the plot
plt.figure(figsize=(20, 20))

# Loop through each bird species
for idx, bird in enumerate(bird_species):
    # Filter the DataFrame to get file paths for this bird species
    bird_df = df[df['labels'] == bird].sample(num_images)  # Get a random sample of 16 images
    
    # Loop through the 16 images and plot them
    for i, file in enumerate(bird_df['filepaths'].values):
        plt.subplot(len(bird_species), num_images, idx * num_images + i + 1)
        img = Image.open(file)
        plt.imshow(img)
        plt.axis('off')
        plt.title(bird)
        
# Show the plot
plt.tight_layout()
plt.show()

```
![](https://github.com/Arif-miad/-Hybrid-Deep-Learning-Approaches-for-Bird-Species-Classification/blob/main/b9.png)

- **Train-Test Split**: Dividing the data into training and testing sets.
- **Create Image Data Generator**: Utilizing data augmentation for improved model generalization.
- **Build ANN Model**: Constructing an Artificial Neural Network model for classification.
- **Build CNN Model**: Constructing a Convolutional Neural Network model for enhanced image recognition.
- **Display Model Performance**: Evaluating and comparing model accuracy, loss, and other metrics.
```python
# Plot Training and Validation Loss
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_loss, 'purple', label= 'Training loss')
plt.plot(Epochs, val_loss, 'gold', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'darkblue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout
plt.show()
```
![](https://github.com/Arif-miad/-Hybrid-Deep-Learning-Approaches-for-Bird-Species-Classification/blob/main/b19.png)

```python
# Plot Training and Validation Accuracy
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_acc, 'purple', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'gold', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'darkblue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show()
```
![](https://github.com/Arif-miad/-Hybrid-Deep-Learning-Approaches-for-Bird-Species-Classification/blob/main/b20.png)
```python
# Plot Training and Validation Loss
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_loss, 'purple', label= 'Training loss')
plt.plot(Epochs, val_loss, 'gold', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'darkblue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout
plt.show()
```

![](https://github.com/Arif-miad/-Hybrid-Deep-Learning-Approaches-for-Bird-Species-Classification/blob/main/b77.png)

```python
# Plot Training and Validation Accuracy
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_acc, 'purple', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'gold', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'darkblue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show()
```
![](https://github.com/Arif-miad/-Hybrid-Deep-Learning-Approaches-for-Bird-Species-Classification/blob/main/b78.png)


- **Get Predictions**: Making predictions on test data with each model.
- **Model Evaluation**: Analyzing model performance with metrics like accuracy, precision, and recall.
```python
# Plotting image to compare
img = array_to_img(images[5])
img
```
![](https://github.com/Arif-miad/-Hybrid-Deep-Learning-Approaches-for-Bird-Species-Classification/blob/main/b149.png)


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bird-species-classification.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the project by following these steps:
1. Preprocess the data.
2. Split the data into training and testing sets.
3. Train the ANN and CNN models.
4. Evaluate and compare model performances.

## Results
This section should summarize the performance results for both the CNN and ANN models. Include any charts or tables of accuracy, confusion matrices, and other relevant metrics to showcase the model evaluation outcomes.

## Future Work
- Enhance model accuracy by experimenting with other deep learning architectures.
- Integrate additional bird species for a more comprehensive classification model.
- Explore real-time bird species classification using the trained model.

## Contributing
Please feel free to submit issues or pull requests. Contributions to improve the project are always welcome!

## License
This project is licensed under the MIT License.

