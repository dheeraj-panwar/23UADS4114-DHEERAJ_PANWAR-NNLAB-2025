## 1. Objective  
WAP to retrain a pretrained imagenet model to classify a medical image dataset.

## 2. Description of the Model  
This project leverages transfer learning using VGG16, a deep convolutional neural network pre-trained on ImageNet. Only the last convolutional block is fine-tuned, and a custom classifier is appended to the model for binary classification. The classifier comprises multiple fully connected layers with ReLU activation and dropout regularization, ending with a sigmoid activation function to output a probability.

### Model Architecture  
**Base model:** VGG16 (pretrained on ImageNet)  
**Frozen layers:** All layers except the last convolutional block  
**Custom classifier:**
```python
nn.Sequential(
    nn.Linear(512 * 7 * 7, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(128, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
```

## 4. Description of Code  
**Data Preparation**  
Data is located in `Dataset/` containing `CT_COVID` and `CT_NonCOVID` folders.  
Applied transformations include resizing, normalization, random horizontal flips, and rotations.

**Model Training**  
The model is trained for 20 epochs with a batch size of 32 and a learning rate of 1e-4.  
Training and validation sets are split in an 80:20 ratio.

**Training and Validation**  
Training involves forward propagation, loss computation, backpropagation, and weight updates.  
Validation is done at each epoch to monitor model generalization.

**Saving Model**  
The trained model weights are saved as `covid_classifier_vgg16.pt`.

**Visualization**  
Accuracy and loss curves for both training and validation phases are plotted.

## 5. Performance Evaluation  
**Final Test Accuracy:** ~90%  
**Loss and Accuracy Plots:** Training and validation loss/accuracy are plotted for visual inspection of model performance.

## 6. My Comments   
- **Overfitting Risk:** High training accuracy and lower validation accuracy may suggest overfitting. Use of regularization (dropout) is a can be useful.  
- **Model Complexity:** Using of some smaller models like EfficientNet might provide comparable accuracy with reduced computational cost.  
- **Binary Loss Function:** While BCELoss is standard, switching to BCEWithLogitsLoss (and removing sigmoid) can offer better numerical stability.  
- **Data Size:** Increasing dataset size can significantly improve performance.


