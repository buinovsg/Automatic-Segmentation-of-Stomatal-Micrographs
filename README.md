# Automatic Segmentation of Stomatal Micrographs

The measurement of stomatal features is a highly specialised task which has applications in many fields of research including plant physiological adaptations, morphological evolution and inference of paleoclimates based on stomatal conductance. The measurements are also widely used for the purpose of stomatal conductance modelling based on maximal stomatal conductance calculation. The manual measurement of the features, however, often proves to be laborious and time-consuming which significantly delays progress of research. The aim of this research project was to set preliminary groundwork for a tool which would automatically segment stomatal micrograph images and would generate stomatal measurements. U-net, a convolutional neural network for biomedical image segmentation was implemented for the task due to its advanced architecture which is able to be trained on fewer images whilst yielding a more precise segmentation.



### Data

Data available for training contains eleven stomatal micrograph images of Neea buxifolia (1391 x 1039, 200x magnification) which undergo data augmentation prior to model training.

### Implementation
The U-net implementation used can be found at https://github.com/zhixuhao/unet.

The implementation of training is available in a notebook format; data arrangement into folders for training purposes was kept same as outlined in https://github.com/zhixuhao/unet. Two out of the eleven available images was used for testing of the trained model.
