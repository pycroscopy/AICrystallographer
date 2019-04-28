Status: <b>active project</b><br>
<p align="justify">
Please feel free to suggest possible modifications of a workflow to better suite the microscopy/materials community needs.</p>

# DefectNet
<p align="justify">
Complete workflow for locating atomic defects in electron microscopy movies using only a single movie frame for generation of a training set. This is based on the idea described originally in our paper npj Computational Materials 5, 12 (2019), but now with the updated augmentation procedure (includes adding noise, zoom-in and flip/rotations) and using PyTorch instead of Keras for model training/predictions. See notebook DefectSniffer.ipynb, which should be opened and executed in Google Colab. The augmentation procedure introduced here should be applicable to other types of data (including manually labeled image-groundtruth pairs).<br><br>
<p align="center">
  <img src="https://github.com/pycroscopy/AICrystallographer/blob/master/DefectNet/DefectNet.jpg" width="75%" title="DefectNet">
<p align="justify">
<br><br>
We believe that this particular approach can help extracting information on the point defects evolution from the long electron microscopy movies (more than 1000 frames) from different materials. Currently it accepts inputs in the form of 3D numpy arrays (stack of images). We plan to incorporate Pycroscopy translators in the near future to make it posssible working directly with .dm3 and other popular file formats.
