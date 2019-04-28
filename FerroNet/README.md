Status: <b>active project</b><br>
<p align="justify">
Please feel free to suggest possible modifications of a workflow to better suite the microscopy/ferroelectric community needs.</p>

<h1>FerroNet</h1>
<p align="justify">
Application of different machine learning tools (neural networks, dimensionality reduction, clustering/unmixing) for analysis of ferroic distortions in high-resolution scanning transmission electron microscopy data from perovskites.</p><br>
<h2>How to use</h2>
<p align="justify">
This is a Colaboratory notebook centric package meaning that i) all the analysis is done in a Colab/Jupyter notebook, ii) the notebooks should be opened and executed in Google Colaboratory by clicking on 'Open in Colab' icon in the notebook (once opened, you can run and modify the notebooks without worrying about overwriting the source).</p>
<h4>Before you start:</h4>
<ul>
<li><p align="justify">Make sure you are logged in with your Google account (try to avoid using Internet Explorer or Microsoft Edge).</p></li>
<li><p align="justify">Remember that when working with a notebook, you need to run notebook cells from top to bottom (without skipping any cell) by selecting a cell and pressing Shift+Enter.</p></li>
</ul>
<h4>Input data format:</h4>
<ul>
<li><p align="justify">The input image data should be in a numpy format. If you have a preferred method of translating your experimental data to the numpy format, please use it. You can also make use of PyUSID/Pycroscopy translators. Finally, to make a quick translation of your data from .dm3 file format to the numpy, you may use the "dm3_to_numpy" notebook in this directory (click on "Open in Colab" and just follow the instructions in the notebook).</p></li>
</ul>
<h4>Analysis</h4>
<ul>
<li><p align="justify">Once you have your image in the numpy format, do the analysis using the "Test" notebook - just upload a file (it will ask you for it) and run each cell.</p></li>
<li><p align="justify">We recommend to start with the "FerroicBlocks" notebook and/or use uploaded experimental images of LBFO (already in numpy format) to get a better understanding of how the whole thing works. As long as your images are close in size and structure of the lattice to those in the examples, everything should work fine 😊 We also plan to add a more universal network in the near future.</p></li>
</ul>
