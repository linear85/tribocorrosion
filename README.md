# tribocorrosion
Accelerating the Design and Discovery of Tribocorrosion-Resistant Metals by Interfacing Multiphysics Modeling with Machine Learning and Genetic Algorithms 

Y Gu et al., npj Materials Degradation, Dec 2024, DOI : 10.1038/s41529-024-00549-4

1. This package contains Python files for tuning and training artificial neural network (ANN) models, along with genetic algorithm (GA) code. Additionally, it includes example data on strain, wear, and corrosion from multiphysics simulations.


2. Environment Setup

To run the codes, ensure the following libraries and their compatible versions are installed in your environment:

	Required Libraries

	TensorFlow: For deep learning tasks and neural network operations.
	-----------------------------------------
	pip install tensorflow
	-----------------------------------------

	Keras: Integrated with TensorFlow for building and managing neural networks.

	Keras Tuner: For hyperparameter optimization in deep learning models.
	-----------------------------------------
	pip install keras-tuner
	-----------------------------------------

	NumPy: For numerical computations and matrix operations.
	-----------------------------------------
	pip install numpy
	-----------------------------------------
	Pandas: For data analysis and manipulation.
	-----------------------------------------
	pip install pandas
	-----------------------------------------

	Matplotlib: For data visualization.
	-----------------------------------------
	pip install matplotlib
	-----------------------------------------

	Additional Notes

		Python Version: Ensure you have Python 3.7 or later installed.

		Virtual Environment: Using a virtual environment is recommended to manage dependencies. To set one up:
		-----------------------------------------
		python -m venv env
		source env/bin/activate    # On Windows, use `env\Scripts\activate`
		pip install --upgrade pip  # Upgrade pip to the latest version
		-----------------------------------------

		Install Dependencies: Once the virtual environment is activated, install all required packages:
		-----------------------------------------
		pip install tensorflow keras-tuner numpy pandas matplotlib
		-----------------------------------------

	Optional GPU Support: If you plan to use GPU acceleration, install the GPU-compatible version of TensorFlow along with CUDA and cuDNN. Refer to the TensorFlow GPU installation guide for detailed instructions.

3. To use this package:

(1) Run Corrosion_Model_HP_Tuning.py to tune hyperparameter (HP):

	Use the following input files one at a time:
		Surface_profile_principle_strain_data_correct_strain1.xlsx
		Surface_profile_principle_strain_data_correct_strain2.xlsx
		Surface_profile_principle_strain_data_correct_strain3.xlsx
		Surface_profile_principle_strain_data_correct_wear.xlsx
		Surface_profile_principle_strain_data_correct_Corr.xlsx	
	Update the input layer shape and output file names for each run, as the number of columns in each input file varies.

(2) Analyze HP Tuning Results:

	Use TensorBoard to review the HP tuning results.	
	
	Identify the top three desired HP combinations for each module:
		Strain Module 1
		Strain Module 2
		Strain Module 3
		Wear Module
		Corrosion Module

(3)Train Modules with Selected HP Combinations:

	Use Corrosion_Model_Training.py to train each module using the chosen HP combinations and corresponding .xlsx input files, one at a time.
	
	Ensure you update the input layer shape and output file names for each run.
	
	After this step, a total of 15 trained models (3 for each module) should be saved for subsequent G) processing.

(4) Run Genetic Algorithm (GA):

	Load the trained models into Corrosion_GA_Model.py.
	
	Adjust the GA parameters based on your specific requirements before running the algorithm.


Free Use Statement

	This project is provided for free use under the following conditions:

		The code and resources are available for educational, research, and non-commercial purposes.
		Proper attribution must be given to the original authors if the code or resources are used in publications, projects, or derivative works.
		Redistribution or modification of the code is allowed, provided that this statement is included in all copies or significant portions of the software.

Disclaimer: The project is provided "as is," without warranty of any kind, express or implied. The authors are not responsible for any damages or issues arising from the use of this software.
