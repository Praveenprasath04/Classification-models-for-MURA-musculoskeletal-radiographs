Creating Workflow structure.

Run template.py inorder to create workflow structure (template.py is a general template for implementing any model).

Requirements.txt :  
Any librarires wished to be use in model should included in this file. -e. signifies project module (refer setup.py)
TO install requirements.txt  use.
pip install -r requirements.txt

Setup.py:  
setup.py is a module used to build and distribute Python packages. It typically contains information about the package, such as its name, version, and dependencies, as well as instructions for building and installing the package. This information is used by the pip tool, which is a package manager for Python that allows users to install and manage Python packages from the command line. By running the setup.py file with the pip tool, you can build and distribute your Python package so that others can use it. 
We are using here inorder to export src/project as a module to use in our project and export to any other project.

to install any function from scripts in src/project in any scripts or notebook 

[from project.{scipt_name} import {function_name}]

Custom Logging :
Inside the project module. we have added a custom logger function which if used logs the action in logs file
Inorder to use the Logger funtion 

[from project import logger]

Utilis : 
Utils can be used to store utility function which are used frequently in various modules.


Pipeline setup

Pipelines for the model should be update via following the following steps.

Workflows

Update config.yaml   
Update params.yaml  
Update the entity  
Update the configuration manager in src config  
Update the components  
Update the pipeline  
Update the main.py  
Update the dvc.yaml

Example: Data_ingestion

Ensure to have the zip file of data "MURA-v1.1.zip" on workflow.

step 1: Update config.yaml.

In config\config.yaml. add directory for artifacts folder which will be used for datastorage, training _model storage..etc which not needed to be pushed to git repo (add aftifacts to .gitignore).Create a seperate section name data_ingestion and store every unique metrics needed for data_ingestion.





